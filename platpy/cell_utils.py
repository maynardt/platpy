"""
cell_utils.py -- Tools for building single-cell AnnData objects from spatial data.

Two functions with a clear split of responsibility:

  build_cells_xenium(sample)
      Xenium-specific fast path.  Xenium assigns every transcript to a cell at
      instrument time, so we just pivot the transcript table — no spatial
      containment test needed.

  make_cells_from_masks(omic_channel, boundaries, min_obs=1)
      Platform-agnostic.  Given a list of cell boundary polygons (e.g. from
      Cellpose) and any OmicChannel with obsm['spatial'] coordinates, aggregate
      the observations inside each boundary into one row per cell.  Works on
      Visium spots, Xenium transcripts, or any future platform.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad


def _encode_coord(val_um, precision=10):
    """Encode a µm coordinate as letters: each decimal digit → a–j (0→a, 9→j).
    Negative values get an 'n' prefix. Cell at (452.3, 1234.5) µm → 'efcd_bcded'.
    Coordinates are stored in obs metadata so no decoder is needed."""
    sign = 'n' if val_um < 0 else ''
    digits = str(int(abs(val_um) * precision))
    return sign + ''.join(chr(ord('a') + int(d)) for d in digits)


# =============================================================================
# Xenium-specific
# =============================================================================

# Bumped when build_cells_xenium()'s algorithm changes (binning rule, centroid
# source, dropped-codeword pattern, etc). Stamped onto every AnnData this
# builds (adata.uns['build_cells_xenium_version']) so a save-time cache from
# an older platpy is never silently trusted as if it were current -- see
# save_sample()'s build_default_cells handling in io_utils.py.
BUILD_CELLS_XENIUM_VERSION = 3


def build_cells_xenium(sample, roi=None, discard_straddlers: bool = True,
                        force_recompute: bool = False, verbose: bool = True):
    """
    Build a cells x genes AnnData from Xenium transcript data.

    Uses the cell assignments made by the Xenium instrument (cell_id column).
    Unassigned transcripts (cell_id 0, -1, or 'UNASSIGNED') are dropped.

    Cell centroid and area come from the instrument's own segmented polygon
    (sample.cells['cell_boundaries'].summary, derived at load time from the
    cell_boundaries.parquet) rather than the transcripts -- the true cell
    shape/centroid, not an approximation from wherever its transcripts happened
    to land. Falls back to a transcript-mean centroid (with no area) only for
    bundles saved before that summary existed.

    save_sample() caches this result on sample.cells['cell_boundaries'].data
    at save time (the expensive part -- loading/processing the raw data --
    already happened once then, so this is a small marginal cost). By default
    this function returns that cache directly rather than recomputing, as long
    as it was stamped with the current BUILD_CELLS_XENIUM_VERSION -- the cache
    is a convenience, not the only path: the raw transcripts stay available
    regardless, so pass force_recompute=True for a different cell definition
    (different binning, boundary source, QC thresholds, etc).

    Parameters
    ----------
    sample : SampleSet
        Loaded Xenium sample.  Must have sample.omics['transcripts'] with obs
        columns: cell_id, feature, x, y.
    roi : (min_x_um, max_x_um, min_y_um, max_y_um) or None
        Restrict to transcripts in this bbox -- for a lazily-loaded sample
        (see lazy_transcripts.py), only the overlapping spatial blocks are
        ever materialized, not the whole section. No margin/buffer around
        the box (unlike extract_roi()'s cell_margin_um): a cell whose body
        straddles the boundary only gets counted for the transcripts that
        landed inside the box, so its gene counts come out undercounted
        relative to a full-bundle build -- its centroid/area (from the real
        instrument polygon, unaffected by this) can also end up just outside
        the box for the same reason. Fine for "give me approximate cells in
        this region" use; not a substitute for extract_roi() + a full build
        when exact per-cell counts at the edges matter. Disables the
        save-time cache (which covers the whole bundle, not a sub-region) --
        force_recompute is implied whenever roi is given.
    discard_straddlers : bool
        Only meaningful when roi is given. True (default): drop any cell
        whose boundary polygon isn't *entirely* inside the roi -- its full
        vertex set, not just its centroid (a cell can have an in-bounds
        centroid while still having vertices, and therefore transcripts,
        outside the box). The alternative to dropping it is keeping it with
        an undercounted expression profile, which is worse for most
        analysis uses than just not having the cell. False keeps every cell
        with any transcript in the roi, undercounted edges and all -- only
        useful if you specifically want that tradeoff. Requires
        sample.cells['cell_boundaries'].boundaries (the real per-vertex
        polygon table); falls back to keeping straddlers (with a printed
        notice when verbose) if that channel isn't present, since there's
        nothing to test against.
    force_recompute : bool
        Skip the save-time cache (if any) and recompute from the raw
        transcripts every time.
    verbose : bool
        Print progress/diagnostics. save_sample() passes False to keep an
        automatic save-time build quiet.

    Returns
    -------
    AnnData  (n_cells x n_genes)
        obs : sample_id, x_micron, y_micron, area_um2 (if available),
              n_transcripts, n_genes
        obsm['spatial'] : (n_cells, 2) centroid coordinates in microns
    """
    from .lazy_value import resolve as _resolve_lazy

    boundary_chan = sample.cells.get('cell_boundaries')
    # Only resolve the cached .data if it's actually going to be used -- a
    # plain `boundary_chan.data is not None` check would already force the
    # resolve as a side effect of the attribute access itself (CellChannel's
    # __getattribute__ resolves transparently on *any* read), which would
    # defeat the point of it being deferred for exactly the roi=/
    # force_recompute=True calls that were never going to touch it. Peek the
    # raw stored value via object.__getattribute__ (bypassing that
    # transparency on purpose, just for this existence check) instead --
    # True for either an unresolved Deferred or an already-resolved AnnData,
    # same as `is not None` would have meant, just without forcing the read.
    has_cached_data = (boundary_chan is not None
                        and object.__getattribute__(boundary_chan, 'data') is not None)
    if roi is None and not force_recompute and has_cached_data:
        cached_data = boundary_chan.data  # intentional resolve -- we're using it now
        cached_version = cached_data.uns.get('build_cells_xenium_version')
        if cached_version == BUILD_CELLS_XENIUM_VERSION:
            if verbose:
                print(f"Using cell x gene matrix cached at save time "
                      f"({cached_data.n_obs:,} cells x {cached_data.n_vars:,} genes).")
            return cached_data
        elif verbose:
            print(f"Cached cell x gene matrix is from an older platpy "
                  f"(version {cached_version}, current {BUILD_CELLS_XENIUM_VERSION}) -- recomputing.")

    # Pull only the columns we need to keep memory footprint small
    from .lazy_transcripts import LazyTranscripts
    tx_data = sample.omics['transcripts'].data

    if isinstance(tx_data, LazyTranscripts):
        if roi is not None:
            min_x, max_x, min_y, max_y = roi
            hit_ranges = tx_data.row_ranges_for_block_index(min_x, max_x, min_y, max_y)
            tx = tx_data.materialize(hit_ranges)
            tx = tx[(tx['x'] >= min_x) & (tx['x'] <= max_x) &
                    (tx['y'] >= min_y) & (tx['y'] <= max_y)]
        else:
            tx = tx_data.materialize()
        select_cols = ['cell_id', 'feature', 'x', 'y']
        if 'is_gene' in tx.columns:
            select_cols.append('is_gene')
        tx = tx[select_cols].copy()
    else:
        # Non-lazy fallback: tx_data is a plain DataFrame for a freshly
        # load_xenium()'d sample not yet saved (platpy's ingestion no longer
        # wraps transcripts in AnnData until save_sample() does, right before
        # write), or an AnnData for an older in-memory object / pre-this-fix
        # bundle -- hasattr distinguishes them the same way elsewhere in this
        # codebase handles the same AnnData-or-DataFrame ambiguity.
        tx_src = tx_data.obs if hasattr(tx_data, 'obs') else tx_data
        select_cols = ['cell_id', 'feature', 'x', 'y']
        if 'is_gene' in tx_src.columns:
            select_cols.append('is_gene')
        tx = tx_src[select_cols].copy()
        if roi is not None:
            min_x, max_x, min_y, max_y = roi
            tx = tx[(tx['x'] >= min_x) & (tx['x'] <= max_x) &
                    (tx['y'] >= min_y) & (tx['y'] <= max_y)]

    # Drop unassigned transcripts — Xenium uses 0 (int) or 'UNASSIGNED' (str)
    tx = tx[~tx['cell_id'].astype(str).isin(['0', '-1', 'UNASSIGNED'])]
    tx['cell_id'] = tx['cell_id'].astype(str)

    if roi is not None and discard_straddlers:
        boundary_chan = sample.cells.get('cell_boundaries')
        if boundary_chan is None or boundary_chan.boundaries is None:
            if verbose:
                print("discard_straddlers=True requested but no cell_boundaries "
                      "channel is present -- keeping straddler cells (undercounted).")
        else:
            min_x, max_x, min_y, max_y = roi
            b = _resolve_lazy(boundary_chan.boundaries)
            b_obs = b.obs if hasattr(b, 'obs') else b
            # A cell's full polygon is inside the roi only if EVERY vertex is --
            # a vectorized per-cell min/max is equivalent to that (and far
            # cheaper than a groupby().apply() over every vertex) since "all
            # vertices within [min_x,max_x]x[min_y,max_y]" <=> "the vertex
            # bounding box is within it".
            vbounds = b_obs.groupby('cell_id', observed=True).agg(
                vx_min=('x', 'min'), vx_max=('x', 'max'),
                vy_min=('y', 'min'), vy_max=('y', 'max'),
            )
            whole = vbounds[
                (vbounds['vx_min'] >= min_x) & (vbounds['vx_max'] <= max_x) &
                (vbounds['vy_min'] >= min_y) & (vbounds['vy_max'] <= max_y)
            ]
            keep_ids = set(whole.index.astype(str))
            n_before = tx['cell_id'].nunique()
            tx = tx[tx['cell_id'].isin(keep_ids)]
            if verbose:
                n_after = tx['cell_id'].nunique()
                print(f"Discarded {n_before - n_after:,} straddler cell(s) "
                      f"(boundary polygon not entirely inside roi) of {n_before:,}.")

    # Drop non-gene transcripts (negative/genomic controls, deprecated/
    # unassigned codewords). Prefer the raw is_gene flag when present --
    # confirmed 2026-10-23 against a real 147M-row file to be a correct,
    # complete ground-truth (matches codeword_category == 'predesigned_gene'
    # with zero crossover). Older bundles saved before is_gene was read fall
    # back to the previous name-regex filter, which has a known gap: it only
    # catches "...Codeword"-suffixed names, missing the "...Probe"-suffixed
    # negative_control_probe/genomic_control_probe categories entirely --
    # reingest (with is_gene present) to get the complete filter instead.
    if 'is_gene' in tx.columns:
        non_gene_mask = ~tx['is_gene'].astype(bool)
    else:
        non_gene_mask = tx['feature'].str.contains(
            r'^(?:NegControl|Unassigned|Deprecated|BLANK)Codeword', regex=True
        )
    n_codeword = non_gene_mask.sum()
    if n_codeword > 0:
        tx = tx[~non_gene_mask]
        if verbose:
            print(f"Dropped {n_codeword:,} non-gene transcripts "
                  f"(negative/genomic controls, deprecated/unassigned codewords)")

    if verbose:
        print(f"Transcripts assigned to cells: {len(tx):,}")

    # Sorted lists set the row/column order of the matrix
    cells = sorted(tx['cell_id'].unique())
    genes = sorted(tx['feature'].unique())
    n_cells, n_genes = len(cells), len(genes)
    if verbose:
        print(f"  Cells: {n_cells:,}   Genes: {n_genes:,}")

    cell_idx = {c: i for i, c in enumerate(cells)}
    gene_idx = {g: i for i, g in enumerate(genes)}

    # int32 rather than pandas' default int64 for these -- cell/gene counts
    # are nowhere near int32's ~2.1 billion ceiling for any realistic single
    # section, and indices/indptr dominate a sparse matrix's memory footprint
    # (bigger than the actual float32 data for a typical count matrix), so
    # this roughly halves the matrix's size for free.
    rows = tx['cell_id'].map(cell_idx).values.astype(np.int32)
    cols = tx['feature'].map(gene_idx).values.astype(np.int32)

    # One transcript = one count; sparse to keep memory low
    X = sp.csr_matrix(
        (np.ones(len(tx), dtype=np.float32), (rows, cols)),
        shape=(n_cells, n_genes)
    )

    boundary_summary = boundary_chan.summary if boundary_chan is not None else None

    if boundary_summary is not None:
        bs = boundary_summary.copy()
        bs.index = bs.index.astype(str)
        centroids = bs.reindex(cells)[['x', 'y', 'area_um2']]
        area_um2 = centroids['area_um2'].values
    else:
        if verbose:
            print("No cell_boundaries summary found -- falling back to transcript-mean "
                  "centroid (no polygon area available). Re-extract this bundle with a "
                  "current platpy to get true instrument-segmented centroids.")
        centroids = tx.groupby('cell_id')[['x', 'y']].mean().loc[cells]
        area_um2 = np.full(n_cells, np.nan)

    obs = pd.DataFrame({
        'sample_id':     sample.name,
        'x_micron':      centroids['x'].values,
        'y_micron':      centroids['y'].values,
        'area_um2':      area_um2,
        'n_transcripts': np.asarray(X.sum(axis=1)).flatten().astype(int),
        'n_genes':       np.asarray((X > 0).sum(axis=1)).flatten().astype(int),
    }, index=pd.Index(cells, name='cell_id'))

    var = pd.DataFrame(index=pd.Index(genes, name='gene'))

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obs_names_make_unique()
    adata.obsm['spatial'] = obs[['x_micron', 'y_micron']].values
    adata.uns['spatial'] = {
        sample.name: {
            'images': {},
            'scalefactors': {'tissue_hires_scalef': 1.0, 'spot_diameter_fullres': 15.0}
        }
    }
    adata.uns['build_cells_xenium_version'] = BUILD_CELLS_XENIUM_VERSION

    return adata


# =============================================================================
# Platform-agnostic
# =============================================================================

def make_cells_from_masks(omic_channel, boundaries, min_obs=1, bin_size_um=200.0):
    """
    Aggregate observations within each cell boundary polygon into a cells x genes AnnData.

    Platform-agnostic: works on Visium spots, Xenium transcripts, or any OmicChannel
    where obsm['spatial'] holds (x, y) micron coordinates.

    Typical use cases:
      - Visium: boundaries = Cellpose polygons on NeuN/DAPI layer
                observations = spots (each with a full gene expression vector)
      - Xenium: boundaries = Cellpose polygons (alternative to instrument segmentation)
                observations = individual transcripts

    Parameters
    ----------
    omic_channel : OmicChannel
        Must have .data AnnData with obsm['spatial'] as (N, 2) micron coordinates
        and a count matrix in .data.X.
    boundaries : list of array-like, each shape (M, 2)
        Polygon vertex coordinates in micron space for each detected cell.
        Matches the format returned by Cellpose (list of outlines).
    min_obs : int
        Minimum number of observations (spots or transcripts) a boundary must
        contain to be kept.  Boundaries with fewer are silently skipped.
    bin_size_um : float
        Spatial bin size for the grid index in microns (default 200 µm).
        Each boundary searches only the bins that overlap its bounding box,
        so contains_points runs on a small candidate set rather than all spots.

    Returns
    -------
    AnnData  (n_cells x n_genes)
        obs columns:
            x_micron, y_micron   — polygon centroid in micron space
            n_obs                — bins/transcripts captured inside the boundary
            area_um2             — polygon area in µm² (shoelace formula)
            total_counts         — sum of all gene counts in the cell
            n_genes              — number of genes with at least one count
            counts_per_um2       — total_counts / area_um2 (expression density)
        var : inherited from omic_channel (same gene list)
        obsm['spatial'] : (n_cells, 2) centroid coordinates in microns
    """
    from matplotlib.path import Path
    from collections import defaultdict

    adata  = omic_channel.data
    coords = omic_channel.get_active_coords()   # (N, 2) — prefers spatial_fine if present
    X      = adata.X
    if hasattr(X, 'compute'):   # Dask-backed array: materialise once to avoid repeated lazy loads
        X = X.compute()

    # ── Spatial grid index ────────────────────────────────────────────────────
    # Bin every spot into a (row_bin, col_bin) tile.  For each boundary we only
    # test spots in tiles that overlap the polygon bounding box — typically a
    # handful of tiles vs. the full spot array.
    x_all, y_all = coords[:, 0], coords[:, 1]
    x_min, y_min = float(x_all.min()), float(y_all.min())

    col_bins = ((x_all - x_min) / bin_size_um).astype(int)
    row_bins = ((y_all - y_min) / bin_size_um).astype(int)

    grid = defaultdict(list)
    for i, (r, c) in enumerate(zip(row_bins, col_bins)):
        grid[(r, c)].append(i)
    grid = {k: np.array(v) for k, v in grid.items()}
    print(f"  Spatial index: {len(grid)} bins at {bin_size_um:.0f} µm  |  {len(coords):,} spots")

    cell_vecs         = []
    centroids         = []
    n_obs_list        = []
    areas             = []
    total_counts_list = []
    n_genes_list      = []

    for verts in boundaries:
        verts = np.asarray(verts)

        # Bins overlapping this polygon's bounding box
        x_lo, x_hi = float(verts[:, 0].min()), float(verts[:, 0].max())
        y_lo, y_hi = float(verts[:, 1].min()), float(verts[:, 1].max())
        c_lo = max(0, int((x_lo - x_min) / bin_size_um))
        c_hi =        int((x_hi - x_min) / bin_size_um)
        r_lo = max(0, int((y_lo - y_min) / bin_size_um))
        r_hi =        int((y_hi - y_min) / bin_size_um)

        cand_parts = [grid[(r, c)]
                      for r in range(r_lo, r_hi + 1)
                      for c in range(c_lo, c_hi + 1)
                      if (r, c) in grid]
        if not cand_parts:
            continue

        cand_idx    = np.concatenate(cand_parts)
        cand_coords = coords[cand_idx]

        # contains_points on candidates only
        poly_closed = np.vstack([verts, verts[0]])
        in_cand     = Path(poly_closed).contains_points(cand_coords)
        n_obs       = int(in_cand.sum())

        if n_obs < min_obs:
            continue

        hit_idx  = cand_idx[in_cand]
        cell_vec = np.asarray(X[hit_idx].sum(axis=0)).flatten()
        cell_vecs.append(cell_vec)

        centroids.append(verts.mean(axis=0))
        n_obs_list.append(n_obs)

        # Geometric and expression QC — computed here while verts/cell_vec are in hand
        x_v, y_v = verts[:, 0], verts[:, 1]
        area = 0.5 * abs(np.dot(x_v, np.roll(y_v, 1)) - np.dot(y_v, np.roll(x_v, 1)))
        areas.append(float(area))
        total_counts_list.append(float(cell_vec.sum()))
        n_genes_list.append(int((cell_vec > 0).sum()))

    if not cell_vecs:
        raise ValueError(
            f"No cells found with min_obs >= {min_obs}. "
            "Check that boundary coordinates are in micron space."
        )

    n_cells = len(cell_vecs)
    print(f"  Cells retained: {n_cells:,}  (from {len(boundaries)} boundaries, "
          f"min_obs={min_obs})")

    X_cells   = sp.csr_matrix(np.vstack(cell_vecs), dtype=np.float32)
    centroids = np.array(centroids)   # (n_cells, 2)

    obs = pd.DataFrame({
        'x_micron':      centroids[:, 0],
        'y_micron':      centroids[:, 1],
        'n_obs':         n_obs_list,
        'area_um2':      areas,
        'total_counts':  total_counts_list,
        'n_genes':       n_genes_list,
        'counts_per_um2': [tc / max(a, 1.0)
                           for tc, a in zip(total_counts_list, areas)],
    }, index=pd.Index(
        [f'{_encode_coord(x)}_{_encode_coord(y)}' for x, y in centroids],
        name='cell_id'
    ))

    result = ad.AnnData(X=X_cells, obs=obs, var=adata.var.copy())
    result.obs_names_make_unique()
    result.obsm['spatial'] = centroids
    result.uns['spatial'] = {
        omic_channel.name: {
            'images': {},
            'scalefactors': {'tissue_hires_scalef': 1.0, 'spot_diameter_fullres': 15.0}
        }
    }

    return result
