import json
import logging
import shutil
import os
import stat
from pathlib import Path
import numpy as np
import pandas as pd
import dask.array as da
import zarr
import anndata as ad
from typing import Optional, Any, Dict, Tuple


# platpy imports
from .channels import SampleSet, CellChannel, ImageChannel, OmicChannel, ROIChannel
from .transform_utils import SpatialMetadata
from .cell_utils import build_cells_xenium

logger = logging.getLogger(__name__)

# anndata still defaults to writing zarr format 2 (its own default is changing in a
# future release, independent of the zarr library's own default_zarr_format=3) —
# set explicitly so omic/cell AnnData writes use zarr v3, matching the image side.
ad.settings.zarr_write_format = 3

# On-disk format version stamps (platpy_migration_plan.md Item 0). Independent
# per-feature counters -- bumping one doesn't force a rebuild/reload of the
# others, even where two currently happen to be built in the same function call.
# Missing field on load = version 0 (pre-optimization), same graceful fallback
# the presence-checks already use -- not a breaking change for older files.
PLATPY_FORMAT_VERSION = 1
OPT_BLOCK_INDEX_VERSION = 1
OPT_TRANSCRIPT_DISPLAY_STATS_VERSION = 1


def build_spatial_block_index(adata, mpp, chunk_px=1024, gene_col=None):
    """
    Sort rows by spatial Dask block (aligned to chunk_px x mpp microns), then
    optionally by gene within each block. Stores a block->row-range lookup.
    Done once at initial load; persists in Zarr.

    Parameters
    ----------
    adata     : AnnData with obsm['spatial'] = (N, 2) float [x_um, y_um], OR a
                plain DataFrame with 'x'/'y' columns. The DataFrame path
                exists so Xenium's ingestion can sort the raw transcript
                table by spatial block BEFORE paying AnnData's construction
                overhead rather than after -- sorting (which needs a full
                reordered copy regardless) is much cheaper against a lean
                DataFrame than against an already-AnnData-wrapped object,
                and avoids ever holding two full-weight copies of the
                (much larger) wrapped version in memory at once. See
                xenium_loader.py's _load_transcripts().
    mpp       : float — microns per pixel of the reference morphology image
    chunk_px  : int   — image chunk size in pixels (default 1024, matches io_utils save)
    gene_col  : str or None — column for secondary sort (e.g. 'feature' for Xenium)

    Returns
    -------
    AnnData in -> sorted AnnData out (uns['block_index'] set), matching the
    original behavior. DataFrame in -> (sorted_df, block_index_json) tuple --
    a plain DataFrame has no .uns to carry the index on, so it's returned
    alongside instead; the caller is responsible for persisting it (e.g. in
    an OmicChannel's .metadata, or on an AnnData's .uns once one exists).
    """
    block_um = chunk_px * mpp
    is_df = isinstance(adata, pd.DataFrame)
    coords = adata[['x', 'y']].to_numpy() if is_df else adata.obsm['spatial']
    assert not np.any(np.isnan(coords)), "NaN coordinates — cannot sort by block"

    # Floor division (not truncation) so negative coordinates land in the correct block.
    # e.g. x=-400, block_um=333: floor gives bx=-2 (correct), int() gives bx=-1 (wrong).
    bx = np.floor(coords[:, 0] / block_um).astype(int)
    by = np.floor(coords[:, 1] / block_um).astype(int)

    if gene_col is not None:
        gene_vals = adata[gene_col].values if is_df else adata.obs[gene_col].values
        sort_df = pd.DataFrame({'by': by, 'bx': bx, 'gene': gene_vals})
        order = sort_df.sort_values(['by', 'bx', 'gene']).index.to_numpy()
    else:
        sort_df = pd.DataFrame({'by': by, 'bx': bx})
        order = sort_df.sort_values(['by', 'bx']).index.to_numpy()

    adata_sorted = adata.iloc[order].reset_index(drop=True) if is_df else adata[order].copy()

    # Find contiguous runs using pairwise comparison — avoids hash collisions with negative indices.
    s_bx = bx[order]
    s_by = by[order]
    same_block = (s_bx[1:] == s_bx[:-1]) & (s_by[1:] == s_by[:-1])
    change_mask = np.concatenate([[True], ~same_block])
    starts = np.where(change_mask)[0]
    ends = np.concatenate([starts[1:], [len(adata_sorted)]])

    # 1e-6 µm (sub-nanometer) margin guards against FP rounding at block boundaries.
    # block_um (= chunk_px * mpp) is not exactly representable in float64, so
    # floor(x / block_um) * block_um can be fractionally larger than x for boundary transcripts.
    eps = 1e-6
    blocks = [
        {
            'bx': int(s_bx[starts[k]]), 'by': int(s_by[starts[k]]),
            'min_x': float(s_bx[starts[k]] * block_um) - eps,
            'max_x': float((s_bx[starts[k]] + 1) * block_um) + eps,
            'min_y': float(s_by[starts[k]] * block_um) - eps,
            'max_y': float((s_by[starts[k]] + 1) * block_um) + eps,
            'row_start': int(starts[k]),
            'row_end':   int(ends[k]),
            'n':         int(ends[k] - starts[k]),
        }
        for k in range(len(starts))
    ]

    # Store as a JSON string — AnnData's zarr backend converts list-of-dicts
    # entries in uns to str() representations (single quotes, not JSON-compatible).
    # A single string value round-trips correctly through zarr.
    block_index_json = json.dumps({
        'block_size_um': float(block_um),
        'mpp':           float(mpp),
        'chunk_px':      int(chunk_px),
        'blocks':        blocks,
    })

    if is_df:
        # No .uns to carry this on -- hand it back alongside instead; the
        # caller persists it wherever makes sense (OmicChannel.metadata until
        # an AnnData exists, then its .uns -- see xenium_loader.py and
        # save_sample()'s pyramid-normalization step).
        return adata_sorted, block_index_json

    adata_sorted.uns['block_index'] = block_index_json
    # Independent version counters (see PLATPY_FORMAT_VERSION block above) — kept as
    # separate uns keys rather than nested inside block_index's own JSON blob, even
    # though both features are built right here, so each can be bumped/tested on its
    # own without disturbing the other.
    adata_sorted.uns['opt_block_index_version'] = OPT_BLOCK_INDEX_VERSION
    return adata_sorted


def query_block_index(block_index_json: str, min_x: float, max_x: float,
                       min_y: float, max_y: float):
    """
    Given a JSON-encoded block_index (see build_spatial_block_index above) and
    a query bounding box, return the list of (row_start, row_end) ranges for
    every block whose own bounds overlap the box.

    Shared by DataPort's extraction path (dataport.py) and ViewPort's Xenium
    render path (render_utils.py) so both narrow to the same row ranges
    before doing their own precise per-row filter -- one implementation of
    "which blocks does this query touch", not two copies that can drift
    apart from each other.
    """
    blocks = json.loads(block_index_json)['blocks']
    return [
        (b['row_start'], b['row_end'])
        for b in blocks
        if b['max_x'] >= min_x and b['min_x'] <= max_x
        and b['max_y'] >= min_y and b['min_y'] <= max_y
    ]


# Percentile ticks for the transcript_display_stats table. Coarser than the image side's
# 16-point display_stats (which is one array total per image) since this multiplies
# per-gene and panels can run to thousands of genes. Zeros are dropped before
# computing percentiles -- this is a *display* table, not an analysis statistic,
# so dynamic range shouldn't depend on how much non-expressing "blank space"
# surrounds the signal, mirroring the image side's own zero-pixel exclusion.
_TRANSCRIPT_DISPLAY_STATS_PCT_KEYS = [0.1, 1, 5, 25, 50, 95, 99, 100]

# Genes with fewer than this many total counts skip the full percentile
# computation -- checked 2026-08-26 against a real 549K-spot / 19K-gene whole-
# transcriptome Visium HD sample: mean per-gene zero-fraction was 99.5%, and
# 34% of genes had sum <= 100. Below this floor a gene's signal is either a
# handful of genuinely high cells or noise scattered everywhere; neither case
# is usefully described by a table of percentiles. Cheap to tune
# later -- doesn't force a version bump, see genes_reduced below.
_TRANSCRIPT_DISPLAY_STATS_MIN_SUM = 250


def _percentile_row(values: np.ndarray) -> list:
    """
    Shared tail of the transcript_display_stats per-gene stat row: 8 percentiles computed
    over nonzero values only, + rank of the first nonzero entry in the full
    ascending-sorted array (zeros included -- a cheap zero-fraction proxy), + sum
    of all values (not derivable from the percentiles, and the most commonly
    needed downstream statistic e.g. for total-count normalization).
    """
    values = np.sort(values)
    first_nonzero_rank = int(np.searchsorted(values, 1))
    nonzero = values[first_nonzero_rank:]
    pct_source = nonzero if len(nonzero) > 0 else values
    percentiles = np.percentile(pct_source, _TRANSCRIPT_DISPLAY_STATS_PCT_KEYS).tolist()
    return percentiles + [first_nonzero_rank, float(values.sum())]


def build_transcript_display_stats_table(adata, platform: str, bin_um: Optional[float] = None,
                                  gene_col: str = 'feature'):
    """
    Precompute a small per-gene percentile table for fast display/contrast
    scaling of rendered gene density -- the omic-side analog of the image
    `display_stats` pattern (see xenium_loader.py / visium_loader.py).

    Dispatches on platform:
      - visium (binned, has a genuine spot x gene count matrix already): computes
        directly from each gene's column in adata.X, one value per spot -- no
        spatial re-binning needed, spots are already the natural unit. Callers
        should pass a single reference pyramid level's adata (not run this once
        per level) and reuse the resulting table across all saved levels, so the
        reference stats aren't individually noisy at each resolution -- see
        save_sample()'s omic loop, which also stamps the actual spot-bin level
        used into the returned dict's 'bin_um' field after calling this.
      - xenium (continuous point cloud, one row per detected transcript, no
        per-row "expression value" to take a percentile of): builds a per-gene
        spatial histogram (transcript count per bin_um x bin_um tile) first --
        a cheap "pseudo-render" -- and computes percentiles over that instead.
        bin_um should roughly match a realistic interactive-viewing resolution
        (e.g. the paired image channel's second-coarsest pyramid level, not the
        full-slide thumbnail level, which would under-represent the contrast a
        user actually sees when zoomed in); required for this platform.
        Computed via a sparse (gene, bin) groupby rather than a dense
        n_genes x n_bins array -- a real 5K-panel / 73M-transcript Xenium
        section has enough spatial bins at typical bin_um that the dense
        version can reach multi-TiB memory. _percentile_row()'s math only
        ever needs each gene's *nonzero* per-bin counts plus a zero-bin count
        (pure arithmetic: n_bins - nnz), never a materialized dense row, so
        the sparse groupby result is sufficient and stays O(N) regardless of
        how many bins bin_um produces.

    Genes with fewer than _TRANSCRIPT_DISPLAY_STATS_MIN_SUM total counts skip the full
    percentile computation entirely and get a cheap [max, sum] pair in
    'genes_reduced' instead -- below that floor, a gene's signal is either a
    handful of genuinely high cells or scattered noise, and neither case is
    usefully described by a table of percentiles. This is also the actual
    performance-critical path: the sort+percentile call is the expensive part
    per gene, not computing a sum or max, so checking the (vectorized, cheap)
    sum first avoids that cost for the ~30%+ of genes typically below floor in
    real whole-transcriptome data.

    Returns a dict ready to be JSON-stringified into adata.uns['transcript_display_stats']:
        {'bin_um': <the spatial bin size, in microns, the stats were computed
                    over -- histogram tile size for xenium, source spot-bin
                    level for visium (set by the caller for visium; None here)>,
         'min_sum_threshold': _TRANSCRIPT_DISPLAY_STATS_MIN_SUM,
         'genes': {gene: [p0.1, p1, p5, p25, p50, p95, p99, p100,
                           first_nonzero_rank, sum], ...},
         'genes_reduced': {gene: [max, sum], ...}}
    Returns None if gene_col / expected data isn't present (nothing to build).
    """
    if platform == 'xenium':
        if gene_col not in adata.obs.columns or 'spatial' not in adata.obsm:
            return None
        if bin_um is None:
            raise ValueError("bin_um is required to build transcript_display_stats for a xenium channel")

        coords = adata.obsm['spatial']
        genes_arr = np.asarray(adata.obs[gene_col].values)
        min_x, min_y = coords[:, 0].min(), coords[:, 1].min()
        bx = np.floor((coords[:, 0] - min_x) / bin_um).astype(int)
        by = np.floor((coords[:, 1] - min_y) / bin_um).astype(int)
        n_bx, n_by = int(bx.max()) + 1, int(by.max()) + 1
        bin_id = by * n_bx + bx
        n_bins = n_bx * n_by  # used only as a scalar from here on -- never allocated

        # sort=True matches np.unique's lexicographic gene ordering.
        gene_codes, unique_genes = pd.factorize(genes_arr, sort=True)
        n_genes = len(unique_genes)
        gene_totals = np.bincount(gene_codes, minlength=n_genes)

        # Sparse (gene, bin) -> count. Size is bounded by the number of DISTINCT
        # (gene, bin) pairs actually observed (<= N transcripts) -- never
        # n_genes * n_bins, which is what made the old dense bincount-per-gene
        # version blow up on real whole-panel Xenium data.
        sparse_counts = (
            pd.DataFrame({'gene_code': gene_codes, 'bin_id': bin_id})
            .groupby(['gene_code', 'bin_id'])
            .size()
        )

        gene_table, genes_reduced = {}, {}
        for gene_code, per_gene in sparse_counts.groupby(level='gene_code'):
            gene = unique_genes[gene_code]
            total = int(gene_totals[gene_code])
            nonzero_vals = per_gene.to_numpy()  # this gene's nonzero per-bin counts only

            if total < _TRANSCRIPT_DISPLAY_STATS_MIN_SUM:
                genes_reduced[str(gene)] = [float(nonzero_vals.max()), float(total)]
            else:
                nonzero_sorted = np.sort(nonzero_vals)
                first_nonzero_rank = n_bins - len(nonzero_sorted)  # = count of zero bins
                percentiles = np.percentile(nonzero_sorted, _TRANSCRIPT_DISPLAY_STATS_PCT_KEYS).tolist()
                gene_table[str(gene)] = percentiles + [int(first_nonzero_rank), float(total)]

        return {'bin_um': float(bin_um), 'min_sum_threshold': _TRANSCRIPT_DISPLAY_STATS_MIN_SUM,
                'genes': gene_table, 'genes_reduced': genes_reduced}

    else:  # visium (and other binned platforms with a genuine spot x gene matrix)
        if adata.n_vars == 0:
            return None
        import scipy.sparse as sp
        X = adata.X
        if sp.issparse(X):
            # CSC (not CSR) for fast column slicing -- CSR's column access is
            # O(n_spots) per column regardless of sparsity, CSC's is O(nnz).
            X = X.tocsc()
        gene_sums = np.asarray(X.sum(axis=0)).ravel()

        gene_table, genes_reduced = {}, {}
        for i, gene in enumerate(adata.var_names):
            col = X[:, i]
            if gene_sums[i] < _TRANSCRIPT_DISPLAY_STATS_MIN_SUM:
                # Skip densifying the full column for low-signal genes -- just
                # need the max, which sparse .data (nonzero entries only) gives
                # directly without materializing all n_spots values.
                if sp.issparse(col):
                    col_max = float(col.data.max()) if col.nnz > 0 else 0.0
                else:
                    col_max = float(np.asarray(col).max()) if np.size(col) else 0.0
                genes_reduced[str(gene)] = [col_max, float(gene_sums[i])]
            else:
                # adata.X is virtually always scipy sparse for real 10x/scanpy
                # data -- np.asarray() does NOT densify a sparse column slice,
                # it wraps the sparse object itself, corrupting everything
                # downstream, so densify explicitly via .toarray() instead.
                col = col.toarray().ravel() if sp.issparse(col) else np.asarray(col).ravel()
                gene_table[str(gene)] = _percentile_row(col)

        return {'bin_um': None, 'min_sum_threshold': _TRANSCRIPT_DISPLAY_STATS_MIN_SUM,
                'genes': gene_table, 'genes_reduced': genes_reduced}


def _force_remove_tree(path):
    """
    Removes a directory tree, clearing Windows readonly bits first (a stale
    read-only file, e.g. left by a previous crashed write, would otherwise
    block a plain shutil.rmtree). Avoids shutil.rmtree's onerror/onexc
    callback entirely -- onexc is 3.12+ only and onerror's callback
    signature changed across versions, so clearing permissions up front is
    the one approach that doesn't depend on the running Python's version.
    """
    for root, dirs, files in os.walk(path):
        for entry in dirs + files:
            try:
                os.chmod(os.path.join(root, entry), stat.S_IWRITE)
            except OSError:
                pass
    shutil.rmtree(path)


# Big per-row transcript arrays worth resharding -- everything else in the
# bundle (categories, var, uns, small metadata) is tiny and left at
# AnnData's own default chunking.
_RESHARD_1D_PATHS = ['obs/x', 'obs/y', 'obs/qv', 'obs/transcript_id', 'obs/is_gene',
                      'obs/feature/codes', 'obs/cell_id/codes', 'obs/codeword_category/codes']


def _reshard_transcript_arrays(dest_path, n_obs: int, block_index_json: Optional[str],
                                min_chunk_rows: int = 20_000, max_chunk_rows: int = 100_000,
                                target_shard_rows: int = 500_000):
    """
    Re-chunks the big per-row transcript arrays AnnData's write_zarr() just
    wrote, using a row-count-informed uniform chunk size grouped into zarr
    v3 shards -- AnnData.write_zarr() has no `shards` param, and its own
    default per-array chunking is purely byte-size-driven (same chunk BYTE
    budget per array, so e.g. a uint64 column gets half the row-chunk of a
    float32 one) with zero spatial awareness: a small ROI query was measured
    (2026-10-25, real data) to read 4-8x more rows than it needed under
    that default. Sharding was confirmed (2026-10-25, via zarr-python's
    sharding codec source + a direct byte-range-read check) to genuinely
    fetch only the inner chunk(s) a partial read needs, not the whole shard
    -- so this gets small, query-friendly inner chunks without an explosion
    of small files on disk (verified: 725 -> 445 files on the 21.99M-row
    cropped test slide).

    True variable/ragged per-region chunk sizes (one physical chunk per
    Mondrian bucket or spatial block, each sized to its own local density)
    aren't available in stable zarr yet -- "rectilinear chunk grids" is an
    experimental, likely-v4-era proposal. This is the available proxy: one
    informed uniform inner-chunk size (from the real block_index's median
    block row count, clamped to [min_chunk_rows, max_chunk_rows]), not a
    per-block/per-bucket custom size. Shelved until rectilinear grids ship
    -- see project_platpy_streaming_ingestion memory.

    CRITICAL, confirmed the hard way (2026-10-25): the bundle has
    consolidated metadata (write_zarr produces it), which caches every
    array's chunk/shard shape. Rewriting arrays in place without
    re-consolidating left that cache stale -- reads then decoded the NEW
    chunk/shard bytes using the OLD cached shape, producing a zstd
    "destination buffer too small" error on every single array, not just
    the resharded ones. zarr.consolidate_metadata() at the end is required,
    not optional.
    """
    if not block_index_json or n_obs == 0:
        return
    blocks = json.loads(block_index_json).get('blocks', [])
    if not blocks:
        return
    block_sizes = sorted(b['row_end'] - b['row_start'] for b in blocks)
    median_block_rows = block_sizes[len(block_sizes) // 2]
    chunk_rows = int(np.clip(2 * median_block_rows, min_chunk_rows, max_chunk_rows))
    shard_rows = int(-(-target_shard_rows // chunk_rows) * chunk_rows)  # round up to a multiple

    root = zarr.open_group(str(dest_path), mode='a')
    for path in _RESHARD_1D_PATHS:
        if path not in root:
            continue
        arr = root[path]
        if arr.shape[0] != n_obs:
            continue
        root.create_array(path, chunks=(chunk_rows,), shards=(shard_rows,),
                           data=arr[:], overwrite=True)

    if 'obsm/spatial' in root:
        data = root['obsm/spatial'][:]
        root.create_array('obsm/spatial', chunks=(chunk_rows, data.shape[1]),
                           shards=(shard_rows, data.shape[1]), data=data, overwrite=True)

    zarr.consolidate_metadata(root.store)
    logger.info(f"     + Re-sharded transcript arrays: chunk_rows={chunk_rows:,}, "
                f"shard_rows={shard_rows:,} (median block size {median_block_rows:,})")


def _numpy_encoder(obj: Any) -> Any:
    """Helper to convert numpy arrays/types to standard JSON for Zarr attributes."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    return str(obj)

def _get_channel_info_string(d_arr: da.Array) -> str:
    """Returns a short string describing color space and bit-depth (e.g., '8g', '24c')."""
    dtype = d_arr.dtype
    bits = 8
    if dtype == np.uint16: bits = 16
    elif dtype in [np.float32, np.int32]: bits = 32
    elif dtype == np.float64: bits = 64
    
    ctype = 'g'
    if d_arr.ndim == 3:
        if d_arr.shape[-1] in [3, 4]:
            ctype = 'c'
            if bits == 8: bits = d_arr.shape[-1] * 8
            
    return f"{bits}{ctype}"

def _sanitize_for_zarr(obj: Any) -> Any:
    """
    Recursively converts dictionary values to JSON-serializable formats 
    (handling NumPy scalars/arrays) before writing to Zarr/AnnData.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_zarr(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_zarr(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    return obj

################################
## Save SampleSet to Zarr
##
## Saves both Images and Omics
## in a unified directory
## including metadta and ROIs.
################################

def _derive_transcript_display_stats_bin_um(sample: SampleSet, omic_chan: OmicChannel) -> float:
    """
    Picks a spatial bin size for transcript_display_stats's Xenium pseudo-render
    histogram: the second-coarsest level of an image channel's pyramid in the
    same sample (any one -- they're registered to the same coordinate system,
    so which specific channel doesn't matter). Deliberately not the coarsest
    (thumbnail) level, which would cover far more area per bin than a realistic
    interactive-viewing resolution and under-represent the contrast a user
    actually sees when zoomed in.

    Falls back to the same spatial block size build_spatial_block_index()
    already uses (chunk_px x mpp) if no image is present, or its pyramid has
    fewer than 2 levels (e.g. generate_pyramid() was never called).
    """
    for img_chan in sample.images.values():
        levels = sorted(img_chan.pyramid.keys())
        if len(levels) >= 2:
            return levels[-2]
        elif levels:
            return levels[-1]
    mpp = float(omic_chan.spatial_meta.transform_matrix[0, 0])
    return 1024 * mpp


def save_sample(
    sample: SampleSet,
    out_prefix: str,
    overwrite: bool = False,
    clevel: int = 5,
    keep_2um_bins: bool = False,
    build_default_cells: bool = True,
    **kwargs
):
    """
    Saves a platpy SampleSet to disk as a unified Zarr directory structure.

    Args:
        sample: The SampleSet to save.
        out_prefix: Directory path to save the sample.
        overwrite: If True, replaces existing data at out_prefix.
        clevel: Compression level (0-9). Default 5. Uses Blosc LZ4 + Bitshuffle.
        build_default_cells: For a Xenium sample with a 'cell_boundaries' cell
            channel, build the default cell x gene matrix (build_cells_xenium())
            and cache it on that channel's .data before saving, so later
            load_sample() + build_cells_xenium() calls skip recomputing it --
            the expensive part (loading/processing the raw data) already
            happened once to get here, so this is a small marginal cost now
            vs. a repeated one on every future reload. Set False to skip this
            when the caller already knows they want a different cell
            definition (different binning, boundary source, QC thresholds)
            and don't want save time spent on a default that won't be used.
            The raw transcripts/boundaries are saved either way -- this cache
            is a convenience, never the only path to a cell x gene matrix.
    """
    out_path = Path(out_prefix)
    
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Directory '{out_path}' already exists. Use overwrite=True to replace it."
        )

    out_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving platpy Sample '{sample.name}' to {out_path}")

    compressors = [zarr.codecs.BloscCodec(cname='lz4', clevel=clevel, shuffle=zarr.codecs.BloscShuffle.bitshuffle)]

    # ========
    # 1. Save Images (morph.zarr)
    # ========  

    if sample.images:
        morph_path = out_path / "morph.zarr"

        # Explicitly remove existing morph directory if overwrite is enabled
        if overwrite and morph_path.exists():
            _force_remove_tree(morph_path)

        morph_path.mkdir(parents=True, exist_ok=True)

        import ome_zarr.writer as ome_writer
        import ome_zarr.format as ome_format

        root: Any = zarr.open_group(str(morph_path), mode='a')  # type: ignore
        # ome_zarr's write_multiscale() still passes a singular 'compressor'
        # through to da.to_zarr() (not yet updated for zarr's current
        # 'compressors' kwarg -- confirmed harmless, just a deprecation
        # warning, nothing platpy can fix on its own side).
        ome_compressor = compressors[0]

        for name, img_chan in sample.images.items():
            img_data = img_chan.image
            is_rgb = img_data.ndim == 3
            # OME-NGFF requires non-space axes before space axes -- 'c' must
            # come before 'y'/'x', the reverse of platpy's own (H, W, C)
            # convention, so the array itself needs moveaxis'ing to match,
            # not just the axes label list.
            axes = ['c', 'y', 'x'] if is_rgb else ['y', 'x']
            chunks = (3, 1024, 1024) if is_rgb else (1024, 1024)

            base_mpp = float(img_chan.spatial_meta.transform_matrix[0, 0])
            tx = float(img_chan.spatial_meta.transform_matrix[0, 2])
            ty = float(img_chan.spatial_meta.transform_matrix[1, 2])

            def _to_ome_axes(arr):
                return da.moveaxis(arr, -1, 0) if (is_rgb and isinstance(arr, da.Array)) else (
                    np.moveaxis(arr, -1, 0) if is_rgb else arr)

            # Base level + persisted pyramid levels (skipped when
            # persist_pyramid=False, e.g. Xenium channels that already have
            # _hires/_lowres companions acting as the pyramid), largest/finest
            # first -- write_multiscale()'s required order.
            levels = {base_mpp: img_data}
            if img_chan.persist_pyramid:
                levels.update(img_chan.pyramid)
            sorted_mpps = sorted(levels.keys())
            pyramid_arrays = [_to_ome_axes(levels[m]) for m in sorted_mpps]

            coordinate_transformations = []
            for mpp in sorted_mpps:
                scale = [1.0, mpp, mpp] if is_rgb else [mpp, mpp]
                translation = [0.0, ty, tx] if is_rgb else [ty, tx]
                coordinate_transformations.append([
                    {'type': 'scale', 'scale': scale},
                    {'type': 'translation', 'translation': translation},
                ])

            if overwrite and name in root:
                del root[name]
            chan_group = root.require_group(name)
            ome_writer.write_multiscale(
                pyramid=pyramid_arrays,
                group=chan_group,
                fmt=ome_format.FormatV05(),
                axes=axes,
                coordinate_transformations=coordinate_transformations,
                storage_options={'chunks': chunks, 'compressor': ome_compressor},
            )

            # Non-standard extras platpy still needs, alongside the 'ome' key
            # write_multiscale() already wrote above -- the spec permits
            # additional attrs it doesn't define, so these just ride along.
            chan_group.attrs['platpy_meta'] = _sanitize_for_zarr({
                'unit': img_chan.spatial_meta.unit,
                'name': img_chan.name,
                'base_layer': img_chan.base_layer,
                'user_metadata': img_chan.metadata,
            })

            logger.info(f"  + Saved Image (OME-Zarr 0.5): {name} "
                        f"({len(pyramid_arrays)} level(s), clevel={clevel})")

    # ========
    # 2. Save Omics (omics/{name}/ pyramid directory, one zarr per level)
    # ========

    # Drop 2µm bins before saving — they are artifact-laden (intensity banding)
    # and the 4µm aggregated bin is a better starting point for all analyses.
    if not keep_2um_bins:
        for omic_chan in sample.omics.values():
            if 2.0 in omic_chan.pyramid and 4.0 in omic_chan.pyramid:
                del omic_chan.pyramid[2.0]
                logger.info("  - Dropped 2µm bin before save (keep_2um_bins=False)")

    if sample.omics:
        omics_dir = out_path / "omics"
        omics_dir.mkdir(exist_ok=True)
        for name, omic_chan in sample.omics.items():
            chan_dir = omics_dir / name
            if overwrite and chan_dir.exists():
                _force_remove_tree(chan_dir)
            chan_dir.mkdir(exist_ok=True)

            spatial_list = np.array(omic_chan.spatial_meta.transform_matrix).tolist()
            channel_meta = _sanitize_for_zarr(omic_chan.metadata)
            platform = omic_chan.metadata.get('platform', 'visium')
            stats_bin_um = _derive_transcript_display_stats_bin_um(sample, omic_chan) if platform == 'xenium' else None

            # Normalize every pyramid level to a real AnnData up front -- write_zarr
            # needs real data regardless, and a LazyTranscripts level (a sample
            # loaded lazily and resaved without ever being cropped) only gets
            # materialized once here, reused below for both the stats reference
            # level and the actual write, rather than pulling the full table from
            # zarr twice. A plain resave's row order/blocks are unchanged from the
            # original load, so the existing block_index (and anything else in
            # LazyTranscripts.uns) carries straight over instead of being rebuilt.
            from .lazy_transcripts import LazyTranscripts
            normalized_pyramid = {}
            for bin_um, level_data in omic_chan.pyramid.items():
                if isinstance(level_data, LazyTranscripts):
                    df = level_data.materialize()
                    df.index = df.index.astype(str)
                    level_adata = ad.AnnData(obs=df)
                    level_adata.obsm['spatial'] = df[['x', 'y']].to_numpy()
                    for k, v in level_data.uns.items():
                        level_adata.uns[k] = v
                elif isinstance(level_data, pd.DataFrame):
                    # Fresh ingestion (xenium_loader.py's _load_transcripts())
                    # stays a plain DataFrame all the way here -- this is the
                    # one place it finally gets wrapped, right before write,
                    # instead of up front at load time (see that function's
                    # comment for why: avoids ever holding two full-weight
                    # AnnData-sized copies at once during the spatial sort).
                    df = level_data.reset_index(drop=True)
                    df.index = df.index.astype(str)
                    level_adata = ad.AnnData(obs=df)
                    level_adata.obsm['spatial'] = df[['x', 'y']].to_numpy()
                    if 'block_index' in omic_chan.metadata:
                        level_adata.uns['block_index'] = omic_chan.metadata['block_index']
                        level_adata.uns['opt_block_index_version'] = omic_chan.metadata.get(
                            'opt_block_index_version', OPT_BLOCK_INDEX_VERSION)
                else:
                    level_adata = level_data
                normalized_pyramid[bin_um] = level_adata

            # Compute transcript_display_stats ONCE from a single reference level, not once per
            # pyramid level -- otherwise each resolution's table would be individually
            # noisy (Visium's 4um bins in particular). Mirrors the image side's own
            # choice to compute display_stats from one smooth thumbnail regardless of
            # render resolution. For Visium: second-coarsest kept bin level (same
            # "not the single coarsest option" principle as the Xenium bin_um choice
            # above -- coarsest available today is 16um, so this lands on 8um).
            # For Xenium there's only ever one pyramid entry (the 0.0 flag) anyway.
            stats_table = None
            if normalized_pyramid:
                levels_sorted = sorted(normalized_pyramid.keys())
                ref_level = levels_sorted[0] if platform == 'xenium' else (
                    levels_sorted[-2] if len(levels_sorted) >= 2 else levels_sorted[-1]
                )
                ref_adata = normalized_pyramid[ref_level]
                stats_table = build_transcript_display_stats_table(ref_adata, platform, bin_um=stats_bin_um)
                if platform != 'xenium' and stats_table is not None:
                    stats_table['bin_um'] = float(ref_level)  # which spot-bin level was used

            for bin_um, adata_level in normalized_pyramid.items():
                adata_level.uns['spatial_meta'] = spatial_list
                adata_level.uns['channel_meta'] = channel_meta

                if stats_table is not None:
                    adata_level.uns['transcript_display_stats'] = json.dumps(stats_table)
                    adata_level.uns['opt_transcript_display_stats_version'] = OPT_TRANSCRIPT_DISPLAY_STATS_VERSION

                dest_path = chan_dir / f"{float(bin_um):.1f}um.zarr"
                adata_level.write_zarr(dest_path)

                if platform == 'xenium' and name == 'transcripts':
                    _reshard_transcript_arrays(
                        dest_path, adata_level.n_obs,
                        adata_level.uns.get('block_index'),
                    )

            logger.info(f"  + Saved Omic Channel: {name} (levels: {sorted(omic_chan.pyramid.keys())})")

    # ========
    # 3. Save Cells (cells/*.zarr)
    # ========
    if sample.cells:
        cells_dir = out_path / "cells"
        cells_dir.mkdir(exist_ok=True)

        transcripts_chan = sample.omics.get('transcripts')
        is_xenium = transcripts_chan is not None and transcripts_chan.metadata.get('platform') == 'xenium'
        if build_default_cells and is_xenium and 'cell_boundaries' in sample.cells:
            default_cells = build_cells_xenium(sample, verbose=False)
            sample.cells['cell_boundaries'].data = default_cells
            logger.info(f"  + Built default cell x gene matrix "
                        f"({default_cells.n_obs:,} cells x {default_cells.n_vars:,} genes)")

        from .lazy_value import resolve as _resolve_lazy

        for name, cell_chan in sample.cells.items():
            dest_path = cells_dir / f"{name}.zarr"
            if overwrite and dest_path.exists():
                _force_remove_tree(dest_path)

            # A resave always needs the real boundaries/data regardless of
            # whether anything touched them yet -- write_zarr() needs actual
            # bytes either way, so there's no deferred-write equivalent here.
            adata: Any = _resolve_lazy(cell_chan.boundaries)
            if isinstance(adata, pd.DataFrame):
                df_obs = adata.reset_index(drop=True)
                df_obs.index = df_obs.index.astype(str)
                adata = ad.AnnData(obs=df_obs)
            
            adata.uns['spatial_meta'] = np.array(cell_chan.spatial_meta.transform_matrix).tolist()
            # FIX: Persist cell channel metadata
            adata.uns['channel_meta'] = _sanitize_for_zarr(cell_chan.metadata)

            adata.write_zarr(dest_path)

            # Per-cell centroid/area summary (derived from boundaries, not a
            # second source of truth) -- stored as its own parquet since it's
            # one row per cell, a different row count than the per-vertex
            # table above.
            if cell_chan.summary is not None:
                summary_path = cells_dir / f"{name}_summary.parquet"
                cell_chan.summary.to_parquet(summary_path)
                logger.info(f"  + Saved Cell Summary: {name} ({len(cell_chan.summary):,} cells)")

            # Default cell x gene matrix (see build_default_cells above) -- its
            # own zarr since it's one row per cell, a different row count than
            # the per-vertex boundaries table.
            if cell_chan.data is not None:
                cell_data = _resolve_lazy(cell_chan.data)
                data_path = cells_dir / f"{name}_cells.zarr"
                if overwrite and data_path.exists():
                    _force_remove_tree(data_path)
                cell_data.write_zarr(data_path)
                logger.info(f"  + Saved Cell x Gene Matrix: {name} "
                            f"({cell_data.n_obs:,} cells x {cell_data.n_vars:,} genes)")

            logger.info(f"  + Saved Cell Channel: {name}")


    # ========
    # 4. Save Core Metadata & ROIs
    # ========
    meta_path = out_path / "sample_meta.json"
    sample_info = {
        'name': sample.name,
        'platpy_format_version': PLATPY_FORMAT_VERSION,
        'metadata': sample.metadata,
        'rois': {}
    }
    
    for name, roi_chan in sample.rois.items():
        sample_info['rois'][name] = {
            'polygons': {k: v.tolist() for k, v in roi_chan.polygons.items()},
            'metadata': roi_chan.metadata
        }
        
    with open(meta_path, 'w') as f:
        json.dump(sample_info, f, default=_numpy_encoder, indent=4)
        
    logger.info("Successfully saved SampleSet.")
    return True


################################
## Load SampleSet from Zarr
##
## Loads both Images and Omics
## from a unified directory
## including metadata and ROIs.
################################


def load_sample(in_path_str: str, verbose: bool = True, load_2um_bin: bool = False, images_only: bool = False) -> SampleSet:
    """
    Loads a platpy SampleSet from a Zarr directory structure.
    Images are loaded as lazy dask arrays to preserve RAM.
    """
    in_path = Path(in_path_str)

    if not in_path.exists():
        raise FileNotFoundError(f"Sample directory not found: {in_path}")
        
    logger.info(f"Loading platpy Sample from {in_path}")

    with open(in_path / "sample_meta.json", 'r') as f:
        sample_info = json.load(f)

    sample = SampleSet(name=sample_info['name'])
    sample.metadata = sample_info.get('metadata', {})

    morph_path = in_path / "morph.zarr"
    summary_list = []
    if morph_path.exists():
        from .transform_utils import create_transform_matrix

        # Open with mode='r' (zarr auto-detects version on read)
        root: Any = zarr.open_group(str(morph_path), mode='r') # type: ignore
        for name in root.group_keys():
            chan_group = root[name]
            ome_meta = chan_group.attrs.get('ome', {}).get('multiscales', [{}])[0]
            platpy_meta = chan_group.attrs.get('platpy_meta', {})
            axes = [a['name'] for a in ome_meta.get('axes', [{'name': 'y'}, {'name': 'x'}])]
            y_idx, x_idx = axes.index('y'), axes.index('x')
            c_idx = axes.index('c') if 'c' in axes else None

            # Parse every dataset entry once into (mpp -> (array, tx, ty)) --
            # avoids a second pass to recover the base level's own translation.
            # Arrays get moveaxis'd back to platpy's native (y, x[, c]) order
            # here too -- OME-NGFF requires 'c' before the space axes, the
            # reverse of platpy's own (H, W, C) convention -- so every other
            # line below can just assume that native order, not OME's.
            levels: Dict[float, Any] = {}
            translations: Dict[float, Tuple[float, float]] = {}
            for ds in ome_meta.get('datasets', []):
                xforms = {t['type']: t for t in ds['coordinateTransformations']}
                mpp = float(xforms['scale']['scale'][x_idx])
                translation = xforms['translation']['translation']
                arr = da.from_zarr(chan_group[ds['path']])
                if c_idx is not None:
                    arr = da.moveaxis(arr, c_idx, -1)
                levels[mpp] = arr
                translations[mpp] = (float(translation[x_idx]), float(translation[y_idx]))

            if not levels:
                continue  # malformed/empty channel group -- nothing to load

            base_mpp = min(levels.keys())
            d_arr = levels[base_mpp]
            info_str = _get_channel_info_string(d_arr)
            summary_list.append(f"'{name}' ({info_str})")
            tx, ty = translations[base_mpp]

            h_px, w_px = d_arr.shape[0], d_arr.shape[1]
            spatial = SpatialMetadata(
                transform_matrix=create_transform_matrix(scale_x=base_mpp, scale_y=base_mpp, tx=tx, ty=ty),
                local_bounds=(0.0, float(w_px), 0.0, float(h_px)),
                unit=platpy_meta.get('unit', 'microns')
            )
            img_chan = ImageChannel(
                name=platpy_meta.get('name', name),
                image=d_arr,
                spatial_meta=spatial,
                metadata=platpy_meta.get('user_metadata', {})
            )
            img_chan.base_layer = platpy_meta.get('base_layer', None)

            for mpp, level_arr in levels.items():
                if abs(mpp - base_mpp) < 1e-9:
                    continue  # base level — already img_chan.image
                img_chan.pyramid[mpp] = level_arr
            if not img_chan.pyramid:
                img_chan.generate_pyramid()

            sample.images[name] = img_chan

    if verbose:
        if summary_list:
            print(f"  - Images available: {', '.join(summary_list)}")

    if images_only:
        if verbose:
            print(f"  - Image channels loaded: {', '.join(sample.images.keys())}")
            print(f"  (images_only=True — skipped omics, cells, ROIs)")
        sample._already_saved = True
        return sample

    omics_dir = in_path / "omics"
    if omics_dir.exists():
        # Flat single-level channels (old format and single-level new saves)
        for omic_path in omics_dir.glob("*.zarr"):
            name = omic_path.stem
            adata = ad.read_zarr(omic_path)
            spatial_mat = np.array(adata.uns.get('spatial_meta', np.eye(3)))
            spatial = SpatialMetadata(transform_matrix=spatial_mat)
            channel_meta = adata.uns.get('channel_meta', {})
            omic_chan = OmicChannel(name=name, data=adata, spatial_meta=spatial, metadata=channel_meta)
            sample.omics[name] = omic_chan

        # Pyramid channels: subdirectories containing {bin_um}um.zarr files
        for chan_dir in omics_dir.iterdir():
            if not chan_dir.is_dir() or chan_dir.suffix == '.zarr':
                continue
            def _parse_bin_um(p):
                try:
                    return float(p.stem.replace('um', ''))
                except ValueError:
                    return None
            valid_files = [p for p in chan_dir.glob("*.zarr") if _parse_bin_um(p) is not None]
            skipped = [p.name for p in chan_dir.glob("*.zarr") if _parse_bin_um(p) is None]
            if skipped:
                print(f"  [load_sample] Skipping unparseable zarr files in {chan_dir.name}/: {skipped}")
            level_files = sorted(valid_files, key=_parse_bin_um)
            if not level_files:
                continue

            # A single level stored at the 0.0 sentinel is how continuous point
            # clouds (Xenium transcripts) are always saved (__post_init__ /
            # save_sample()) -- Visium's real binned pyramids always have 2+
            # levels. Peek channel_meta cheaply (a handful of scalars, not the
            # row data) via zarr directly to confirm before committing to the
            # lazy path -- ad.read_zarr() is exactly the eager read this is
            # meant to avoid.
            is_lazy_candidate = (
                len(level_files) == 1
                and _parse_bin_um(level_files[0]) == 0.0
            )
            if is_lazy_candidate:
                from .lazy_transcripts import read_zarr_scalar_dict, load_transcripts_lazy
                probe_root = zarr.open_group(str(level_files[0]), mode='r')
                probe_meta = (read_zarr_scalar_dict(probe_root['uns']['channel_meta'])
                              if 'uns' in probe_root and 'channel_meta' in probe_root['uns'] else {})
                is_lazy_candidate = (probe_meta.get('platform') == 'xenium'
                                      and bool(probe_meta.get('is_continuous')))

            pyramid: Dict[float, Any] = {}
            spatial_mat = np.eye(3)
            channel_meta: Dict[str, Any] = {}

            if is_lazy_candidate:
                level_path = level_files[0]
                try:
                    lazy_data = load_transcripts_lazy(level_path)
                except Exception as e:
                    print(f"  [load_sample] WARNING: could not lazily read {level_path.name} in {chan_dir.name}/ — skipping ({type(e).__name__}: {e})")
                    lazy_data = None
                if lazy_data is not None:
                    spatial_mat = np.array(lazy_data.uns.get('spatial_meta', np.eye(3)))
                    channel_meta = probe_meta
                    pyramid[0.0] = lazy_data
            else:
                for level_path in level_files:
                    bin_um = float(level_path.stem.replace('um', ''))
                    try:
                        adata = ad.read_zarr(level_path)
                    except Exception as e:
                        print(f"  [load_sample] WARNING: could not read {level_path.name} in {chan_dir.name}/ — skipping ({type(e).__name__}: {e})")
                        continue
                    spatial_mat = np.array(adata.uns.get('spatial_meta', np.eye(3)))
                    channel_meta = adata.uns.get('channel_meta', {})
                    pyramid[bin_um] = adata

            if not load_2um_bin and 2.0 in pyramid and 4.0 in pyramid:
                del pyramid[2.0]

            if not pyramid:
                print(f"  [load_sample] WARNING: no valid zarr levels loaded for '{chan_dir.name}' — channel skipped.")
                continue

            name = chan_dir.name
            finest_bin = sorted(pyramid.keys())[0]
            # Keep bin_size_um in sync with the actual finest level in use —
            # the stored value may be stale if 2µm was originally finest but was dropped.
            # Skip for continuous point-cloud channels (Xenium) — they have no bin size
            # and injecting bin_size_um would wrongly route them to the Visium spot renderer.
            if (not channel_meta.get('is_continuous')
                    and channel_meta.get('bin_size_um') != finest_bin):
                channel_meta = {**channel_meta, 'bin_size_um': finest_bin}
            spatial = SpatialMetadata(transform_matrix=spatial_mat)
            omic_chan = OmicChannel(
                name=name, data=pyramid[finest_bin],
                spatial_meta=spatial, metadata=channel_meta,
                pyramid=pyramid
            )
            sample.omics[name] = omic_chan

    cells_dir = in_path / "cells"
    if cells_dir.exists():
        from .lazy_value import Deferred
        from .lazy_transcripts import read_zarr_scalar_dict

        for cell_path in cells_dir.glob("*.zarr"):
            # {name}_cells.zarr is the cached default cell x gene matrix for
            # the "{name}" channel this loop is building, not a boundary
            # table of its own -- skip it here, loaded explicitly below.
            if cell_path.stem.endswith('_cells'):
                continue
            name = cell_path.stem

            # Neither .boundaries (a per-vertex polygon table that can run to
            # tens-hundreds of MB) nor .data (the cached cell x gene matrix)
            # get loaded here -- only something that actually touches them
            # should pay for it. spatial_meta/channel_meta are still needed
            # up front though, so peek them cheaply (a 3x3 array, a handful
            # of scalars) directly via zarr rather than through ad.read_zarr(),
            # which would materialize the whole vertex table just to read a
            # few small uns entries.
            probe_root = zarr.open_group(str(cell_path), mode='r')
            probe_uns = probe_root['uns'] if 'uns' in probe_root else None
            spatial_mat = (np.array(probe_uns['spatial_meta'][()])
                           if probe_uns is not None and 'spatial_meta' in probe_uns else np.eye(3))
            channel_meta = (read_zarr_scalar_dict(probe_uns['channel_meta'])
                            if probe_uns is not None and 'channel_meta' in probe_uns else {})
            spatial = SpatialMetadata(transform_matrix=spatial_mat)

            boundaries = Deferred(loader=lambda p=cell_path: ad.read_zarr(p))

            summary_path = cells_dir / f"{name}_summary.parquet"
            summary = pd.read_parquet(summary_path) if summary_path.exists() else None

            # Cached default cell x gene matrix (see save_sample()'s
            # build_default_cells). Built once at save time and never mutated
            # afterward, so skipping AnnData's O(n) obs-index uniqueness check
            # on reload is safe by construction -- the index was already
            # validated the one time it was created. check_uniqueness=False
            # only needs to apply while THIS load actually runs, not globally,
            # so it's set inside the deferred loader itself rather than around
            # this whole function.
            data_path = cells_dir / f"{name}_cells.zarr"
            data = None
            if data_path.exists():
                def _load_cell_data(p=data_path):
                    with ad.settings.override(check_uniqueness=False):
                        return ad.read_zarr(p)
                data = Deferred(loader=_load_cell_data)

            cell_chan = CellChannel(name=name, boundaries=boundaries, data=data, summary=summary,
                                     spatial_meta=spatial, metadata=channel_meta)
            sample.cells[name] = cell_chan

    # save_sample() always writes sample_info['rois'] (polygons + metadata) into
    # sample_meta.json, but this reconstruction loop was previously missing
    # entirely -- any ROI added via add_roi() was silently lost on every reload.
    for name, roi_info in sample_info.get('rois', {}).items():
        polygons = {k: np.array(v) for k, v in roi_info.get('polygons', {}).items()}
        sample.rois[name] = ROIChannel(
            name=name, polygons=polygons, metadata=roi_info.get('metadata', {})
        )

    if verbose:
        print(f"  - Image channels loaded: {', '.join(sample.images.keys())}")
        print(f"  - Omic channels loaded: {', '.join(sample.omics.keys())}")
        print(f"  - Cell channels loaded: {', '.join(sample.cells.keys())}")
        print(f"  - ROIs loaded: {', '.join(sample.rois.keys())}")

    # Any sample returned by load_sample() came from disk — the save cell guard
    # uses this flag to skip the re-save step when using the alternate entry point.
    sample._already_saved = True

    return sample