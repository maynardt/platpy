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
from typing import Optional, Any, Dict


# platpy imports
from .channels import SampleSet, CellChannel, ImageChannel, OmicChannel, ROIChannel
from .transform_utils import SpatialMetadata 

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
OPT_NAME_OFFSET_VERSION = 1
OPT_TRANSCRIPT_ALPHA_VERSION = 1


def build_spatial_block_index(adata, mpp, chunk_px=1024, gene_col=None):
    """
    Sort AnnData rows by spatial Dask block (aligned to chunk_px x mpp microns),
    then optionally by gene within each block. Stores a block->row-range lookup
    in adata.uns['block_index']. Done once at initial load; persists in Zarr.

    Parameters
    ----------
    adata     : AnnData with obsm['spatial'] = (N, 2) float [x_um, y_um]
    mpp       : float — microns per pixel of the reference morphology image
    chunk_px  : int   — image chunk size in pixels (default 1024, matches io_utils save)
    gene_col  : str or None — obs column for secondary sort (e.g. 'feature' for Xenium)

    Returns
    -------
    adata_sorted : new AnnData with rows reordered; original is unchanged
    """
    block_um = chunk_px * mpp
    coords = adata.obsm['spatial']  # (N, 2): col0=x, col1=y
    assert not np.any(np.isnan(coords)), "NaN coordinates in obsm['spatial'] — cannot sort by block"

    # Floor division (not truncation) so negative coordinates land in the correct block.
    # e.g. x=-400, block_um=333: floor gives bx=-2 (correct), int() gives bx=-1 (wrong).
    bx = np.floor(coords[:, 0] / block_um).astype(int)
    by = np.floor(coords[:, 1] / block_um).astype(int)

    if gene_col is not None:
        sort_df = pd.DataFrame({'by': by, 'bx': bx, 'gene': adata.obs[gene_col].values})
        order = sort_df.sort_values(['by', 'bx', 'gene']).index.to_numpy()
    else:
        sort_df = pd.DataFrame({'by': by, 'bx': bx})
        order = sort_df.sort_values(['by', 'bx']).index.to_numpy()

    adata_sorted = adata[order].copy()

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

    # Add coarse gene name-prefix index within each block (Xenium only — gene_col must be set).
    # Genes are already sorted alphabetically within each block, so searchsorted finds boundaries
    # in O(log n) at query time with zero extra storage overhead beyond these small per-block dicts.
    if gene_col is not None:
        gene_vals = adata_sorted.obs[gene_col].values
        for b in blocks:
            genes = gene_vals[b['row_start']:b['row_end']]
            letters = sorted(set(g[0].upper() for g in genes if g))
            b['name_offsets'] = {
                letter: (
                    int(np.searchsorted(genes, letter)),
                    int(np.searchsorted(genes, chr(ord(letter) + 1), 'left'))
                )
                for letter in letters
            }

    # Store as a JSON string — AnnData's zarr backend converts list-of-dicts
    # entries in uns to str() representations (single quotes, not JSON-compatible).
    # A single string value round-trips correctly through zarr.
    adata_sorted.uns['block_index'] = json.dumps({
        'block_size_um': float(block_um),
        'mpp':           float(mpp),
        'chunk_px':      int(chunk_px),
        'blocks':        blocks,
    })
    # Independent version counters (see PLATPY_FORMAT_VERSION block above) — kept as
    # separate uns keys rather than nested inside block_index's own JSON blob, even
    # though both features are built right here, so each can be bumped/tested on its
    # own without disturbing the other.
    adata_sorted.uns['opt_block_index_version'] = OPT_BLOCK_INDEX_VERSION
    if gene_col is not None:
        adata_sorted.uns['opt_name_offset_version'] = OPT_NAME_OFFSET_VERSION
    return adata_sorted


# Percentile ticks for the transcript_alpha table. Coarser than the image side's
# 16-point display_stats (which is one array total per image) since this multiplies
# per-gene and panels can run to thousands of genes. Zeros are dropped before
# computing percentiles -- this is a *display* table, not an analysis statistic,
# so dynamic range shouldn't depend on how much non-expressing "blank space"
# surrounds the signal, mirroring the image side's own zero-pixel exclusion.
_TRANSCRIPT_ALPHA_PCT_KEYS = [0.1, 1, 5, 25, 50, 95, 99, 100]

# Genes with fewer than this many total counts skip the full percentile
# computation -- checked 2026-08-26 against a real 549K-spot / 19K-gene whole-
# transcriptome Visium HD sample: mean per-gene zero-fraction was 99.5%, and
# 34% of genes had sum <= 100. Below this floor a gene's signal is either a
# handful of genuinely high cells or noise scattered everywhere; neither case
# is usefully described by a table of percentiles (per Tom). Cheap to tune
# later -- doesn't force a version bump, see genes_reduced below.
_TRANSCRIPT_ALPHA_MIN_SUM = 250


def _percentile_row(values: np.ndarray) -> list:
    """
    Shared tail of the transcript_alpha per-gene stat row: 8 percentiles computed
    over nonzero values only, + rank of the first nonzero entry in the full
    ascending-sorted array (zeros included -- a cheap zero-fraction proxy), + sum
    of all values (not derivable from the percentiles, and the most commonly
    needed downstream statistic e.g. for total-count normalization).
    """
    values = np.sort(values)
    first_nonzero_rank = int(np.searchsorted(values, 1))
    nonzero = values[first_nonzero_rank:]
    pct_source = nonzero if len(nonzero) > 0 else values
    percentiles = np.percentile(pct_source, _TRANSCRIPT_ALPHA_PCT_KEYS).tolist()
    return percentiles + [first_nonzero_rank, float(values.sum())]


def build_transcript_alpha_table(adata, platform: str, bin_um: Optional[float] = None,
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

    Genes with fewer than _TRANSCRIPT_ALPHA_MIN_SUM total counts skip the full
    percentile computation entirely and get a cheap [max, sum] pair in
    'genes_reduced' instead -- below that floor, a gene's signal is either a
    handful of genuinely high cells or scattered noise, and neither case is
    usefully described by a table of percentiles. This is also the actual
    performance-critical path: the sort+percentile call is the expensive part
    per gene, not computing a sum or max, so checking the (vectorized, cheap)
    sum first avoids that cost for the ~30%+ of genes typically below floor in
    real whole-transcriptome data.

    Returns a dict ready to be JSON-stringified into adata.uns['transcript_alpha']:
        {'bin_um': <the spatial bin size, in microns, the stats were computed
                    over -- histogram tile size for xenium, source spot-bin
                    level for visium (set by the caller for visium; None here)>,
         'min_sum_threshold': _TRANSCRIPT_ALPHA_MIN_SUM,
         'genes': {gene: [p0.1, p1, p5, p25, p50, p95, p99, p100,
                           first_nonzero_rank, sum], ...},
         'genes_reduced': {gene: [max, sum], ...}}
    Returns None if gene_col / expected data isn't present (nothing to build).
    """
    if platform == 'xenium':
        if gene_col not in adata.obs.columns or 'spatial' not in adata.obsm:
            return None
        if bin_um is None:
            raise ValueError("bin_um is required to build transcript_alpha for a xenium channel")

        coords = adata.obsm['spatial']
        genes_arr = np.asarray(adata.obs[gene_col].values)
        min_x, min_y = coords[:, 0].min(), coords[:, 1].min()
        bx = np.floor((coords[:, 0] - min_x) / bin_um).astype(int)
        by = np.floor((coords[:, 1] - min_y) / bin_um).astype(int)
        n_bx, n_by = int(bx.max()) + 1, int(by.max()) + 1
        bin_id = by * n_bx + bx
        n_bins = n_bx * n_by

        # Total transcript count per gene -- just a row count per gene, no
        # binning needed -- so this cheap pre-filter costs nothing extra even
        # though the histogram itself still needs building either way to get max.
        unique_genes, gene_totals = np.unique(genes_arr, return_counts=True)

        gene_table, genes_reduced = {}, {}
        for gene, total in zip(unique_genes, gene_totals):
            counts = np.bincount(bin_id[genes_arr == gene], minlength=n_bins)
            if total < _TRANSCRIPT_ALPHA_MIN_SUM:
                genes_reduced[str(gene)] = [float(counts.max()) if counts.size else 0.0, float(total)]
            else:
                gene_table[str(gene)] = _percentile_row(counts)

        return {'bin_um': float(bin_um), 'min_sum_threshold': _TRANSCRIPT_ALPHA_MIN_SUM,
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
            if gene_sums[i] < _TRANSCRIPT_ALPHA_MIN_SUM:
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

        return {'bin_um': None, 'min_sum_threshold': _TRANSCRIPT_ALPHA_MIN_SUM,
                'genes': gene_table, 'genes_reduced': genes_reduced}


def _remove_readonly(func, path, exc):
    """Clear the readonly bit and re-attempt purge (Windows fix)."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

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

def _derive_transcript_alpha_bin_um(sample: SampleSet, omic_chan: OmicChannel) -> float:
    """
    Picks a spatial bin size for transcript_alpha's Xenium pseudo-render
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
    **kwargs
):
    """
    Saves a platpy SampleSet to disk as a unified Zarr directory structure.
    
    Args:
        sample: The SampleSet to save.
        out_prefix: Directory path to save the sample.
        overwrite: If True, replaces existing data at out_prefix.
        clevel: Compression level (0-9). Default 5. Uses Blosc LZ4 + Bitshuffle.
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
            shutil.rmtree(morph_path, onexc=_remove_readonly)
        
        morph_path.mkdir(parents=True, exist_ok=True)
        
        for name, img_chan in sample.images.items():
            img_data = img_chan.image
            
            root: Any = zarr.open_group(str(morph_path), mode='a') # type: ignore
            chunks = (1024, 1024, 3) if img_data.ndim == 3 else (1024, 1024)
            z = root.require_dataset(
                name, shape=img_data.shape, chunks=chunks,
                dtype=img_data.dtype, overwrite=True, compressors=compressors
            )
            if isinstance(img_data, da.Array):
                da.store(img_data, z)
            else:
                z[:] = img_data
            
            safe_matrix = np.array(img_chan.spatial_meta.transform_matrix).tolist()
            meta_dict = {
                'transform_matrix': safe_matrix,
                'local_bounds': img_chan.spatial_meta.local_bounds,
                'unit': img_chan.spatial_meta.unit,
                'name': img_chan.name,
                'base_layer': img_chan.base_layer,
                'user_metadata': img_chan.metadata
            }
            root[name].attrs.put(meta_dict) # type: ignore

            # Save pre-computed pyramid levels so renders don't re-derive from full-res.
            # Skipped when persist_pyramid=False (e.g. Xenium channels that already have
            # _hires/_lowres companions acting as the pyramid).
            # Key format: {name}__pyr{factor} where factor = 2, 4, 8, ...
            base_mpp = img_chan.spatial_meta.transform_matrix[0, 0]
            for level_mpp, level_arr in img_chan.pyramid.items():
                if abs(level_mpp - base_mpp) < 1e-9:
                    continue  # skip base level — already saved as the main component
                factor = round(level_mpp / base_mpp)
                pyr_key = f"{name}__pyr{factor}"
                pyr_chunks = (1024, 1024, 3) if level_arr.ndim == 3 else (1024, 1024)
                zp = root.require_dataset(
                    pyr_key, shape=level_arr.shape, chunks=pyr_chunks,
                    dtype=level_arr.dtype, overwrite=True, compressors=compressors
                )
                if isinstance(level_arr, da.Array):
                    da.store(level_arr, zp)
                else:
                    zp[:] = level_arr
                root[pyr_key].attrs.put({'mpp': float(level_mpp)})

            logger.info(f"  + Saved Image: {name} ({len(img_chan.pyramid)} pyramid level(s), clevel={clevel})")

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
                shutil.rmtree(chan_dir, onexc=_remove_readonly)
            chan_dir.mkdir(exist_ok=True)

            spatial_list = np.array(omic_chan.spatial_meta.transform_matrix).tolist()
            channel_meta = _sanitize_for_zarr(omic_chan.metadata)
            platform = omic_chan.metadata.get('platform', 'visium')
            alpha_bin_um = _derive_transcript_alpha_bin_um(sample, omic_chan) if platform == 'xenium' else None

            # Compute transcript_alpha ONCE from a single reference level, not once per
            # pyramid level -- otherwise each resolution's table would be individually
            # noisy (Visium's 4um bins in particular). Mirrors the image side's own
            # choice to compute display_stats from one smooth thumbnail regardless of
            # render resolution. For Visium: second-coarsest kept bin level (same
            # "not the single coarsest option" principle as the Xenium bin_um choice
            # above -- coarsest available today is 16um, so this lands on 8um).
            # For Xenium there's only ever one pyramid entry (the 0.0 flag) anyway.
            alpha_table = None
            if omic_chan.pyramid:
                levels_sorted = sorted(omic_chan.pyramid.keys())
                ref_level = levels_sorted[0] if platform == 'xenium' else (
                    levels_sorted[-2] if len(levels_sorted) >= 2 else levels_sorted[-1]
                )
                ref_adata = omic_chan.pyramid[ref_level]
                if isinstance(ref_adata, pd.DataFrame):
                    ref_adata = ad.AnnData(obs=ref_adata)
                alpha_table = build_transcript_alpha_table(ref_adata, platform, bin_um=alpha_bin_um)
                if platform != 'xenium' and alpha_table is not None:
                    alpha_table['bin_um'] = float(ref_level)  # which spot-bin level was used

            for bin_um, adata_level in omic_chan.pyramid.items():
                if isinstance(adata_level, pd.DataFrame):
                    adata_level = ad.AnnData(obs=adata_level)
                adata_level.uns['spatial_meta'] = spatial_list
                adata_level.uns['channel_meta'] = channel_meta

                if alpha_table is not None:
                    adata_level.uns['transcript_alpha'] = json.dumps(alpha_table)
                    adata_level.uns['opt_transcript_alpha_version'] = OPT_TRANSCRIPT_ALPHA_VERSION

                dest_path = chan_dir / f"{float(bin_um):.1f}um.zarr"
                adata_level.write_zarr(dest_path)

            logger.info(f"  + Saved Omic Channel: {name} (levels: {sorted(omic_chan.pyramid.keys())})")

    # ========
    # 3. Save Cells (cells/*.zarr)
    # ========
    if sample.cells:
        cells_dir = out_path / "cells"
        cells_dir.mkdir(exist_ok=True)
        for name, cell_chan in sample.cells.items():
            dest_path = cells_dir / f"{name}.zarr"
            if overwrite and dest_path.exists():
                shutil.rmtree(dest_path, onexc=_remove_readonly)
                
            adata: Any = cell_chan.boundaries
            if isinstance(adata, pd.DataFrame):
                df_obs = adata.reset_index(drop=True)
                df_obs.index = df_obs.index.astype(str)
                adata = ad.AnnData(obs=df_obs)
            
            adata.uns['spatial_meta'] = np.array(cell_chan.spatial_meta.transform_matrix).tolist()
            # FIX: Persist cell channel metadata
            adata.uns['channel_meta'] = _sanitize_for_zarr(cell_chan.metadata)
            
            adata.write_zarr(dest_path)
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
        # Open with mode='r' (zarr auto-detects version on read)
        root: Any = zarr.open_group(str(morph_path), mode='r') # type: ignore
        for name in root.array_keys():
            if '__pyr' in name:
                continue  # pyramid level — attached to parent channel below
            z_arr = root[name]
            d_arr = da.from_zarr(z_arr)
            info_str = _get_channel_info_string(d_arr)
            summary_list.append(f"'{name}' ({info_str})")
            
            attrs = z_arr.attrs.asdict() # type: ignore
            spatial = SpatialMetadata(
                transform_matrix=np.array(attrs.get('transform_matrix', np.eye(3))),
                local_bounds=attrs.get('local_bounds'),
                unit=attrs.get('unit', 'microns')
            )
            img_chan = ImageChannel(
                name=attrs.get('name', name),
                image=d_arr,
                spatial_meta=spatial,
                metadata=attrs.get('user_metadata', {})
            )
            img_chan.base_layer = attrs.get('base_layer', None)

            # Load pre-computed pyramid levels if saved; otherwise generate on-the-fly.
            pyr_keys = [k for k in root.array_keys() if k.startswith(f"{name}__pyr")]
            if pyr_keys:
                for pyr_key in pyr_keys:
                    level_mpp = root[pyr_key].attrs.asdict().get('mpp')
                    if level_mpp is not None:
                        img_chan.pyramid[float(level_mpp)] = da.from_zarr(root[pyr_key])
            else:
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

            pyramid: Dict[float, Any] = {}
            spatial_mat = np.eye(3)
            channel_meta: Dict[str, Any] = {}
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
        for cell_path in cells_dir.glob("*.zarr"):
            name = cell_path.stem
            adata = ad.read_zarr(cell_path)
            spatial_mat = np.array(adata.uns.get('spatial_meta', np.eye(3)))
            spatial = SpatialMetadata(transform_matrix=spatial_mat)
            
            # FIX: Restore the channel metadata
            channel_meta = adata.uns.get('channel_meta', {})
            
            cell_chan = CellChannel(name=name, boundaries=adata, spatial_meta=spatial, metadata=channel_meta)
            sample.cells[name] = cell_chan

    if verbose:
        print(f"  - Image channels loaded: {', '.join(sample.images.keys())}")
        print(f"  - Omic channels loaded: {', '.join(sample.omics.keys())}")
        print(f"  - Cell channels loaded: {', '.join(sample.cells.keys())}")
        print(f"  - ROIs loaded: {', '.join(sample.rois.keys())}")

    # Any sample returned by load_sample() came from disk — the save cell guard
    # uses this flag to skip the re-save step when using the alternate entry point.
    sample._already_saved = True

    return sample