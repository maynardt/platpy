import os
import json
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import dask
import dask.array as da
import tifffile

# Import core platpy architecture (Visium Parity)
from .channels import SampleSet, CellChannel, ImageChannel, OmicChannel
from .transform_utils import SpatialMetadata, create_transform_matrix
from .io_utils import build_spatial_block_index, OPT_BLOCK_INDEX_VERSION

# Set up library-level logger
logger = logging.getLogger(__name__)


def _read_tiff_level(img_path: Path, level: int) -> np.ndarray:
    """
    Reads one pyramid level's pixel data via its own freshly-opened
    tifffile.TiffFile handle, rather than a handle shared with other levels.

    load_xenium() wraps each level's read in dask.delayed() for lazy loading
    -- these only actually execute later, at save_sample() time, under
    dask's default threaded scheduler, which can run them concurrently.
    tifffile's FileHandle tracks a single read position per open file; two
    levels sharing one handle and reading concurrently can have their
    seek+read calls interleave, so one thread ends up decoding bytes read at
    another thread's offset. Seen in production as an intermittent
    "not a J2K or JP2 data stream" JPEG2000 decode error on otherwise-valid
    tiles (confirmed via standalone decode of every tile in the failing
    file -- the bytes on disk are fine). A fresh handle per delayed call has
    its own independent file descriptor/read position, so concurrent reads
    can't corrupt each other.
    """
    with tifffile.TiffFile(img_path, is_ome=False) as tif:
        return tif.series[0].levels[level].asarray()

def _polygon_summary(vertices: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a one-row-per-cell summary (true polygon centroid + area) from a
    long-format vertex table (cell_id, x, y, ...).

    Uses the shoelace formula, vectorized across all cells at once -- a plain
    mean of a cell's vertices is biased toward whatever side of the outline
    happens to carry more vertices, so this walks the actual polygon edges
    instead. Vertices within each cell_id's run are assumed to already trace
    the outline in order (as Xenium exports them); the edge from the last
    vertex back to the first is closed by wrapping within each group, not
    across group boundaries.
    """
    df = vertices.reset_index(drop=True)
    cell_id = df['cell_id'].to_numpy()
    x = df['x'].to_numpy(dtype=np.float64)
    y = df['y'].to_numpy(dtype=np.float64)
    n = len(df)

    is_group_start = np.empty(n, dtype=bool)
    is_group_start[0] = True
    is_group_start[1:] = cell_id[1:] != cell_id[:-1]
    group_start_idx = np.nonzero(is_group_start)[0]
    group_id_of_row = np.cumsum(is_group_start) - 1

    is_group_end = np.empty(n, dtype=bool)
    is_group_end[:-1] = is_group_start[1:]
    is_group_end[-1] = True

    # Next vertex along the outline, wrapping to each group's own first vertex
    # (not the next cell's first vertex) at each group's last row.
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    x_next[is_group_end] = x[group_start_idx[group_id_of_row[is_group_end]]]
    y_next[is_group_end] = y[group_start_idx[group_id_of_row[is_group_end]]]

    cross = x * y_next - x_next * y

    agg = pd.DataFrame({
        'cell_id':  df['cell_id'],
        'cross':    cross,
        'cx_term':  (x + x_next) * cross,
        'cy_term':  (y + y_next) * cross,
    }).groupby('cell_id', sort=False, observed=True).sum()

    signed_area2 = agg['cross']  # = 2x the signed polygon area
    area_um2 = (signed_area2 / 2.0).abs()

    with np.errstate(divide='ignore', invalid='ignore'):
        cx = agg['cx_term'] / (3.0 * signed_area2)
        cy = agg['cy_term'] / (3.0 * signed_area2)

    # Degenerate polygons (near-zero area -- duplicate/collinear vertices)
    # can't use the area-weighted formula; fall back to a plain vertex mean.
    degenerate = signed_area2.abs() < 1e-6
    if degenerate.any():
        g = df.groupby('cell_id', sort=False, observed=True)
        vx_mean, vy_mean = g['x'].mean(), g['y'].mean()
        cx = cx.where(~degenerate, vx_mean)
        cy = cy.where(~degenerate, vy_mean)

    g = df.groupby('cell_id', sort=False, observed=True)
    summary = pd.DataFrame({
        'x':          cx,
        'y':          cy,
        'area_um2':   area_um2,
        'n_vertices': g.size(),
    })
    if 'label_id' in df.columns:
        summary['label_id'] = g['label_id'].first()

    summary.index.name = 'cell_id'
    return summary


def _load_boundaries(x_dir: Path) -> Optional[CellChannel]:
    """Loads Xenium cell boundary polygons into a CellChannel."""
    bounds_path = x_dir / "cell_boundaries.parquet"
    if not bounds_path.exists():
        bounds_path = x_dir / "cell_boundaries.csv.gz" # Fallback
        
    if bounds_path.exists():
        logger.info(f"   - Loading Cell Boundaries: {bounds_path.name}")
        if bounds_path.suffix == '.parquet':
            df = pd.read_parquet(bounds_path)
        else:
            df = pd.read_csv(bounds_path)
            
        # ==========================================
        # platpy STANDARDIZATION
        # Translate Xenium 'vertex_x' to standard 'x'
        # ==========================================
        rename_map = {}
        if 'vertex_x' in df.columns: rename_map['vertex_x'] = 'x'
        if 'vertex_y' in df.columns: rename_map['vertex_y'] = 'y'
        if rename_map:
            df = df.rename(columns=rename_map)

        # cell_id as categorical instead of raw object/string -- confirmed
        # 2026-10-23 this is 81% of this table's in-memory footprint (467MB
        # of 579MB on a 279K-cell/6.98M-vertex run) despite being tiny on
        # disk (24.5MB parquet), same shape of issue already fixed for
        # transcripts' feature/cell_id columns. Also speeds up the groupby
        # calls in _polygon_summary() below and downstream (build_cells_xenium
        # etc.) -- categorical code comparisons are cheaper than string ones.
        df['cell_id'] = df['cell_id'].astype('category')

        summary = _polygon_summary(df)
        logger.info(f"     + Derived per-cell centroid/area for {len(summary):,} cells")

        return CellChannel(
            name="cell_boundaries",
            boundaries=df,
            summary=summary,
            metadata={"source": bounds_path.name}
        )
        
    logger.warning("Cell boundaries not found. Skipping CellChannel.")
    return None

def _load_transcripts(x_dir: Path) -> Optional[OmicChannel]:
    """Loads Xenium transcripts into a lean OmicChannel."""
    tx_path = x_dir / "transcripts.parquet"
    if not tx_path.exists():
        tx_path = x_dir / "transcripts.csv.gz" # Fallback for older XOA
        
    if tx_path.exists():
        logger.info(f"   - Loading Transcripts: {tx_path.name}")
        
        # Core columns needed for 2D spatial omics
        # (Dropping z_location, overlaps_nucleus, nucleus_distance, fov_name,
        # codeword_index, etc., to save massive RAM)
        core_cols = ['transcript_id', 'cell_id', 'feature_name', 'x_location', 'y_location', 'qv']
        # is_gene: correct, complete real-gene-vs-control flag (confirmed
        # 2026-10-23: matches codeword_category == 'predesigned_gene' with
        # zero crossover on a real 147M-row file) -- replaces the old
        # name-regex filter in build_cells_xenium(), which was silently
        # missing *_control_probe categories (they're "...Probe"-suffixed,
        # not "...Codeword"-suffixed, so the regex never caught them).
        # codeword_category: kept (not just the boolean) since per-cell
        # negative-control counts are a useful noise/background QC covariate,
        # not just something to drop -- only 6 distinct values, so it costs
        # almost nothing once AnnData's write_zarr() auto-categoricals it,
        # same as feature/cell_id already get.
        optional_cols = ['is_gene', 'codeword_category']

        if tx_path.suffix == '.parquet':
            # Parquet allows us to read only the columns we want, saving tons of memory!
            # Older Xenium Ranger outputs may not have is_gene/codeword_category
            # (newer-only columns) -- check the schema first rather than
            # letting read_parquet's columns= raise on a missing column.
            import pyarrow.parquet as pq
            available = set(pq.ParquetFile(tx_path).schema.names)
            cols_to_read = core_cols + [c for c in optional_cols if c in available]
            df = pd.read_parquet(tx_path, columns=cols_to_read)
        else:
            # CSV requires reading everything, then dropping
            df = pd.read_csv(tx_path)
            df = df[[c for c in core_cols + optional_cols if c in df.columns]]
            
        # ==========================================
        # platpy STANDARDIZATION
        # Translate Xenium coordinates to standard 'x', 'y', and 'feature'
        # ==========================================
        rename_map = {}
        if 'x_location' in df.columns: rename_map['x_location'] = 'x'
        if 'y_location' in df.columns: rename_map['y_location'] = 'y'
        if 'feature_name' in df.columns: rename_map['feature_name'] = 'feature'
        
        if rename_map:
            df = df.rename(columns=rename_map)                  # type: ignore
            
        # Sort the lean plain DataFrame BEFORE ever wrapping it in AnnData --
        # not after. The old order built the AnnData first, then
        # build_spatial_block_index()'s row-reorder copied that whole
        # already-AnnData-wrapped object just to sort it, meaning two
        # full-weight copies of the (much larger) wrapped version existed
        # simultaneously at peak. Sorting a plain DataFrame is the same
        # unavoidable reorder-copy cost, just against something far leaner.
        # platpy stays DataFrame-only all the way to save_sample(), which
        # wraps each pyramid level in AnnData right before write_zarr() --
        # there's no cell x gene X matrix here to justify holding AnnData's
        # overhead for the whole ingest, only at the point it's actually
        # needed. See project_platpy_bundle_creation_memory memory / the
        # 2026-10-23 OOM report this traces back to.
        df = df.reset_index(drop=True)
        logger.info("   - Sorting transcripts by spatial block and building index...")
        df, block_index_json = build_spatial_block_index(df, mpp=0.2125, gene_col='feature')
        _bi = json.loads(block_index_json)
        logger.info(f"     + Block index: {len(_bi['blocks'])} blocks ({_bi['block_size_um']:.1f} µm each)")

        return OmicChannel(
            name="transcripts",
            data=df,
            metadata={
                "platform": "xenium",
                "is_continuous": True,
                "source": tx_path.name,
                "total_transcripts": len(df),
                "block_index": block_index_json,
                "opt_block_index_version": OPT_BLOCK_INDEX_VERSION,
            }
        )
        
    logger.warning("Transcripts file not found. Skipping OmicChannel.")
    return None


def load_xenium(x_dir: str | Path, name: str = "XeniumSample", load_boundaries: bool = True, force_8bit=True, debug: bool = False) -> SampleSet:
    """
    Ingests a Xenium output directory into the platpy framework.
    Maps morphology images to the Visium multi-res standard using Dask/Zarr
    and Absolute Spatial Truth (Affine Matrices).
    """
    x_dir = Path(x_dir)
    sample = SampleSet(name=name)

    # ==========================================
    # 1. Multi-Res Morphology via Dask/Zarr
    # ==========================================
    # Hunt for the multi-file directory or the single file
    focus_dir = x_dir / "morphology_focus"
    img_paths = []
    
    if focus_dir.exists() and focus_dir.is_dir():
        # Original Xenium: morphology_focus_0000.ome.tif, ...
        # Xenium Prime/5K: ch0000_dapi.ome.tif, ch0001_..., ...
        img_paths = sorted(list(focus_dir.glob("morphology_focus_*.ome.tif")))
        if not img_paths:
            img_paths = sorted(list(focus_dir.glob("ch*.ome.tif")))
    else:
        # Fallbacks for single-file formats
        for fallback in ["morphology_focus.ome.tif", "morphology_mip.ome.tif", "morphology.ome.tif"]:
            if (x_dir / fallback).exists():
                img_paths = [x_dir / fallback]
                break

    if img_paths:
        base_mpp = 0.2125 # Standard Xenium micron-per-pixel scale

        for img_idx, img_path in enumerate(img_paths):
            # Derive channel name from filename when possible:
            #   "ch0000_dapi.ome.tif" → "DAPI"
            #   "ch0001_atp1a1_cd45_e-cadherin.ome.tif" → "atp1a1_cd45_e-cadherin"
            #   "morphology_focus_0000.ome.tif" → "DAPI" / "Marker_N"
            stem = img_path.stem.replace('.ome', '')
            if stem.startswith('ch'):
                parts = stem.split('_', 1)
                base_name = parts[1].upper() if parts[1].lower() == 'dapi' else parts[1] if len(parts) > 1 else f"Marker_{img_idx}"
            else:
                base_name = "DAPI" if img_idx == 0 else f"Marker_{img_idx}"
            logger.info(f"   - Found {base_name}: {img_path.name}")

            tif = tifffile.TiffFile(img_path, is_ome=False)
            sample.metadata[f'_tif_handle_{img_idx}'] = tif  # keeps it alive with the sample
            series = tif.series[0]
            available_levels = len(series.levels)
            base_shape = series.levels[0].shape

            def _load_level(target_level):
                level = min(target_level, available_levels - 1)
                lvl = series.levels[level]
                scale = base_shape[-1] / lvl.shape[-1]
                # lvl.shape/.dtype (header metadata, already parsed when tif
                # was opened above) are read eagerly here, safe and cheap --
                # only the actual pixel read (the expensive, race-prone part)
                # is deferred to a delayed call with its own fresh handle.
                arr = da.from_delayed(
                    dask.delayed(_read_tiff_level)(img_path, level),
                    shape=lvl.shape,
                    dtype=lvl.dtype
                )
                while arr.ndim > 2:
                    arr = arr[0]
                return arr, base_mpp * scale

            base_img, base_img_mpp = _load_level(0)
            h_px, w_px = base_img.shape[0], base_img.shape[1]

            meta = SpatialMetadata(
                transform_matrix=create_transform_matrix(
                    scale_x=base_img_mpp, scale_y=base_img_mpp, tx=0.0, ty=0.0
                ),
                local_bounds=(0.0, float(w_px), 0.0, float(h_px))
            )
            channel = ImageChannel(
                name=base_name,
                image=base_img,
                spatial_meta=meta,
                metadata={"source": img_path.name, "level": 0}
            )

            # Populate pyramid from the tiff's pre-existing levels — no resampling needed
            for target_level in [1, 3]:
                pyr_img, pyr_mpp = _load_level(target_level)
                if abs(pyr_mpp - base_img_mpp) > 1e-6:
                    channel.pyramid[pyr_mpp] = pyr_img

            if force_8bit and channel.image.dtype != np.uint8:
                channel.to_8bit(p_max=99.9, in_place=True, debug=debug)
                logger.info(f"     + Converted {base_name} to 8-bit.")

            # Pre-compute percentile table from the lowest-res pyramid level so
            # adjust_display() can set contrast without loading the full-res array.
            _PCT_KEYS = [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5, 99.9, 100]
            thumb = channel.image  # fall back to base if no pyramid yet
            if channel.pyramid:
                max_mpp_key = max(k for k in channel.pyramid if abs(k - channel.mpp) > 1e-6)
                thumb = channel.pyramid[max_mpp_key]
            if hasattr(thumb, 'compute'):
                thumb = thumb.compute()
            flat = thumb[thumb > 0].flatten() if np.any(thumb > 0) else thumb.flatten()
            source_mpp = max_mpp_key if channel.pyramid else channel.mpp
            channel.metadata['display_stats'] = {
                'percentile_keys':   _PCT_KEYS,
                'percentile_values': np.percentile(flat.astype(np.float32), _PCT_KEYS).tolist(),
                'source_mpp':        float(source_mpp),
            }

            sample.images[base_name] = channel
            logger.info(f"     + Created {base_name} (MPP: {base_img_mpp:.4f}, {len(channel.pyramid)} pyramid levels)")
    else:
        logger.warning("⚠️ No morphology images found. Skipping image load.")

    # ==========================================
    # 2. Load Boundaries (Cell Polygons)
    # ==========================================
    if load_boundaries:
        bounds_channel = _load_boundaries(x_dir)
        if bounds_channel:
            sample.cells['cell_boundaries'] = bounds_channel

    # ==========================================
    # 3. Load Transcripts (Omics)
    # ==========================================
    tx_channel = _load_transcripts(x_dir)
    if tx_channel:
        sample.omics['transcripts'] = tx_channel
        logger.info(f"   - Successfully loaded {len(tx_channel.data):,} transcripts.")


    return sample