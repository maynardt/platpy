import os
import json
import logging
from pathlib import Path
from typing import Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import psutil
import dask
import dask.array as da
import tifffile

# Import core platpy architecture (Visium Parity)
from .channels import SampleSet, CellChannel, ImageChannel, OmicChannel
from .transform_utils import SpatialMetadata, create_transform_matrix
from .io_utils import build_spatial_block_index, OPT_BLOCK_INDEX_VERSION

# Set up library-level logger
logger = logging.getLogger(__name__)


def _estimate_sort_peak_gb(n_rows: int) -> float:
    """
    Calibrated cost of sorting (via build_spatial_block_index()) a table of
    n_rows transcript rows in one piece, with a ~5% safety margin.
    peak_GB ~= 0.8 + 0.25 * rows_millions -- fit (2026-10-24) from 6 isolated
    single-process runs on real LRRK2_NIGRA data at 5M-80M rows (linear fit
    consistent to within ~4% across every pair of points). Used both to gate
    Pass-3 parallel worker count (per-bucket) and to auto-select between the
    streaming and non-streaming transcript-load paths (whole-table estimate).
    """
    return 0.8 + 0.25 * (n_rows / 1e6)


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


def _sort_one_bucket_worker(bucket_path: Path, sorted_path: Path, rename_map: dict,
                             mpp: float, chunk_px: int) -> tuple:
    """
    Pass-3 worker body: reads one bucket's routed parquet, sorts it via the
    existing build_spatial_block_index(), writes the sorted result straight
    to its own output parquet file. Returns small metadata only (row count +
    block_index JSON), never the DataFrame itself -- keeps worker->main IPC
    cheap and avoids doubling memory via pickling a multi-million-row frame.
    Module-level (not a closure) so it's picklable for ProcessPoolExecutor's
    spawn-based workers on Windows.
    """
    bdf = pd.read_parquet(bucket_path).rename(columns=rename_map)
    bdf_sorted, bi_json = build_spatial_block_index(bdf, mpp=mpp, chunk_px=chunk_px, gene_col='feature')
    bdf_sorted.to_parquet(sorted_path)
    return len(bdf_sorted), bi_json


def _build_density_map(tx_path: Path, block_um: float, batch_size: int = 2_000_000) -> dict:
    """
    Cheap single pass over just x_location/y_location, binning rows into a
    sparse {(bx, by): row_count} map. No upfront min/max scan needed --
    floor(coord / block_um) is a valid bin index regardless of the
    section's bounding box, so unoccupied blocks simply never get a dict
    entry. Verified (2026-10-24, output-cropped-270_WT, 21.99M rows): 1.7s
    single-threaded; this is the Pass-1 building block for the 2D
    density-balanced ("Mondrian") split in _mondrian_split().
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(tx_path)
    counts: dict = {}
    for batch in pf.iter_batches(batch_size=batch_size, columns=['x_location', 'y_location']):
        x = batch.column(0).to_numpy(zero_copy_only=False)
        y = batch.column(1).to_numpy(zero_copy_only=False)
        bx = np.floor(x / block_um).astype(np.int64)
        by = np.floor(y / block_um).astype(np.int64)
        keys = bx * 2_000_000_000 + by
        uniq, cts = np.unique(keys, return_counts=True)
        for k, c in zip(uniq.tolist(), cts.tolist()):
            bx_k, by_k = divmod(k, 2_000_000_000)
            key = (bx_k, by_k)
            counts[key] = counts.get(key, 0) + c
    return counts


def _mondrian_split(blocks: dict, target_rows: int, min_span: int = 1) -> list:
    """
    Recursively cuts the occupied-block region along whichever axis (x or y)
    gives the more balanced row-count split, at a block-aligned position
    (bx/by are already integer block units, so every cut is automatically
    block-aligned). Named for the visual resemblance to Mondrian's paintings
    -- the same shape as the "Mondrian process" recursive space partition
    used in ML (Mondrian forests). Prototyped and verified (2026-10-24,
    output-cropped-270_WT): exact row conservation, zero overlap, full
    coverage at every target size tested, cutting the worst-case bucket-size
    imbalance from the 1D y-only approach's 32x spread down to ~1.2-2.3x.

    blocks: dict {(bx, by): row_count} (or a subset of one, during recursion).
    Returns a list of leaves: [{'blocks': {(bx, by): count, ...}, 'rows': int}, ...]
    """
    total = sum(blocks.values())
    if total <= target_rows or len(blocks) <= 1:
        return [{'blocks': blocks, 'rows': total}]

    bxs = sorted(set(bx for bx, by in blocks))
    bys = sorted(set(by for bx, by in blocks))
    if len(bxs) <= min_span and len(bys) <= min_span:
        return [{'blocks': blocks, 'rows': total}]

    def best_cut(axis_vals, axis_idx):
        if len(axis_vals) <= 1:
            return None
        sums = {v: 0 for v in axis_vals}
        for key, c in blocks.items():
            sums[key[axis_idx]] += c
        prefix = 0
        best = None
        for i in range(len(axis_vals) - 1):
            prefix += sums[axis_vals[i]]
            imbalance = abs(prefix - (total - prefix))
            if best is None or imbalance < best[0]:
                best = (imbalance, axis_vals[i])
        return best

    x_cut = best_cut(bxs, 0)
    y_cut = best_cut(bys, 1)
    if x_cut is None and y_cut is None:
        return [{'blocks': blocks, 'rows': total}]

    use_x = x_cut is not None and (y_cut is None or x_cut[0] <= y_cut[0])
    cut_val = x_cut[1] if use_x else y_cut[1]
    axis_idx = 0 if use_x else 1

    left = {k: v for k, v in blocks.items() if k[axis_idx] <= cut_val}
    right = {k: v for k, v in blocks.items() if k[axis_idx] > cut_val}
    if not left or not right:
        return [{'blocks': blocks, 'rows': total}]

    return _mondrian_split(left, target_rows, min_span) + _mondrian_split(right, target_rows, min_span)


def _stream_sort_transcripts(tx_path: Path, cols_to_read: list, rename_map: dict,
                              mpp: float = 0.2125, chunk_px: int = 1024,
                              max_buckets: int = 8, swap_dir: Optional[Path] = None,
                              n_workers: Optional[int] = None,
                              memory_budget_gb: float = 8.0) -> tuple:
    """
    Streaming/bucketed alternative to reading the whole transcripts.parquet
    into memory at once before sorting. Prototyped and verified (2026-10-24,
    real 265M-row production data) to produce row-identical output to the
    direct read+build_spatial_block_index() path, at a fraction of the peak
    memory (15.36GB vs an inferred ~46-48GB) -- the OOM report this traces
    back to was a 418M-row section expected to exceed even that; this is the
    path meant to let sections past any single box's RAM load at all.

    FIRST-PASS INTEGRATION NOTE: this still concatenates every bucket's
    sorted result into one in-memory DataFrame before returning, matching
    _load_transcripts()'s existing contract (callers expect one complete
    DataFrame back). That means THIS function does not yet reduce
    _load_transcripts()'s own peak memory in the real pipeline -- it
    validates that the streaming/bucketing mechanism produces an identical
    bundle through the real save_sample() path. The actual memory win needs
    a later increment that pushes the piecemeal per-bucket writes all the
    way to the zarr output instead of concatenating here (see
    project_platpy_streaming_ingestion memory for the full design).

    Bucketing is now the 2D density-balanced ("Mondrian") split from
    _mondrian_split() -- supersedes the original 1D y-only bucketing, which
    left a 32x row-count spread between the smallest and largest bucket on
    real data; the 2D split gets that down to ~1.2-2.3x. Correctness note:
    unlike the 1D version, bucket (and therefore final row) order no longer
    matches build_spatial_block_index()'s single-pass global (by, bx) sort
    exactly -- Mondrian leaves aren't simple contiguous y-strips, so
    concatenating them doesn't reproduce the same row permutation. This
    doesn't affect correctness: block_index entries carry their own
    bx/by/row_start/row_end regardless of physical block order, so lookups
    are unaffected -- it just means "row-for-row identical to the default
    path" is no longer the right correctness check; compare as row sets
    instead (see project_platpy_streaming_ingestion memory).

    Pass 3 (per-bucket sort+write) runs in parallel across worker processes,
    gated by memory_budget_gb: worker count is picked from the calibrated
    cost formula peak_GB ~= 0.8 + 0.25 * rows_millions (~5% safety margin
    over the measured linear fit), applied to the MAX bucket's row count --
    not the average, since real bucket sizes can spread >3x (confirmed on
    production data) and gating on the average would risk over-committing.
    Prototyped and verified (2026-10-24, output-cropped-270_WT, 7 buckets):
    3.5x wall-clock speedup over sequential Pass 3 at no increase in peak
    memory (workers stayed under the formula's own estimate), with
    row-for-row and block-index-identical output. Pass n_workers explicitly
    to override the memory-based estimate (e.g. n_workers=1 forces the old
    sequential behavior).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import shutil

    swap_dir = Path(swap_dir) if swap_dir is not None else tx_path.parent / "_stream_sort_tmp"
    swap_dir.mkdir(parents=True, exist_ok=True)
    for f in swap_dir.iterdir():
        f.unlink()

    try:
        # Pass 1: cheap x+y density scan, then a 2D density-balanced
        # ("Mondrian") split into buckets of roughly even row count --
        # replaces the earlier 1D y-only bucketing, which left a 32x
        # spread between the smallest and largest bucket on real data;
        # this gets that down to ~1.2-2.3x (see _mondrian_split()).
        block_um = chunk_px * mpp
        density = _build_density_map(tx_path, block_um)
        total_rows = sum(density.values())
        target_rows = max(1, -(-total_rows // max_buckets))  # ceil div
        leaves = _mondrian_split(density, target_rows)
        actual_n_buckets = len(leaves)

        # Pack (bx, by) -> bucket id into a single int64-keyed dict so Pass 2
        # can route rows with a vectorized pandas .map() instead of a
        # per-row Python loop.
        block_to_bucket = {}
        for b, leaf in enumerate(leaves):
            for (bx, by) in leaf['blocks']:
                block_to_bucket[bx * 2_000_000_000 + by] = b

        logger.info(f"     + Streaming load: {actual_n_buckets} density-balanced buckets "
                     f"(block_um={block_um:.1f}, requested max {max_buckets}, "
                     f"target_rows={target_rows:,})")

        # Pass 2: stream, route each batch's rows to their bucket's own
        # temp parquet file.
        writers = {}
        pf = pq.ParquetFile(tx_path)
        for batch in pf.iter_batches(batch_size=2_000_000, columns=cols_to_read):
            bdf = batch.to_pandas()
            bx = np.floor(bdf['x_location'].to_numpy() / block_um).astype(np.int64)
            by = np.floor(bdf['y_location'].to_numpy() / block_um).astype(np.int64)
            keys = bx * 2_000_000_000 + by
            bucket_id = pd.Series(keys).map(block_to_bucket).to_numpy()
            for b in range(actual_n_buckets):
                mask = bucket_id == b
                if not mask.any():
                    continue
                sub = bdf[mask]
                path = swap_dir / f"bucket_{b}.parquet"
                table = pa.Table.from_pandas(sub, preserve_index=False)
                if b not in writers:
                    writers[b] = pq.ParquetWriter(str(path), table.schema)
                writers[b].write_table(table)
        for w in writers.values():
            w.close()

        # Pass 3: sort each bucket (reusing the existing, already-committed
        # DataFrame-mode build_spatial_block_index() unchanged -- each
        # bucket is just a smaller version of the same problem). Workers
        # write their sorted bucket straight to its own parquet file rather
        # than returning the DataFrame, keeping IPC cheap; the main process
        # reads each back in bucket order, shifts block_index row offsets by
        # the running total, and accumulates.
        bucket_paths = {b: swap_dir / f"bucket_{b}.parquet"
                         for b in range(actual_n_buckets) if (swap_dir / f"bucket_{b}.parquet").exists()}
        sorted_dir = swap_dir / "_sorted"
        sorted_dir.mkdir(exist_ok=True)

        if n_workers is None:
            max_rows = max(pq.ParquetFile(p).metadata.num_rows for p in bucket_paths.values())
            est_peak_per_worker_gb = _estimate_sort_peak_gb(max_rows)
            n_workers = max(1, min(len(bucket_paths), os.cpu_count() or 1,
                                    int(memory_budget_gb / est_peak_per_worker_gb)))
        n_workers = max(1, min(n_workers, len(bucket_paths)))
        logger.info(f"     + Streaming load: sorting {len(bucket_paths)} buckets "
                     f"with {n_workers} worker(s) (memory_budget={memory_budget_gb:.1f}GB)")

        bucket_results = {}
        if n_workers == 1:
            for b, path in bucket_paths.items():
                sorted_path = sorted_dir / f"sorted_{b}.parquet"
                n_rows, bi_json = _sort_one_bucket_worker(path, sorted_path, rename_map, mpp, chunk_px)
                bucket_results[b] = (sorted_path, n_rows, bi_json)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = {}
                for b, path in bucket_paths.items():
                    sorted_path = sorted_dir / f"sorted_{b}.parquet"
                    fut = ex.submit(_sort_one_bucket_worker, path, sorted_path, rename_map, mpp, chunk_px)
                    futures[fut] = (b, sorted_path)
                for fut in as_completed(futures):
                    b, sorted_path = futures[fut]
                    n_rows, bi_json = fut.result()
                    bucket_results[b] = (sorted_path, n_rows, bi_json)

        all_sorted = []
        all_blocks = []
        row_offset = 0
        for b in sorted(bucket_results):
            sorted_path, n_rows, bi_json = bucket_results[b]
            bi = json.loads(bi_json)
            for blk in bi['blocks']:
                blk['row_start'] += row_offset
                blk['row_end'] += row_offset
                all_blocks.append(blk)
            all_sorted.append(pd.read_parquet(sorted_path))
            row_offset += n_rows

        final_df = pd.concat(all_sorted, ignore_index=True)
        final_block_index_json = json.dumps({
            'block_size_um': float(block_um),
            'mpp': float(mpp),
            'chunk_px': int(chunk_px),
            'blocks': all_blocks,
        })
        return final_df, final_block_index_json
    finally:
        shutil.rmtree(swap_dir, ignore_errors=True)


def _load_transcripts(x_dir: Path, streaming: Union[bool, str] = "auto", max_buckets: int = 8,
                       swap_dir: Optional[Path] = None, n_workers: Optional[int] = None,
                       memory_budget_gb: float = 8.0,
                       auto_streaming_margin: float = 0.5) -> Optional[OmicChannel]:
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

        # ==========================================
        # platpy STANDARDIZATION
        # Translate Xenium coordinates to standard 'x', 'y', and 'feature'
        # ==========================================
        rename_map = {'x_location': 'x', 'y_location': 'y', 'feature_name': 'feature'}

        if tx_path.suffix == '.parquet':
            # Older Xenium Ranger outputs may not have is_gene/codeword_category
            # (newer-only columns) -- check the schema first rather than
            # letting read_parquet's columns= raise on a missing column.
            import pyarrow.parquet as pq
            available = set(pq.ParquetFile(tx_path).schema.names)
            cols_to_read = core_cols + [c for c in optional_cols if c in available]
            rename_map = {k: v for k, v in rename_map.items() if k in available}

            if streaming == "auto":
                # Row count is free from parquet metadata -- no need to read
                # any actual data to make this call. Memory is the priority
                # here, not speed: auto_streaming_margin (default 0.5) means
                # the fast non-streaming path is only chosen when its
                # estimated peak would use at most half of currently
                # *available* memory, leaving a deliberately generous
                # margin -- the calibrated formula only accounts for the
                # transcript sort itself, not whatever else is resident in
                # the same process (e.g. an already-loaded image pyramid
                # level), and this is a one-shot estimate with no visibility
                # into other processes that might start up concurrently.
                total_rows = pq.ParquetFile(tx_path).metadata.num_rows
                available_gb = psutil.virtual_memory().available / 1e9
                est_fast_path_gb = _estimate_sort_peak_gb(total_rows)
                streaming = est_fast_path_gb > available_gb * auto_streaming_margin
                logger.info(
                    f"   - Auto-selecting transcript load path: {total_rows:,} rows, "
                    f"estimated fast-path peak {est_fast_path_gb:.1f}GB vs "
                    f"{available_gb:.1f}GB available ({auto_streaming_margin:.0%} margin) "
                    f"-> {'streaming' if streaming else 'fast (non-streaming)'}"
                )

            if streaming:
                # Opt-in alternative to reading the whole file into memory at
                # once -- streams + buckets + sorts in bounded-memory pieces.
                # See _stream_sort_transcripts()'s docstring: this first-pass
                # integration still concatenates to one DataFrame at the end
                # (matching this function's existing contract), so it proves
                # the mechanism produces an identical bundle through the real
                # save_sample() path -- it does not yet reduce this
                # function's own peak memory. That needs a later increment.
                df, block_index_json = _stream_sort_transcripts(
                    tx_path, cols_to_read, rename_map, mpp=0.2125,
                    max_buckets=max_buckets, swap_dir=swap_dir,
                    n_workers=n_workers, memory_budget_gb=memory_budget_gb,
                )
                _bi = json.loads(block_index_json)
                logger.info(f"     + Streaming block index: {len(_bi['blocks'])} blocks "
                            f"({_bi['block_size_um']:.1f} µm each)")
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

            # Parquet allows us to read only the columns we want, saving tons of memory!
            df = pd.read_parquet(tx_path, columns=cols_to_read)
        else:
            # CSV requires reading everything, then dropping. Streaming mode
            # isn't implemented for this fallback (CSV has no row-group-wise
            # reader equivalent to pyarrow's parquet batching) -- falls
            # through to the normal path regardless of the streaming flag.
            df = pd.read_csv(tx_path)
            df = df[[c for c in core_cols + optional_cols if c in df.columns]]

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


def load_xenium(x_dir: str | Path, name: str = "XeniumSample", load_boundaries: bool = True, force_8bit=True,
                 debug: bool = False, streaming_load: Union[bool, str] = "auto", max_buckets: int = 8,
                 swap_dir: Optional[str | Path] = None, n_workers: Optional[int] = None,
                 memory_budget_gb: float = 8.0, auto_streaming_margin: float = 0.5) -> SampleSet:
    """
    Ingests a Xenium output directory into the platpy framework.
    Maps morphology images to the Visium multi-res standard using Dask/Zarr
    and Absolute Spatial Truth (Affine Matrices).

    streaming_load: alternative transcript-loading path for sections too
        large to comfortably read+sort in one piece -- streams the raw
        parquet in batches, routes rows into max_buckets density-balanced
        ("Mondrian") groups, and sorts+writes each bucket independently
        (in parallel -- see n_workers/memory_budget_gb) instead of the
        whole table at once. Verified (2026-10-24) to produce a correct
        bundle (same row content, self-consistent block_index -- NOT
        row-for-row identical order to the non-streaming path; see
        project_platpy_streaming_ingestion memory for why that's expected)
        on real production data.

        Default "auto": picks the non-streaming path only when its
        estimated peak memory (from row count alone, free from parquet
        metadata -- no data read needed to decide) would use at most
        auto_streaming_margin of currently *available* memory; otherwise
        picks streaming. Deliberately conservative by default (memory
        safety prioritized over speed) -- pass True/False to force a path
        regardless of the estimate.

        First-pass integration note: still concatenates to one final
        DataFrame before returning (matching the default path's contract),
        so this validates bundle correctness through the real save_sample()
        pipeline but does not yet reduce peak memory in the real pipeline
        -- see project_platpy_streaming_ingestion memory for what's still
        needed (piecemeal writes pushed to the zarr output) before this is
        the memory-saving path it's ultimately meant to be. Parquet only --
        ignored (always non-streaming) for the older CSV fallback format.
    max_buckets: only used when streaming_load resolves to True.
    swap_dir: temp directory for streaming_load's intermediate per-bucket
        files; defaults to a subfolder next to the raw transcripts.parquet.
    n_workers: only used when streaming_load resolves to True. Number of
        parallel processes for Pass 3 (per-bucket sort+write). Defaults to
        None, which auto-picks a count from memory_budget_gb and the
        calibrated per-bucket memory cost (gated on the largest actual
        bucket, not the average). Pass 1 to force the original sequential
        behavior.
    memory_budget_gb: only used when streaming_load resolves to True and
        n_workers is None -- caps how many Pass-3 workers run concurrently
        so their combined estimated peak memory stays under this budget.
        Verified (2026-10-24, 7-bucket real-data run) to give a 3.5x Pass-3
        speedup at no increase in peak memory over the sequential path.
    auto_streaming_margin: only used when streaming_load="auto". Fraction
        of currently-available memory the non-streaming path's estimated
        peak must stay under for "auto" to choose it (default 0.5 -- the
        fast path is only chosen when it's estimated to use at most half of
        what's free, leaving a deliberately generous margin since the
        estimate only accounts for the transcript sort itself, not whatever
        else shares the process, e.g. an already-loaded image pyramid
        level). Lower = more conservative (picks streaming more often).

    WINDOWS CALLERS: when streaming_load=True spawns more than one worker,
    it uses ProcessPoolExecutor, whose Windows ('spawn') start method
    re-imports the calling script's __main__ module in every child process.
    Any top-level code in that script -- not just this call -- runs again in
    each worker. Guard your script's own top-level work with
    `if __name__ == "__main__":` the same way any multiprocessing-using
    script must. Confirmed directly: an unguarded test script's load/save
    calls re-ran inside the workers, racing the real run and corrupting the
    shared output path (concurrent writers on the same morph.zarr) before
    crashing one worker outright (BrokenProcessPool). Not an issue from a
    notebook cell or REPL, or when n_workers resolves to 1.
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
            thumb = channel.image  # fall back to base if no lower-res pyramid level exists
            source_mpp = channel.mpp
            lower_res_keys = [k for k in channel.pyramid if abs(k - channel.mpp) > 1e-6]
            if lower_res_keys:
                source_mpp = max(lower_res_keys)
                thumb = channel.pyramid[source_mpp]
            if hasattr(thumb, 'compute'):
                thumb = thumb.compute()
            flat = thumb[thumb > 0].flatten() if np.any(thumb > 0) else thumb.flatten()
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
    tx_channel = _load_transcripts(x_dir, streaming=streaming_load, max_buckets=max_buckets, swap_dir=swap_dir,
                                    n_workers=n_workers, memory_budget_gb=memory_budget_gb,
                                    auto_streaming_margin=auto_streaming_margin)
    if tx_channel:
        sample.omics['transcripts'] = tx_channel
        logger.info(f"   - Successfully loaded {len(tx_channel.data):,} transcripts.")


    return sample