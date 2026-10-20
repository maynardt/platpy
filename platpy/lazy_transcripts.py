"""
lazy_transcripts.py -- Lazy, dask-backed reader for Xenium's per-transcript
point cloud, bypassing AnnData entirely for this one representation.

Why not just AnnData + anndata.experimental.read_lazy()? Checked directly
(2026-10) against a real saved bundle: read_lazy() does leave the purely
numeric columns (x, y, qv, transcript_id) as genuine dask arrays, but it
eagerly materializes the categorical columns (feature, cell_id -- exactly
what gene/cell filtering depends on) as real pandas Categoricals, plus the
full row index as a numpy object array, on every open. That's not a zarr
storage limitation -- the on-disk layout already stores each categorical
column as a small eager `categories` array (thousands of entries) + a large,
properly-chunked `codes` array (same chunking as the numeric columns) -- it's
specifically read_lazy()'s own choice not to keep codes lazy. Reading the
zarr groups directly, the same way ImageChannel already does for the image
pyramid (da.from_zarr), gets the real laziness read_lazy() doesn't.

This is intentionally NOT an AnnData, and NOT handed to scanpy/squidpy
directly -- platpy always extracts a small, already-filtered, real in-memory
table first (via .materialize()) before anything downstream touches it, so
there's nothing to lose by not preserving AnnData's shape here.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import json
import numpy as np
import pandas as pd
import dask.array as da
import zarr


@dataclass
class LazyCategorical:
    """
    A categorical obs column backed by a small eager `categories` array and a
    large, chunked, lazy `codes` dask array -- mirrors how AnnData/zarr
    already stores these on disk (see module docstring), just read directly
    instead of through AnnData's eager reconstruction.
    """
    categories: np.ndarray   # small (hundreds-thousands of entries), read eagerly
    codes: da.Array          # (n_obs,) int codes, lazy

    def __post_init__(self):
        # Categories are NOT guaranteed sorted on disk (confirmed empirically --
        # Xenium gene panels can be non-lexicographic) -- a dict lookup is
        # correct regardless and still cheap at this size (thousands of
        # entries, not millions).
        self._code_of = {str(v): i for i, v in enumerate(self.categories)}

    def code_for(self, value: str) -> Optional[int]:
        return self._code_of.get(str(value))

    def equals_mask(self, value: str) -> da.Array:
        """Lazy boolean dask mask: True where this column == value."""
        code = self.code_for(value)
        if code is None:
            return da.zeros_like(self.codes, dtype=bool)
        return self.codes == code

    def materialize(self, row_idx: np.ndarray) -> np.ndarray:
        """Real category-label strings for the given row positions."""
        codes_np = np.asarray(self.codes[row_idx].compute())
        return self.categories[codes_np]


@dataclass
class LazyTranscripts:
    """
    Lazy point-cloud view of a saved Xenium transcripts table. Columns stay
    dask-backed (or, for feature/cell_id, LazyCategorical-wrapped) until
    .materialize() is called on a specific row selection -- the only point
    actual bytes get pulled and computed.

    uns (block_index, transcript_display_stats, etc.) is read eagerly at open
    time regardless -- these are small JSON blobs independent of row count,
    the same way they already behave today.
    """
    n_obs: int
    x: da.Array
    y: da.Array
    qv: da.Array
    transcript_id: da.Array
    feature: LazyCategorical
    cell_id: LazyCategorical
    uns: Dict = field(default_factory=dict)

    def row_ranges_for_block_index(self, min_x: float, max_x: float,
                                    min_y: float, max_y: float) -> list:
        """Convenience: resolve a bbox to block_index hit-ranges, or a single
        (0, n_obs) range if no block_index is present (full-scan fallback,
        same behavior as before this existed)."""
        block_index = self.uns.get('block_index')
        if not block_index:
            return [(0, self.n_obs)]
        from .io_utils import query_block_index
        return query_block_index(block_index, min_x, max_x, min_y, max_y)

    def materialize(self, row_ranges: Optional[Sequence[Tuple[int, int]]] = None) -> pd.DataFrame:
        """
        Pull the given row ranges (or everything, if row_ranges is None) into
        a real, small, in-memory DataFrame -- columns: transcript_id, cell_id,
        feature, x, y, qv. This is the one place actual data gets read and
        computed; everything before this call stays lazy.
        """
        if row_ranges is None:
            row_idx = np.arange(self.n_obs)
        elif not row_ranges:
            row_idx = np.array([], dtype=np.int64)
        else:
            row_idx = np.concatenate([np.arange(r0, r1) for r0, r1 in row_ranges])

        if len(row_idx) == 0:
            return pd.DataFrame({
                'transcript_id': pd.array([], dtype='uint64'),
                'cell_id': pd.array([], dtype='object'),
                'feature': pd.array([], dtype='object'),
                'x': pd.array([], dtype='float32'),
                'y': pd.array([], dtype='float32'),
                'qv': pd.array([], dtype='float32'),
            })

        return pd.DataFrame({
            'transcript_id': np.asarray(self.transcript_id[row_idx].compute()),
            'cell_id':       self.cell_id.materialize(row_idx),
            'feature':       self.feature.materialize(row_idx),
            'x':             np.asarray(self.x[row_idx].compute()),
            'y':             np.asarray(self.y[row_idx].compute()),
            'qv':            np.asarray(self.qv[row_idx].compute()),
        })


def _read_lazy_categorical(group: zarr.Group) -> LazyCategorical:
    categories = group['categories'][:]
    codes = da.from_zarr(group['codes'])
    return LazyCategorical(categories=categories, codes=codes)


def load_transcripts_lazy(zarr_path) -> LazyTranscripts:
    """
    Open a saved Xenium transcripts zarr store (as written by save_sample())
    lazily -- reads shape/dtype metadata, the small categorical lookup
    tables, and uns (block_index etc.) eagerly; x/y/qv/transcript_id and the
    feature/cell_id codes stay dask-backed until materialize() is called.
    """
    root = zarr.open_group(str(zarr_path), mode='r')
    obs = root['obs']

    n_obs = obs['x'].shape[0]
    x = da.from_zarr(obs['x'])
    y = da.from_zarr(obs['y'])
    qv = da.from_zarr(obs['qv'])
    transcript_id = da.from_zarr(obs['transcript_id'])
    feature = _read_lazy_categorical(obs['feature'])
    cell_id = _read_lazy_categorical(obs['cell_id'])

    uns = {}
    if 'uns' in root:
        for key in root['uns'].array_keys():
            arr = root['uns'][key]
            val = arr[()]
            # Only 0-d (true scalar) entries get unwrapped to a plain Python
            # value (block_index's JSON string, the opt_*_version ints, etc).
            # Multi-element arrays (e.g. spatial_meta's 3x3 transform matrix)
            # aren't something LazyTranscripts needs -- load_sample() reads
            # those directly off the zarr group itself -- so keep them as-is
            # rather than erroring on .item().
            uns[key] = val.item() if (hasattr(val, 'item') and arr.shape == ()) else val

    return LazyTranscripts(
        n_obs=n_obs, x=x, y=y, qv=qv, transcript_id=transcript_id,
        feature=feature, cell_id=cell_id, uns=uns,
    )
