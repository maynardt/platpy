from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import numpy as np

from .image_channel import ImageChannel
from .omic_channel import OmicChannel
from .cell_channel import CellChannel
from .roi_channel import ROIChannel


def _cropped_omic_metadata(metadata, n_obs):
    """
    Recompute `total_transcripts` for a cropped OmicChannel instead of
    carrying the parent's pre-crop count forward unchanged via .copy() --
    xenium_loader.py sets it once at ingestion as len(df), so without this
    every extracted sub-region/section bundle would report its *parent
    slide's* full count forever, never its own. Preserves the original
    pre-crop count as `source_total_transcripts`, set only the first time a
    channel is cropped so re-cropping an already-cropped SampleSet doesn't
    overwrite it with an intermediate count.
    """
    meta = metadata.copy()
    if 'source_total_transcripts' not in meta:
        meta['source_total_transcripts'] = meta.get('total_transcripts')
    meta['total_transcripts'] = n_obs
    return meta


def _boundary_obs(b):
    """
    Return the (cell_id, x, y, ...) table underlying a CellChannel.boundaries
    value, regardless of whether it's still the plain DataFrame produced by a
    fresh load_xenium() or the AnnData produced by a save_sample()/
    load_sample() round-trip (write_zarr/read_zarr always go through AnnData).
    `b` itself stays indexable by the same boolean row mask either way.
    """
    return b.obs if hasattr(b, 'obs') else b


@dataclass
class SampleSet:
    """The core organizational unit of platpy."""
    name: str
    images: Dict[str, ImageChannel] = field(default_factory=dict)
    omics:  Dict[str, OmicChannel]  = field(default_factory=dict)
    cells:  Dict[str, CellChannel]  = field(default_factory=dict)
    rois:   Dict[str, ROIChannel]   = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Preferred render width in pixels. Passed to ViewPort automatically via
    # sample.viewport(). Change this once to resize all subsequent renders.
    screen_width: int = 1024

    # Default orientation applied to every sample.viewport() call.
    # Set once after loading; individual calls can still override per-kwarg.
    # Example: sample.view_defaults = {'flip_y': True, 'rotation_deg': 90}
    view_defaults: Dict[str, Any] = field(default_factory=dict)

    # Set by load_sample() to prevent redundant re-saves in notebook guard cells.
    _already_saved: bool = field(default=False, repr=False)

    def add_image(self, channel: ImageChannel): self.images[channel.name] = channel
    def add_omic(self, channel: OmicChannel):   self.omics[channel.name] = channel
    def add_cell(self, channel: CellChannel):   self.cells[channel.name] = channel
    def add_roi(self, channel: ROIChannel):     self.rois[channel.name] = channel

    def extract_roi(
        self,
        min_x_um: float,
        max_x_um: float,
        min_y_um: float,
        max_y_um: float,
        new_name: Optional[str] = None,
        cell_margin_um: float = 100.0
    ) -> 'SampleSet':
        """
        Extract a spatial sub-region as a new SampleSet.

        Images are cropped to the bounding box — spatial metadata is updated so
        the result still knows its absolute position in the master slide's
        coordinate frame. Transcript/omic DataFrames are filtered by x/y.
        AnnData omics are filtered by obsm['spatial'].

        Cell boundary polygons are kept whole-or-nothing per cell (never split
        a single cell's vertices across the ROI edge), and cell_margin_um
        expands the boundary test beyond the strict ROI so cells whose
        centroid sits just outside it but whose body still overlaps aren't
        silently dropped — the same edge problem DataPort's buffer_um solves
        for spatial-block queries.

        The returned SampleSet can be saved and reloaded independently with
        save_sample() / load_sample().

        Args:
            min_x_um, max_x_um, min_y_um, max_y_um: ROI bounds in microns.
            new_name: Name for the extracted SampleSet. Defaults to self.name + '_roi'.
            cell_margin_um: Margin (microns) added around the ROI when deciding
                whether a cell's boundary polygon overlaps it. Default 100µm
                (~a few cell diameters) — this is a space-saving filter, not a
                precise clip, so a generous margin costs little and avoids
                losing edge cells.
        """
        out = SampleSet(name=new_name or f"{self.name}_roi")
        out.screen_width  = self.screen_width
        out.view_defaults = dict(self.view_defaults)

        for ch_name, chan in self.images.items():
            try:
                cropped = chan.crop_to_micron_roi(
                    min_x_um=min_x_um, max_x_um=max_x_um,
                    min_y_um=min_y_um, max_y_um=max_y_um
                )
                # crop_to_micron_roi() appends a '_crop' suffix to the channel's own
                # .name, meant for transient render-time crops. Here the result is a
                # permanent SampleSet image, so keep its identity matching the dict key.
                cropped.name = ch_name
                out.images[ch_name] = cropped
            except ValueError:
                pass  # channel doesn't overlap this ROI — skip silently

        for om_name, omic in self.omics.items():
            platform = omic.metadata.get('platform', 'visium')
            if platform == 'xenium':
                import anndata as ad
                from ..lazy_transcripts import LazyTranscripts

                # Parent's block_index (mpp/chunk_px) is what the fresh index
                # below gets rebuilt from -- grab it before it's out of reach,
                # regardless of which branch produces `extracted`.
                parent_uns = omic.data.uns if hasattr(omic.data, 'uns') else {}
                parent_block_index = parent_uns.get('block_index')

                if isinstance(omic.data, LazyTranscripts):
                    # Lazy parent -- only materialize the blocks overlapping this
                    # crop, not the whole (possibly much larger) source section.
                    hit_ranges = omic.data.row_ranges_for_block_index(
                        min_x_um, max_x_um, min_y_um, max_y_um
                    )
                    df = omic.data.materialize(hit_ranges)
                    mask = (
                        (df['x'].to_numpy() >= min_x_um) & (df['x'].to_numpy() <= max_x_um) &
                        (df['y'].to_numpy() >= min_y_um) & (df['y'].to_numpy() <= max_y_um)
                    )
                    df = df.loc[mask].reset_index(drop=True)
                    df.index = df.index.astype(str)
                    extracted = ad.AnnData(obs=df)
                    extracted.obsm['spatial'] = df[['x', 'y']].to_numpy()
                else:
                    coords = omic.data.obsm['spatial']
                    mask = (
                        (coords[:, 0] >= min_x_um) & (coords[:, 0] <= max_x_um) &
                        (coords[:, 1] >= min_y_um) & (coords[:, 1] <= max_y_um)
                    )
                    extracted = omic.data[mask].copy()

                # Block index row ranges always refer to the pre-crop source
                # (master slide, or whatever the lazily-loaded parent covered),
                # never the crop itself -- always rebuild fresh here rather than
                # only when `extracted` happened to inherit one, since the
                # lazy-materialize branch above starts from a bare new AnnData
                # with no uns of its own.
                if parent_block_index:
                    import json
                    from ..io_utils import build_spatial_block_index
                    bi = json.loads(parent_block_index)
                    extracted = build_spatial_block_index(
                        extracted, mpp=bi['mpp'], chunk_px=bi['chunk_px'], gene_col='feature'
                    )
                out.omics[om_name] = OmicChannel(
                    name=om_name,
                    data=extracted,
                    metadata=_cropped_omic_metadata(omic.metadata, extracted.n_obs)
                )
            else:
                coords = omic.data.obsm['spatial']
                mask = (
                    (coords[:, 0] >= min_x_um) & (coords[:, 0] <= max_x_um) &
                    (coords[:, 1] >= min_y_um) & (coords[:, 1] <= max_y_um)
                )
                cropped_data = omic.data[mask].copy()

                # Each pyramid level has its own (different) spot/row count --
                # a coarser bin means fewer spots over the same area -- so the
                # crop mask has to be computed per-level against that level's
                # own coordinates. Reusing omic.data's mask (sized for
                # whichever single level .data happens to point to) across
                # every other level crashes with a shape mismatch the moment
                # more than one bin level is loaded, which is the normal case.
                cropped_pyramid = {}
                for bin_um, level_data in omic.pyramid.items():
                    level_coords = level_data.obsm['spatial']
                    level_mask = (
                        (level_coords[:, 0] >= min_x_um) & (level_coords[:, 0] <= max_x_um) &
                        (level_coords[:, 1] >= min_y_um) & (level_coords[:, 1] <= max_y_um)
                    )
                    cropped_pyramid[bin_um] = level_data[level_mask].copy()

                out.omics[om_name] = OmicChannel(
                    name=om_name,
                    data=cropped_data,
                    spatial_meta=omic.spatial_meta,
                    metadata=_cropped_omic_metadata(omic.metadata, cropped_data.n_obs),
                    pyramid=cropped_pyramid
                )

        margin_min_x, margin_max_x = min_x_um - cell_margin_um, max_x_um + cell_margin_um
        margin_min_y, margin_max_y = min_y_um - cell_margin_um, max_y_um + cell_margin_um

        from ..lazy_value import resolve as _resolve_lazy

        out.cells = {}
        for cell_name, cell_chan in self.cells.items():
            b = _resolve_lazy(cell_chan.boundaries)
            b_obs = _boundary_obs(b)
            if b_obs is not None and {'cell_id', 'x', 'y'}.issubset(b_obs.columns):
                # Boundaries are one row per polygon *vertex* -- filter by whole
                # cell_id, not by row, or a kept cell's polygon would be
                # silently mangled (only the vertices that happened to fall
                # inside the box, rather than its true shape).
                vertex_in_margin = (
                    (b_obs['x'] >= margin_min_x) & (b_obs['x'] <= margin_max_x) &
                    (b_obs['y'] >= margin_min_y) & (b_obs['y'] <= margin_max_y)
                )
                keep_ids = b_obs.loc[vertex_in_margin, 'cell_id'].unique()
                row_mask = b_obs['cell_id'].isin(keep_ids).to_numpy()

                summary = cell_chan.summary
                kept_summary = (summary.loc[summary.index.isin(keep_ids)].copy()
                                if summary is not None else None)

                out.cells[cell_name] = CellChannel(
                    name=cell_chan.name,
                    data=cell_chan.data,
                    boundaries=b[row_mask].copy(),
                    summary=kept_summary,
                    spatial_meta=cell_chan.spatial_meta,
                    metadata=cell_chan.metadata.copy()
                )
            else:
                # Unrecognized boundary shape (no cell_id/x/y columns) -- can't
                # safely margin-filter without knowing the schema, so keep the
                # channel as-is rather than silently dropping or corrupting it.
                out.cells[cell_name] = cell_chan

        out.rois = dict(self.rois)

        # Record the crop actually used as provenance on the returned bundle --
        # previously `out.rois` only ever carried over whatever ROIs already
        # existed on the parent, with no record of the bounds that produced
        # `out` itself. A reloaded bundle had no built-in way to answer "what
        # region of its parent was I cropped from" without an external manifest.
        out.add_roi(ROIChannel(
            name='extraction_bounds',
            polygons={'extraction_bounds': np.array([
                [min_x_um, min_y_um], [max_x_um, min_y_um],
                [max_x_um, max_y_um], [min_x_um, max_y_um],
            ])},
            metadata={
                'source_sample': self.name,
                'min_x_um': float(min_x_um), 'max_x_um': float(max_x_um),
                'min_y_um': float(min_y_um), 'max_y_um': float(max_y_um),
            }
        ))
        return out

    def extract_polygon_roi(
        self,
        polygon_um,
        new_name: Optional[str] = None,
        cell_margin_um: float = 100.0
    ) -> 'SampleSet':
        """
        Extract a polygon-shaped sub-region as a new SampleSet.

        Two-pass: first extracts the polygon's axis-aligned bounding box via
        extract_roi() unchanged (reusing all of its existing crop/filter
        logic as-is), then blanks everything outside the actual polygon --
        image pixels are zeroed at every pyramid level, transcript/spot rows
        and cell boundaries outside the polygon are dropped. Useful when
        tissue sections sit close enough together on a slide that a bounding
        box tight enough to capture one section would still clip into a
        neighbor.

        Args:
            polygon_um: [N, 2] array of (x, y) vertices in absolute microns.
            new_name: Name for the extracted SampleSet. Defaults to self.name + '_roi'.
            cell_margin_um: Same meaning as extract_roi()'s for the initial
                bbox pass. Also used as a buffer distance around the polygon
                itself for the final whole-cell filter, so a cell whose
                centroid sits just outside the polygon but whose body still
                overlaps isn't silently dropped.
        """
        import json
        import cv2
        from matplotlib.path import Path as MplPath
        from shapely.geometry import Polygon as SPoly
        from ..transform_utils import microns_to_pixels
        from ..io_utils import build_spatial_block_index

        poly = np.asarray(polygon_um, dtype=float)
        min_x, max_x = poly[:, 0].min(), poly[:, 0].max()
        min_y, max_y = poly[:, 1].min(), poly[:, 1].max()

        out = self.extract_roi(
            min_x, max_x, min_y, max_y, new_name=new_name, cell_margin_um=cell_margin_um
        )

        # --- Blank image pixels outside the polygon, every pyramid level ---
        # Raw-loaded channels' own transforms are scale + translate only (no
        # rotation -- that's a ViewPort render-time concept), so the polygon's
        # micron vertices convert cleanly to each level's own pixel space.
        for chan in out.images.values():
            levels = {chan.mpp: chan.image, **chan.pyramid}
            for level_mpp, arr in levels.items():
                level_meta = chan.get_spatial_meta_for_level(level_mpp)
                h, w = arr.shape[:2]
                poly_px = microns_to_pixels(level_meta.transform_matrix, poly)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly_px.round().astype(np.int32)], 1)
                mask = mask.astype(bool)

                is_dask = hasattr(arr, 'dask')
                if is_dask:
                    import dask.array as da
                    blanked = da.where(mask[..., None] if arr.ndim == 3 else mask, arr, 0)
                else:
                    blanked = np.where(mask[..., None] if arr.ndim == 3 else mask, arr, 0)

                if level_mpp == chan.mpp:
                    chan.image = blanked
                else:
                    chan.pyramid[level_mpp] = blanked

        # --- Drop omic rows (transcripts / spot centers) outside the polygon ---
        path = MplPath(poly)
        for omic in out.omics.values():
            coords = omic.data.obsm['spatial']
            inside = path.contains_points(coords)
            omic.data = omic.data[inside].copy()
            if omic.pyramid:
                omic.pyramid = {k: v[inside] for k, v in omic.pyramid.items()}
            # Bbox pass already rebuilt this once in extract_roi(); rebuild
            # again now that the polygon has tightened the row set further.
            if 'block_index' in omic.data.uns:
                bi = json.loads(omic.data.uns['block_index'])
                omic.data = build_spatial_block_index(
                    omic.data, mpp=bi['mpp'], chunk_px=bi['chunk_px'], gene_col='feature'
                )
            # extract_roi()'s bbox pass already set total_transcripts once;
            # the polygon pass above tightens the row set further, so redo it
            # (source_total_transcripts is untouched -- already set to the
            # true pre-crop count by that first pass).
            omic.metadata = _cropped_omic_metadata(omic.metadata, omic.data.n_obs)

        # --- Filter cell boundaries by polygon too (whole-cell rule) ---
        buffered = SPoly(poly).buffer(cell_margin_um)
        if buffered.geom_type == 'MultiPolygon':
            buffered = max(buffered.geoms, key=lambda g: g.area)
        buffered_path = MplPath(np.array(buffered.exterior.coords))

        for cell_chan in out.cells.values():
            b = cell_chan.boundaries
            b_obs = _boundary_obs(b)
            if b_obs is not None and {'cell_id', 'x', 'y'}.issubset(b_obs.columns):
                vertex_inside = buffered_path.contains_points(b_obs[['x', 'y']].values)
                keep_ids = b_obs.loc[vertex_inside, 'cell_id'].unique()
                row_mask = b_obs['cell_id'].isin(keep_ids).to_numpy()
                cell_chan.boundaries = b[row_mask].copy()
                if cell_chan.summary is not None:
                    cell_chan.summary = cell_chan.summary.loc[
                        cell_chan.summary.index.isin(keep_ids)
                    ].copy()
            # else: unrecognized boundary schema -- leave as-is, same as extract_roi()

        # The inner extract_roi() call above already recorded 'extraction_bounds'
        # as the bbox it used -- overwrite it with the true polygon actually
        # applied here, since that's the real shape of this crop.
        out.add_roi(ROIChannel(
            name='extraction_bounds',
            polygons={'extraction_bounds': poly},
            metadata={'source_sample': self.name, 'shape': 'polygon'}
        ))
        return out

    def viewport(
        self,
        center_um: Tuple[float, float],
        fov_um: Tuple[float, float],
        **kwargs
    ):
        """
        Creates a ViewPort using this sample's screen_width as the default render
        resolution. Override for a single render with target_width_px=N.

        Example:
            vp = sample.viewport(center_um=(3000, 2000), fov_um=(1000, 800))
            vp = sample.viewport(center_um=(3000, 2000), fov_um=(1000, 800), target_width_px=2048)
        """
        from ..viewport import ViewPort

        # view_defaults provides the baseline; kwargs override on a per-call basis
        merged = {**self.view_defaults, **kwargs}
        return ViewPort.create(
            target_width_px=merged.pop('target_width_px', self.screen_width),
            center_um=center_um,
            fov_um=fov_um,
            **merged
        )
