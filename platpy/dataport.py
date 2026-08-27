from dataclasses import dataclass, field
import numpy as np
from typing import Tuple, Optional, Callable, Generator, Dict, Union
import copy
import warnings

from .channels.image_channel import ImageChannel
from .channels.omic_channel import OmicChannel
from .channels.roi_channel import ROIChannel
from .transform_utils import SpatialMetadata, microns_to_pixels


@dataclass
class DataPort:
    """
    DataPort: The spatial data extraction engine — analytical counterpart to ViewPort.

    Where ViewPort renders layers into a screen-resolution display image, DataPort
    extracts the same spatial regions in their native format for analysis. The ROI
    and coordinate frame are typically inherited directly from a ViewPort, so the
    user can select a region visually and immediately route it to analysis tools.

    Key principles:
      - Native format: uint16 stays uint16, AnnData stays AnnData. No implicit
        normalization, resampling, or colorization unless explicitly requested.
      - Pass-through: analysis functions (CellPose, clustering, etc.) are passed
        IN to DataPort rather than wrapped by it. DataPort handles the spatial
        bookkeeping; the science stays in the calling code.
      - Tiling: large ROIs are broken into manageable chunks automatically.
        Every tile carries its own micron-space extent so results can be mapped
        back to absolute coordinates.

    Typical workflow:
        # 1. User selects ROI visually via ViewPort
        view = ViewPort.create(target_width_px=1000, center_um=(...), fov_um=(...))
        view.add_layer(sample.images['dapi_fullres'], ...)
        rendered = view.render()  # user inspects, decides on ROI

        # 2. Clone spatial state to DataPort — no data loaded yet
        dp = DataPort.from_viewport(view)

        # 3. Extract native data for the ROI
        dapi_crop = dp.extract(sample.images['dapi_fullres'])  # raw uint16, no processing

        # 4. Or tile-process for large ROIs — pass any function in directly
        results = dp.process(sample.images['dapi_fullres'],
                             fn=cellpose_model.eval,
                             tile_size_um=500)
        # results is a list of (extent_um, fn_output) tuples — map back to microns trivially
    """
    roi_extent: Tuple[float, float, float, float]  # (min_x, max_x, min_y, max_y) in microns
    spatial_meta: SpatialMetadata

    # Named data layers — ImageChannel or OmicChannel, keyed by user-assigned name.
    # Mirrors ViewPort's layer stack but stores references to raw channels rather
    # than a render queue. Functions can address layers by name, or auto-route by
    # type (image functions go to ImageChannels, omic functions go to OmicChannels).
    layers: Dict[str, Union[ImageChannel, OmicChannel]] = field(default_factory=dict)

    # Named ROI polygons for spatial stratification.
    # e.g. {'layer_1': poly_um, 'layer_4': poly_um, 'white_matter': poly_um}
    # Used by tiles_by_roi() to iterate anatomical regions rather than a grid.
    rois: Dict[str, ROIChannel] = field(default_factory=dict)

    # =========================================================================
    # LAYER MANAGEMENT
    # =========================================================================

    def add_layer(self, name: str,
                  channel: Union[ImageChannel, OmicChannel]) -> 'DataPort':
        """Add a named data channel. Returns self for chaining."""
        self.layers[name] = channel
        return self

    def add_roi(self, roi: ROIChannel) -> 'DataPort':
        """
        Add a named ROI layer for spatial stratification.
        ROI polygons are in absolute microns — same coordinate system as everything else.
        Multiple ROIs can be loaded (e.g. one per cortical layer) and iterated with
        tiles_by_roi().
        """
        self.rois[roi.name] = roi
        return self

    # =========================================================================
    # CONSTRUCTORS
    # =========================================================================

    @classmethod
    def from_viewport(cls, viewport) -> 'DataPort':
        """
        Create a DataPort anchored to the same ROI as a ViewPort.
        The ViewPort's current camera extent becomes the extraction window.
        No data is loaded — this just clones the spatial state.
        """
        return cls(
            roi_extent=viewport.extent,
            spatial_meta=copy.deepcopy(viewport.spatial_meta)
        )

    @classmethod
    def from_roi(cls, min_x_um: float, max_x_um: float,
                 min_y_um: float, max_y_um: float) -> 'DataPort':
        """
        Create a DataPort directly from a micron-space bounding box,
        without needing a ViewPort first.
        """
        from .transform_utils import create_transform_matrix
        return cls(
            roi_extent=(min_x_um, max_x_um, min_y_um, max_y_um),
            spatial_meta=SpatialMetadata(
                transform_matrix=create_transform_matrix(1.0, 1.0, min_x_um, min_y_um)
            )
        )

    # =========================================================================
    # IMAGE EXTRACTION
    # =========================================================================

    @staticmethod
    def _downsample_if_requested(arr: np.ndarray, native_mpp: float,
                                  downsample_mpp: Optional[float]) -> np.ndarray:
        """
        Shared resize step for extract() and tiles(): if downsample_mpp asks for
        a coarser resolution than the crop's native mpp, resize down to match.
        No-op if downsample_mpp is None or not actually coarser than native.
        """
        if downsample_mpp is None:
            return arr
        if downsample_mpp <= native_mpp:
            return arr
        import cv2
        scale = native_mpp / downsample_mpp
        new_w = max(1, int(arr.shape[1] * scale))
        new_h = max(1, int(arr.shape[0] * scale))
        return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def extract(self, channel: ImageChannel,
                downsample_mpp: Optional[float] = None) -> np.ndarray:
        """
        Extract an ImageChannel's data within the ROI in native format.
        Returns a numpy array — materializes Dask only for the cropped region.

        Args:
            channel:        The ImageChannel to extract from.
            downsample_mpp: Optional explicit target resolution in microns/pixel.
                            Only use when downsampling is deliberate (e.g. to match
                            a model's expected input). Default None = native resolution.
        """
        import dask.array as da

        min_x, max_x, min_y, max_y = self.roi_extent
        cropped = channel.crop_to_micron_roi(
            min_x_um=min_x, max_x_um=max_x,
            min_y_um=min_y, max_y_um=max_y
        )
        arr = cropped.image.compute() if isinstance(cropped.image, da.Array) else cropped.image

        native_mpp = abs(float(cropped.spatial_meta.transform_matrix[0, 0]))
        arr = self._downsample_if_requested(arr, native_mpp, downsample_mpp)

        return arr

    # =========================================================================
    # OMIC EXTRACTION
    # =========================================================================

    def extract_omic(self, channel: OmicChannel, gene: Optional[str] = None):
        """
        Extract OmicChannel data within the ROI in its native format.
        Dispatches to a platform-specific extractor based on channel.metadata['platform'].

        Returns:
            AnnData subset for 'visium' (and similar binned platforms).
            DataFrame subset for 'xenium' (and similar point-cloud platforms).

        Args:
            channel: The OmicChannel to extract from.
            gene:    Optional — filter to a single gene in the result.
        """
        platform = channel.metadata.get('platform', 'visium')
        if platform == 'xenium':
            return self._extract_omic_xenium(channel, gene)
        else:
            return self._extract_omic_visium(channel, gene)

    def _extract_omic_visium(self, channel: OmicChannel, gene: Optional[str] = None):
        """AnnData extraction for binned platforms (Visium, Visium HD, etc.)."""
        min_x, max_x, min_y, max_y = self.roi_extent
        coords = channel.get_active_coords()   # prefers spatial_fine if present
        mask = (
            (coords[:, 0] >= min_x) & (coords[:, 0] <= max_x) &
            (coords[:, 1] >= min_y) & (coords[:, 1] <= max_y)
        )
        subset = channel.data[mask].copy()
        active = coords[mask]                  # fine-aligned coords for this subset
        subset.obs['x_micron'] = active[:, 0]
        subset.obs['y_micron'] = active[:, 1]
        if gene is not None and gene in subset.var_names:
            subset = subset[:, [gene]]
        return subset

    def _extract_omic_xenium(self, channel: OmicChannel, gene: Optional[str] = None,
                             buffer_um: float = 50.0):
        """AnnData extraction for transcript point-cloud platforms (Xenium, etc.).

        Uses the spatial block index (built at load time) to narrow candidates to
        the relevant blocks before doing a precise bbox filter.  Falls back to a
        full scan if no block index is present.

        buffer_um: expand the block-lookup bbox by this amount on all sides so
        transcripts near block edges (whose parent cell may straddle the boundary)
        are not missed.  50 µm ≈ 2-3 cell diameters — conservative but cheap.
        """
        import json
        import numpy as np

        min_x, max_x, min_y, max_y = self.roi_extent
        adata = channel.data

        # --- Block index fast path ---
        if 'block_index' in adata.uns:
            index_data = json.loads(adata.uns['block_index'])
            blocks = index_data['blocks']

            q_min_x = min_x - buffer_um
            q_max_x = max_x + buffer_um
            q_min_y = min_y - buffer_um
            q_max_y = max_y + buffer_um

            hit_ranges = [
                (b['row_start'], b['row_end'])
                for b in blocks
                if b['max_x'] >= q_min_x and b['min_x'] <= q_max_x
                and b['max_y'] >= q_min_y and b['min_y'] <= q_max_y
            ]

            if not hit_ranges:
                empty = adata[np.array([], dtype=int)].copy()
                empty.obs['x_micron'] = np.array([])
                empty.obs['y_micron'] = np.array([])
                return empty

            # Build the fine-filter mask only over hit-block rows — avoids scanning
            # all N rows while also avoiding an expensive intermediate AnnData slice.
            use_fine = channel.use_fine_alignment
            all_coords = (adata.obsm['spatial_fine']
                          if ('spatial_fine' in adata.obsm and use_fine is not False)
                          else adata.obsm['spatial'])
            final_mask = np.zeros(len(adata), dtype=bool)
            for row_start, row_end in hit_ranges:
                c = all_coords[row_start:row_end]
                final_mask[row_start:row_end] = (
                    (c[:, 0] >= min_x) & (c[:, 0] <= max_x) &
                    (c[:, 1] >= min_y) & (c[:, 1] <= max_y)
                )
        else:
            # No block index — full scan fallback
            use_fine = channel.use_fine_alignment
            all_coords = (adata.obsm['spatial_fine']
                          if ('spatial_fine' in adata.obsm and use_fine is not False)
                          else adata.obsm['spatial'])
            final_mask = (
                (all_coords[:, 0] >= min_x) & (all_coords[:, 0] <= max_x) &
                (all_coords[:, 1] >= min_y) & (all_coords[:, 1] <= max_y)
            )

        subset = adata[final_mask].copy()
        active = all_coords[final_mask]
        subset.obs['x_micron'] = active[:, 0]
        subset.obs['y_micron'] = active[:, 1]

        if gene is not None:
            subset = subset[subset.obs['feature'] == gene]

        return subset

    # =========================================================================
    # TILING
    # =========================================================================

    def tiles(self, channel: ImageChannel,
              tile_size_um: float,
              overlap_um: float = 0.0,
              downsample_mpp: Optional[float] = None) -> Generator:
        """
        Generator: yields (tile_extent, tile_array) covering the full ROI.

        tile_extent is (min_x, max_x, min_y, max_y) in absolute microns — use
        this to map any analysis result (e.g. CellPose cell centroids) back to
        the global coordinate system without any extra bookkeeping.

        Tiling itself is only partially implemented: the DataPort can break the
        ROI up into tiles and hand each one to your function, but stitching the
        return values back together (merging/deduplicating results across tile
        boundaries) has not been implemented yet -- that's a future goal. Every
        call warns for this reason; see overlap_um below for the sharpest case
        of why that matters.

        Args:
            channel:        The ImageChannel to tile.
            tile_size_um:   Edge length of each tile in microns.
            overlap_um:     Pads each tile's crop by this many microns on every
                            side, so the function you pass to process() sees
                            context beyond the tile's exact grid boundary (e.g.
                            so a segmentation model doesn't cut cells off at the
                            edge). Since nothing merges or deduplicates results
                            across tiles (see above), an object whose centroid
                            falls in the overlap region between two tiles will
                            be detected independently by each and appear twice.
            downsample_mpp: Optional explicit target resolution in microns/pixel,
                            applied to each tile individually. Same meaning as
                            extract()'s downsample_mpp -- None = native resolution.
        """
        import dask.array as da

        warnings.warn(
            "DataPort analysis has used tiling (tile_size_um set), but results "
            "have not been stitched or deduplicated -- an object spanning a "
            "tile boundary, or falling in an overlap_um region, may appear "
            "more than once across tiles' results. Reconciling that is on the "
            "caller; DataPort does not do it automatically yet.",
            UserWarning, stacklevel=2
        )

        min_x, max_x, min_y, max_y = self.roi_extent
        x = min_x
        while x < max_x:
            tile_max_x = min(x + tile_size_um, max_x)
            y = min_y
            while y < max_y:
                tile_max_y = min(y + tile_size_um, max_y)
                extent = (x - overlap_um, tile_max_x + overlap_um,
                          y - overlap_um, tile_max_y + overlap_um)
                try:
                    cropped = channel.crop_to_micron_roi(
                        min_x_um=extent[0], max_x_um=extent[1],
                        min_y_um=extent[2], max_y_um=extent[3]
                    )
                    arr = cropped.image.compute() if isinstance(cropped.image, da.Array) else cropped.image
                    native_mpp = abs(float(cropped.spatial_meta.transform_matrix[0, 0]))
                    arr = self._downsample_if_requested(arr, native_mpp, downsample_mpp)
                    yield extent, arr
                except ValueError:
                    pass  # tile outside image bounds — skip silently
                y += tile_size_um
            x += tile_size_um

    def crop_channel(self, channel: ImageChannel,
                     roi_channel: Optional[ROIChannel] = None) -> ImageChannel:
        """
        Returns a copy of channel cropped to the DataPort's ROI as a full ImageChannel.

        Unlike extract(), this preserves the spatial metadata wrapper so the result
        can be added back to a SampleSet, passed to a ViewPort, or used anywhere a
        normal ImageChannel is expected.

        If roi_channel is provided, pixels outside its polygons are zeroed. The mask
        is applied to the cropped copy — the original channel is never modified.
        """
        import dask.array as da
        import cv2

        min_x, max_x, min_y, max_y = self.roi_extent
        cropped = channel.crop_to_micron_roi(min_x, max_x, min_y, max_y)

        if roi_channel is None or not roi_channel.polygons:
            return cropped

        # Materialize before pixel-level modification
        arr = cropped.image.compute() if isinstance(cropped.image, da.Array) else cropped.image.copy()

        # Build mask in the cropped tile's local pixel space using the snapped origin
        mask = np.zeros(arr.shape[:2], dtype=np.uint8)

        for poly_um in roi_channel.polygons.values():
            pts = microns_to_pixels(cropped.spatial_meta.transform_matrix, np.array(poly_um)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)

        # Zero pixels outside polygon — broadcast over channel axis if RGB/multichannel
        if arr.ndim == 3:
            arr *= mask[:, :, np.newaxis]
        else:
            arr *= mask

        return ImageChannel(
            name=f"{channel.name}_crop",
            image=arr,
            spatial_meta=cropped.spatial_meta,
            metadata=cropped.metadata
        )

    def tiles_by_roi(self, channel: ImageChannel,
                     roi_name: Optional[str] = None) -> Generator:
        """
        Generator: yields (roi_name, roi_polygon_um, tile_array) for each ROI
        stored on this DataPort (or a single named ROI if roi_name is specified).

        This is the primary tool for anatomically-stratified analysis — e.g. load
        ROIs for cortical layers 1-6, then iterate to extract data from each region
        independently. Extraction is the bounding box of the ROI polygon; the polygon
        itself is returned so callers can apply fine-grained masking if needed.

        TODO: implement polygon masking inside the bounding box crop so pixels
        outside the ROI shape are zeroed rather than just bbox-cropped.
        """
        _ = channel, roi_name  # parameters reserved for implementation
        raise NotImplementedError("tiles_by_roi is not yet implemented.")

    # =========================================================================
    # PASS-THROUGH PROCESSING
    # =========================================================================

    def process(self, channel: ImageChannel,
                fn: Callable,
                tile_size_um: Optional[float] = None,
                overlap_um: float = 0.0,
                downsample_mpp: Optional[float] = None,
                **fn_kwargs) -> list:
        """
        Apply any function to the ROI and return results paired with their
        spatial extents. DataPort doesn't wrap any specific tool — functions
        are passed in like C function pointers.

        If tile_size_um is None, the full ROI is extracted and fn is called once.
        If tile_size_um is set, fn is called on each tile independently.

        Tiling is partially implemented here: the DataPort can break the ROI up
        into tiles and hand each one to fn, but stitching the return values back
        together (merging/deduplicating results across tile boundaries) has not
        been implemented yet -- that's a future goal. Using tile_size_um always
        warns for this reason.

        Args:
            channel:        The ImageChannel to extract from.
            fn:             Any callable: fn(array, **fn_kwargs) -> result
                            e.g. model.eval, a custom segmenter, a lambda.
            tile_size_um:   Optional tile size in microns. None = whole ROI at once.
            overlap_um:     Pads each tile's crop with this much extra context on
                            every side (see tiles()'s docstring) -- does NOT merge
                            or deduplicate results across tiles; an object in the
                            overlap between two tiles will appear in both tiles'
                            results.
            downsample_mpp: Optional target resolution in µm/px, passed to extract()
                            (tile_size_um=None) or to tiles() (tile_size_um set) --
                            applies identically either way, one tile at a time when
                            tiling. Useful when the analysis function expects a
                            specific pixel scale (e.g. Cellpose works best at
                            ~0.5 µm/px for tissue sections).
            **fn_kwargs:    Passed straight through to fn unchanged.

        Returns:
            List of (extent_um, result) tuples. extent_um is in absolute microns
            so any spatial results (centroids, masks, etc.) can be anchored back
            to the global coordinate system without extra work.
        """
        results = []

        if tile_size_um is None:
            arr = self.extract(channel, downsample_mpp=downsample_mpp)
            results.append((self.roi_extent, fn(arr, **fn_kwargs)))
        else:
            for tile_extent, tile_arr in self.tiles(channel, tile_size_um, overlap_um,
                                                      downsample_mpp=downsample_mpp):
                results.append((tile_extent, fn(tile_arr, **fn_kwargs)))

        return results

    @staticmethod
    def mask_to_roi_channel(masks: np.ndarray, extent_um: tuple,
                            name: str = 'cells', prefix: str = 'cell') -> ROIChannel:
        """
        Convert a labeled integer mask to a micron-space ROIChannel.

        Works with any segmentation algorithm that produces an integer mask
        (Cellpose, StarDist, watershed, etc.). extent_um and masks.shape together
        encode the pixel scale — no need to track downsample_mpp separately.

        Args:
            masks:      Labeled integer array (0=background, 1..N=cells).
                        Typically the direct output from dp.process().
            extent_um:  (min_x, max_x, min_y, max_y) in microns — pass directly
                        from dp.process() results or dp.roi_extent.
            name:       Name for the resulting ROIChannel.
            prefix:     Prefix for polygon keys ('cell' → 'cell_1', 'cell_2', ...).

        Returns:
            ROIChannel with one polygon per labeled region in absolute micron space.

        Example:
            results = dp.process(sample.images['neun'], fn=model.eval, downsample_mpp=2)
            extent_um, masks = results[0]
            nuclei_rois = DataPort.mask_to_roi_channel(masks, extent_um, name='nuclei')
        """
        from skimage import measure
        from .transform_utils import create_transform_matrix, pixels_to_microns

        min_x, max_x, min_y, max_y = extent_um
        mpp_x = (max_x - min_x) / masks.shape[1]
        mpp_y = (max_y - min_y) / masks.shape[0]
        T = create_transform_matrix(scale_x=mpp_x, scale_y=mpp_y, tx=min_x, ty=min_y)

        polys = {}
        for label_id in np.unique(masks[masks > 0]):
            contours = measure.find_contours((masks == label_id).astype(np.uint8), 0.5)
            if not contours:
                continue
            contour = max(contours, key=len)
            # skimage returns (row, col) = (y, x); swap to (x, y) before transform
            px_xy = contour[:, ::-1]
            polys[f'{prefix}_{label_id}'] = pixels_to_microns(T, px_xy)

        return ROIChannel(name=name, polygons=polys)
