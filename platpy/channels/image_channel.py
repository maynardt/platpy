from dataclasses import dataclass, field
import numpy as np
import dask.array as da
from typing import Union, Tuple, Optional, List, Dict

# platpy imports
from .base_channel import BaseChannel
from platpy.transform_utils import SpatialMetadata

@dataclass
class ImageChannel(BaseChannel):
    """Contains morphology or standard IF images (numpy or dask)."""
    image: Union[np.ndarray, da.Array] = None   #type: ignore
    base_layer: Optional[str] = None            # Name of the physical layer this derives from

    # Multi-scale pyramid: {mpp_float: image_array}
    # Populated on load via generate_pyramid(); renderers call get_optimal_channel() to
    # pick the right level automatically — no manual resolution management needed.
    pyramid: Dict[float, Union[np.ndarray, da.Array]] = field(default_factory=dict)

    # If True (default), pyramid levels are written to disk by save_sample.
    # Set to False for channels that already have companion lower-res channels
    # (e.g. Xenium _fullres, which has _hires / _lowres as separate channels).
    persist_pyramid: bool = True

    def __post_init__(self):
        # Seed the base level if the pyramid wasn't pre-populated by a loader
        if self.image is not None and not self.pyramid:
            self.pyramid[self.mpp] = self.image

    # -------------------------------------------------------------------------
    # Core properties
    # -------------------------------------------------------------------------

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """Proxy to access physical extent from spatial metadata."""
        return self.spatial_meta.extent

    @property
    def mpp(self) -> float:
        """Microns per pixel of the base image."""
        return float(self.spatial_meta.transform_matrix[0, 0])

    # -------------------------------------------------------------------------
    # Pyramid support
    # -------------------------------------------------------------------------

    def generate_pyramid(self, min_width_px: int = 2000, method: str = 'mean'):
        """
        Auto-generates downsampled levels at 2x steps (always from the base image).
        Stops when the output width would drop below min_width_px.
        Levels are stored as uint16 to save RAM relative to 32-bit source data.
        """
        base_img = self.image if hasattr(self.image, 'compute') else da.from_array(self.image)
        _, w = base_img.shape[:2]

        factor = 2
        while w // factor >= min_width_px:
            coarsen_dict = {0: factor, 1: factor}
            if base_img.ndim == 3:
                coarsen_dict[2] = 1  # Never coarsen channel axis

            if method == 'mean':
                downsampled = da.coarsen(np.mean, base_img, coarsen_dict, trim_excess=True)
            elif method == 'max':
                downsampled = da.coarsen(np.max, base_img, coarsen_dict, trim_excess=True)
            else:
                raise ValueError(f"Unknown pyramid method: {method}")

            # Match base dtype: uint16 only when base is > 8-bit (saves RAM vs float32/64).
            # Casting uint8 base levels to uint16 doubles bytes per pixel with no benefit.
            out_dtype = np.uint16 if base_img.dtype.itemsize > 1 else base_img.dtype
            self.pyramid[self.mpp * factor] = downsampled.astype(out_dtype)
            factor *= 2

    def get_optimal_level(self, target_mpp: float) -> Tuple[float, Union[np.ndarray, da.Array]]:
        """
        Returns the best pre-computed array for the requested resolution.
        Always rounds DOWN (picks the largest mpp <= target_mpp) so we never
        lose visual fidelity by upscaling a coarser level than needed.
        """
        if not self.pyramid:
            return self.mpp, self.image

        available = sorted(self.pyramid.keys())
        best_mpp = available[0]  # default to highest res

        for mpp in available:
            if mpp <= target_mpp:
                best_mpp = mpp
            else:
                break

        return best_mpp, self.pyramid[best_mpp]

    def get_spatial_meta_for_level(self, level_mpp: float) -> SpatialMetadata:
        """
        Returns the correct SpatialMetadata for a given pyramid level.
        Scales the transform matrix so physical extent is preserved while
        pixel count decreases. The translation (origin) is unchanged.
        """
        if level_mpp == self.mpp:
            return self.spatial_meta

        scale_factor = level_mpp / self.mpp
        new_matrix = self.spatial_meta.transform_matrix.copy()
        new_matrix[0, 0] *= scale_factor
        new_matrix[1, 1] *= scale_factor
        # new_matrix[:, 2] (translation) is unchanged — origin stays in the same micron position

        new_bounds = None
        if self.spatial_meta.local_bounds is not None:
            lb = self.spatial_meta.local_bounds
            new_bounds = (lb[0] / scale_factor, lb[1] / scale_factor,
                          lb[2] / scale_factor, lb[3] / scale_factor)

        return SpatialMetadata(
            transform_matrix=new_matrix,
            local_bounds=new_bounds,
            unit=self.spatial_meta.unit
        )

    def get_optimal_channel(self, target_mpp: float) -> 'ImageChannel':
        """
        Returns a lightweight ImageChannel wrapping the best pyramid level for
        the requested resolution. Used by ViewPort.render() to auto-select scale
        before the crop → warp pipeline, with no change to that pipeline's logic.
        """
        level_mpp, level_img = self.get_optimal_level(target_mpp)
        return ImageChannel(
            name=self.name,
            image=level_img,
            spatial_meta=self.get_spatial_meta_for_level(level_mpp),
            metadata=self.metadata,
            base_layer=self.base_layer
        )

    # -------------------------------------------------------------------------
    # Display / conversion
    # -------------------------------------------------------------------------

    def __array__(self, dtype=None):
        """Allows ImageChannel to be used directly anywhere a numpy array is expected."""
        arr = self.image.compute() if hasattr(self.image, 'compute') else self.image
        return arr.astype(dtype) if dtype is not None else arr

    def to_8bit(self, p_max: float = 99.9, in_place: bool = False, debug: bool = False):
        """
        Converts all pyramid levels to 8-bit.
        Clip bounds are computed once from the base image so contrast is
        consistent across zoom levels — no independent normalization per level.
        """
        base_arr = self.image.compute() if hasattr(self.image, 'compute') else self.image  #type: ignore
        p_lo = float(np.percentile(base_arr, 0))
        p_hi = float(np.percentile(base_arr, p_max))

        if debug:
            print(f"to_8bit: '{self.name}' | dtype {base_arr.dtype} | "
                  f"clip [{p_lo:.1f}, {p_hi:.1f}] | shape {base_arr.shape}")

        def _apply(arr):
            a = arr.compute() if hasattr(arr, 'compute') else arr
            if abs(p_hi - p_lo) < 1e-5:
                return np.zeros_like(a, dtype=np.uint8)
            clipped = np.clip((a.astype(np.float32) - p_lo) / (p_hi - p_lo), 0.0, 1.0)
            return (clipped * 255).astype(np.uint8)

        new_pyramid = {mpp: _apply(arr) for mpp, arr in self.pyramid.items()}

        if in_place:
            self.pyramid = new_pyramid
            self.image = self.pyramid[self.mpp]
            return self

        new_chan = ImageChannel(
            name=f"{self.name}_8bit",
            image=new_pyramid[self.mpp],
            spatial_meta=self.spatial_meta,
            metadata=self.metadata.copy()
        )
        new_chan.pyramid = new_pyramid
        return new_chan

    # -------------------------------------------------------------------------
    # Spatial operations (unchanged)
    # -------------------------------------------------------------------------

    def crop_to_micron_roi(
        self,
        min_x_um: Optional[Union[float, List, Tuple, np.ndarray]] = None,
        max_x_um: Optional[float] = None,
        min_y_um: Optional[float] = None,
        max_y_um: Optional[float] = None,
        roi_poly_um: Optional[np.ndarray] = None
    ) -> 'ImageChannel':
        """
        Extracts a sub-region of the image based on physical micron coordinates.
        Supports:
          - Explicit bounds:    crop_to_micron_roi(min_x, max_x, min_y, max_y)
          - Polygon array:      crop_to_micron_roi(roi_poly_um=poly)
          - 4-element list:     crop_to_micron_roi([x_min, y_min, x_max, y_max])

        Preserves sub-pixel origin accuracy in the returned channel's spatial metadata.
        Dask-aware: materializes only the cropped region.
        """
        from ..transform_utils import get_poly_bounds
        from numpy.linalg import inv

        # --- 1. Resolve input to (_min_x, _max_x, _min_y, _max_y) ---
        _min_x, _max_x, _min_y, _max_y = None, None, None, None

        if roi_poly_um is not None:
            _min_x, _max_x, _min_y, _max_y = get_poly_bounds(roi_poly_um)
        elif isinstance(min_x_um, (list, tuple, np.ndarray)):
            roi = np.array(min_x_um)
            if roi.ndim == 2 and roi.shape[1] == 2:
                _min_x, _max_x, _min_y, _max_y = get_poly_bounds(roi)
            elif roi.ndim == 1 and len(roi) == 4:
                _min_x, _min_y, _max_x, _max_y = roi[0], roi[1], roi[2], roi[3]
            else:
                raise ValueError(
                    "ROI must be an (N,2) polygon or a 4-element list [x_min, y_min, x_max, y_max]."
                )
        else:
            _min_x, _max_x, _min_y, _max_y = min_x_um, max_x_um, min_y_um, max_y_um

        if any(v is None for v in (_min_x, _max_x, _min_y, _max_y)):
            raise ValueError("Must provide 4 explicit bounds or a valid ROI polygon.")

        true_min_x, true_max_x = min(_min_x, _max_x), max(_min_x, _max_x)   #type: ignore
        true_min_y, true_max_y = min(_min_y, _max_y), max(_min_y, _max_y)   #type: ignore

        # --- 2. Map micron bounds to pixel coordinates via inverse transform ---
        M = self.spatial_meta.transform_matrix
        M_inv = inv(M)

        # Use all 4 corners of the micron ROI to compute the correct pixel AABB.
        # Using only the diagonal pair (min,min)→(max,max) fails for rotated images —
        # a rotated native canvas from render_spotset_to_image has a non-axis-aligned
        # transform, so only all-corner inspection gives the right pixel bounds.
        corners_um = np.array([
            [true_min_x, true_min_y, 1.0],
            [true_max_x, true_min_y, 1.0],
            [true_max_x, true_max_y, 1.0],
            [true_min_x, true_max_y, 1.0],
        ])
        corners_px = (M_inv @ corners_um.T).T
        xs = [int(np.floor(corners_px[:, 0].min())), int(np.ceil(corners_px[:, 0].max()))]
        ys = [int(np.floor(corners_px[:, 1].min())), int(np.ceil(corners_px[:, 1].max()))]

        # --- 3. Clamp to actual array bounds ---
        h, w = self.image.shape[:2]
        x0, x1 = max(0, xs[0]), min(w, xs[1])
        y0, y1 = max(0, ys[0]), min(h, ys[1])

        if (x1 - x0) <= 0 or (y1 - y0) <= 0:
            raise ValueError(
                f"Requested ROI ({true_min_x:.1f}-{true_max_x:.1f} x "
                f"{true_min_y:.1f}-{true_max_y:.1f} µm) has no overlap with this image."
            )

        # --- 4. Crop (Dask slicing stays lazy until compute() is called) ---
        cropped_img = self.image[y0:y1, x0:x1]

        # --- 5. Recalculate origin: pixel (x0, y0) maps to which micron coordinate? ---
        new_origin = M @ np.array([x0, y0, 1.0])
        new_M = M.copy()
        new_M[0, 2] = new_origin[0]
        new_M[1, 2] = new_origin[1]

        crop_h, crop_w = cropped_img.shape[:2]

        # --- 6. Crop pyramid levels at equivalent pixel coordinates ---
        base_mpp = float(M[0, 0])
        cropped_pyramid = {}
        for level_mpp, level_arr in self.pyramid.items():
            if abs(level_mpp - base_mpp) < 1e-9:
                continue  # base level is handled by image above
            factor = level_mpp / base_mpp
            lh, lw = level_arr.shape[:2]
            lx0 = max(0, int(x0 / factor))
            lx1 = min(lw, int(np.ceil(x1 / factor)))
            ly0 = max(0, int(y0 / factor))
            ly1 = min(lh, int(np.ceil(y1 / factor)))
            if lx1 > lx0 and ly1 > ly0:
                cropped_pyramid[level_mpp] = level_arr[ly0:ly1, lx0:lx1]

        return ImageChannel(
            name=f"{self.name}_crop",
            image=cropped_img,
            spatial_meta=SpatialMetadata(
                transform_matrix=new_M,
                local_bounds=(0.0, float(crop_w), 0.0, float(crop_h)),
                unit=self.spatial_meta.unit
            ),
            metadata=self.metadata.copy(),
            pyramid=cropped_pyramid
        )

    def match_to_reference(self, reference: 'ImageChannel') -> 'ImageChannel':
        """EASY MODE: Warps this image to perfectly match the physical bounds
        and pixel resolution of a reference canvas."""
        from ..viewport import ViewPort

        vp = ViewPort.from_image(reference)
        vp.add_layer(self, blend_mode='normal')
        return vp.render(output_name=f"{self.name}_matched")

    def adjust_display(
        self,
        percentile_clip: Tuple[float, float] = (0, 100),
        gamma: float = 1.0,
        tint: Optional[str] = None
    ) -> 'ImageChannel':
        """EASY MODE: Adjusts contrast, gamma, and optionally tints a grayscale image."""
        from ..render_utils import normalize_array
        import matplotlib.colors as mcolors

        arr = self.image.compute() if hasattr(self.image, 'compute') else self.image  #type: ignore

        stats = (self.metadata or {}).get('display_stats')
        if stats and stats.get('percentile_keys'):
            p_lo = float(np.interp(percentile_clip[0], stats['percentile_keys'], stats['percentile_values']))
            p_hi = float(np.interp(percentile_clip[1], stats['percentile_keys'], stats['percentile_values']))
            arr_f = np.array(arr, dtype=np.float32)
            if abs(p_hi - p_lo) < 1e-5:
                norm = np.where(arr_f > 0, 1.0, 0.0).astype(np.float32)
            else:
                norm = np.clip((arr_f - p_lo) / (p_hi - p_lo), 0.0, 1.0).astype(np.float32)
        else:
            norm = normalize_array(arr, percentile_clip=percentile_clip)  #type: ignore

        if gamma != 1.0:
            norm = np.power(norm, 1.0 / gamma)

        if tint:
            color = mcolors.to_rgba(tint)
            if norm.ndim == 3 and norm.shape[-1] in (3, 4):
                norm = np.mean(norm[..., :3], axis=-1)
            h, w = norm.shape[:2]
            rgba = np.zeros((h, w, 4), dtype=np.float32)
            rgba[..., 0] = norm * color[0]
            rgba[..., 1] = norm * color[1]
            rgba[..., 2] = norm * color[2]
            rgba[..., 3] = np.ones_like(norm) * color[3]
            norm = rgba

        final_uint8 = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)

        return ImageChannel(
            name=f"{self.name}_display",
            image=final_uint8,
            spatial_meta=self.spatial_meta,
            metadata=self.metadata.copy()
        )
