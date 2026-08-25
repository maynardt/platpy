from dataclasses import dataclass, field
from typing import Any, Tuple, Dict, Optional

# platpy imports
from .base_channel import BaseChannel
from .image_channel import ImageChannel
from platpy.transform_utils import SpatialMetadata

@dataclass
class OmicChannel(BaseChannel):
    """Contains transcriptomic point clouds (Xenium) or spot matrices (Visium HD)."""
    data: Any = None

    # Multi-scale pyramid for binned technologies (e.g. Visium HD).
    # Maps {mpp_float: adata_object}. Populated by the loader, not auto-generated.
    # Continuous point clouds (Xenium) use key 0.0 to signal "no binning applies".
    pyramid: Dict[float, Any] = field(default_factory=dict)

    # Set by apply_fine_alignment(). Holds the FineAlignment that produced spatial_fine.
    # None means no fine alignment has been applied to this channel.
    fine_alignment: Optional[Any] = None

    # Controls when get_active_coords() uses fine-aligned coordinates.
    # True     — use fine alignment when its displacement is supra-pixel (default)
    # False    — always return raw coords regardless of fine alignment presence
    # 'always' — force fine alignment on even for low-resolution renders / thumbnails
    use_fine_alignment: Any = True

    def __post_init__(self):
        if self.data is not None and not self.pyramid:
            is_continuous = self.metadata.get('is_continuous', False)
            if is_continuous:
                self.pyramid[0.0] = self.data
            else:
                base_mpp = float(self.spatial_meta.transform_matrix[0, 0])
                self.pyramid[base_mpp] = self.data

    def get_optimal_level(self, target_mpp: float) -> Tuple[float, Any]:
        """
        Returns the best AnnData for the requested render resolution.
        Continuous point clouds (Xenium) always return the base data.
        Binned data (Visium HD) rounds down to the largest bin <= target_mpp.
        """
        is_continuous = self.metadata.get('is_continuous', False)
        if is_continuous or not self.pyramid:
            return 0.0, self.data

        available = sorted(self.pyramid.keys())
        best_mpp = available[0]

        for mpp in available:
            if mpp <= target_mpp:
                best_mpp = mpp
            else:
                break

        return best_mpp, self.pyramid[best_mpp]

    def get_active_coords(self, mpp: float = 0.0) -> 'np.ndarray':
        """Return the best available coordinates in micron space.

        Behaviour is controlled by self.use_fine_alignment:
          True (default) — prefer fine-aligned coords when present; skip when the
                           displacement is sub-pixel at the given render resolution
          False          — always return raw baseline coords
          'always'       — force fine-aligned coords regardless of render resolution

        Falls back to the raw baseline ('spatial' / 'x'+'y') if no fine coords exist.
        """
        import numpy as np
        use = self.use_fine_alignment

        if hasattr(self.data, 'obsm'):
            # AnnData path (Visium / binned Xenium)
            if 'spatial_fine' in self.data.obsm and use is not False:
                if use == 'always':
                    return self.data.obsm['spatial_fine']
                # True: skip fine alignment only when correction is sub-pixel
                if (mpp > 0 and self.fine_alignment is not None
                        and self.fine_alignment.max_displacement_um < mpp * 0.5):
                    return self.data.obsm['spatial']
                return self.data.obsm['spatial_fine']
            return self.data.obsm['spatial']
        else:
            # DataFrame path (Xenium transcripts)
            if 'x_fine' in self.data.columns and use is not False:
                return self.data[['x_fine', 'y_fine']].values
            return self.data[['x', 'y']].values

    def _get_default_render_params(self, target_mpp: float) -> Tuple[Tuple[float, float, float, float], Tuple[int, int]]:
        """Helper to calculate full-channel bounds and pixel shape at a given resolution."""
        extent = self.spatial_meta.extent
        h_px = int((extent[3] - extent[2]) / target_mpp)
        w_px = int((extent[1] - extent[0]) / target_mpp)
        return extent, (h_px, w_px)

    def to_color_image(self, gene_name: str, target_mpp: float = 1.0, cmap_name: str = 'plasma', **kwargs) -> 'ImageChannel':
        """Easy-mode: Renders the entire omic channel to a color ImageChannel."""
        from ..render_utils import render_omic_color_mapped

        extent, shape = self._get_default_render_params(target_mpp)
        return render_omic_color_mapped(self, extent, shape, gene_name, cmap_name=cmap_name, **kwargs)

    def to_grayscale_image(self, gene_name: str, target_mpp: float = 1.0, **kwargs) -> 'ImageChannel':
        """Easy-mode: Renders the entire omic channel to a grayscale ImageChannel."""
        from ..render_utils import render_omic_grayscale

        extent, shape = self._get_default_render_params(target_mpp)
        return render_omic_grayscale(self, extent, shape, gene_name, **kwargs)

    def render_to_canvas(self, canvas: 'ImageChannel', gene_name: str, **kwargs) -> 'ImageChannel':
        """EASY MODE: Renders matching the exact bounds/resolution of a canvas."""
        from ..render_utils import render_omic_to_image

        return render_omic_to_image(self, canvas, gene_name, **kwargs)
