from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
from platpy.transform_utils import SpatialMetadata, pixels_to_microns as _px_to_um, microns_to_pixels as _um_to_px

@dataclass
class BaseChannel:
    """The fundamental building block of platpy."""
    name: str
    spatial_meta: SpatialMetadata = field(default_factory=SpatialMetadata)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set_style(self, **kwargs) -> 'BaseChannel':
        """
        Stores default rendering styles into metadata['layer_defaults'].
        These are automatically picked up when the channel is added to a ViewPort,
        so you can pre-configure display without touching the viewport at all.

        Example:
            roi = ROIChannel(...).set_style(line_color='cyan', line_width=3, fill=False)
            view.add_layer(roi)   # display defaults applied automatically
        """
        self.metadata.setdefault('layer_defaults', {}).update(kwargs)
        return self

    def pixels_to_microns(self, px_coords: np.ndarray) -> np.ndarray:
        """Convert (N, 2) pixel coordinates to absolute micron coordinates."""
        return _px_to_um(self.spatial_meta.transform_matrix, px_coords)

    def microns_to_pixels(self, um_coords: np.ndarray) -> np.ndarray:
        """Convert (N, 2) absolute micron coordinates to pixel coordinates."""
        return _um_to_px(self.spatial_meta.transform_matrix, um_coords)