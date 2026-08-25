from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional

from .image_channel import ImageChannel
from .omic_channel import OmicChannel
from .cell_channel import CellChannel
from .roi_channel import ROIChannel

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
        new_name: Optional[str] = None
    ) -> 'SampleSet':
        """
        Extract a spatial sub-region as a new SampleSet.

        Images are cropped to the bounding box — spatial metadata is updated so
        the result still knows its absolute position in the master slide's
        coordinate frame. Transcript/omic DataFrames are filtered by x/y.
        AnnData omics are filtered by obsm['spatial'].

        The returned SampleSet can be saved and reloaded independently with
        save_sample() / load_sample().

        Args:
            min_x_um, max_x_um, min_y_um, max_y_um: ROI bounds in microns.
            new_name: Name for the extracted SampleSet. Defaults to self.name + '_roi'.
        """
        out = SampleSet(name=new_name or f"{self.name}_roi")
        out.screen_width  = self.screen_width
        out.view_defaults = dict(self.view_defaults)

        for ch_name, chan in self.images.items():
            try:
                out.images[ch_name] = chan.crop_to_micron_roi(
                    min_x_um=min_x_um, max_x_um=max_x_um,
                    min_y_um=min_y_um, max_y_um=max_y_um
                )
            except ValueError:
                pass  # channel doesn't overlap this ROI — skip silently

        for om_name, omic in self.omics.items():
            platform = omic.metadata.get('platform', 'visium')
            if platform == 'xenium':
                coords = omic.data.obsm['spatial']
                mask = (
                    (coords[:, 0] >= min_x_um) & (coords[:, 0] <= max_x_um) &
                    (coords[:, 1] >= min_y_um) & (coords[:, 1] <= max_y_um)
                )
                extracted = omic.data[mask].copy()
                # Block index row ranges refer to the master slide — rebuild for this ROI.
                if 'block_index' in extracted.uns:
                    import json
                    from ..io_utils import build_spatial_block_index
                    bi = json.loads(extracted.uns['block_index'])
                    extracted = build_spatial_block_index(
                        extracted, mpp=bi['mpp'], chunk_px=bi['chunk_px'], gene_col='feature'
                    )
                out.omics[om_name] = OmicChannel(
                    name=om_name,
                    data=extracted,
                    metadata=omic.metadata.copy()
                )
            else:
                coords = omic.data.obsm['spatial']
                mask = (
                    (coords[:, 0] >= min_x_um) & (coords[:, 0] <= max_x_um) &
                    (coords[:, 1] >= min_y_um) & (coords[:, 1] <= max_y_um)
                )
                out.omics[om_name] = OmicChannel(
                    name=om_name,
                    data=omic.data[mask].copy(),
                    spatial_meta=omic.spatial_meta,
                    metadata=omic.metadata.copy(),
                    pyramid={k: v[mask] for k, v in omic.pyramid.items()}
                )

        out.cells = dict(self.cells)
        out.rois  = dict(self.rois)
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
