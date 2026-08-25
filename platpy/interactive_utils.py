import napari
import numpy as np
from magicgui.widgets import Container, FloatSlider, PushButton
from qtpy.QtCore import QEventLoop

def macro_select_rois(master_sample, image_key='DAPI', downsample_width=2000):
    """
    STEP 1: Macro-level isolation using a lightweight downsampled viewport.
    Returns a list of raw polygon coordinates in absolute microns.
    """
    # 1. Create a fast, lightweight "Viewport" image
    macro_img = master_sample.images[image_key].resample_to_bounds(target_width_px=downsample_width)
    
    viewer = napari.Viewer()
    viewer.add_image(macro_img.image, name="Macro View", colormap="gray")
    
    roi_layer = viewer.add_shapes(
        name="Draw Tissues", shape_type='polygon',
        edge_color='magenta', face_color='transparent', edge_width=5
    )

    extracted_microns = []
    panel = Container(widgets=[PushButton(text="Harvest Macros & Close", name="btn")])

    @panel.btn.clicked.connect
    def on_harvest():
        for raw_roi_yx in roi_layer.data:
            # Napari gives (Y, X) in pixels. Swap to (X, Y)
            poly_px = raw_roi_yx[:, [1, 0]]
            
            # 2. Convert UI pixels directly to Absolute Master Microns!
            poly_um = poly_px * macro_img.mpp + [macro_img.extent[0], macro_img.extent[2]]
            extracted_microns.append(poly_um)
            
        viewer.close()

    viewer.window.add_dock_widget(panel, area="right")
    
    loop = QEventLoop()
    viewer.window._qt_window.destroyed.connect(loop.quit)
    loop.exec_()
    
    return extracted_microns


def micro_finetune_roi(master_sample, macro_poly_microns, new_name, image_key='DAPI', margin_um=200.0):
    """
    STEP 2: Micro-level fine-tuning.
    Takes a macro polygon, creates a high-res viewport, allows rotation & tight cropping.
    Returns the final extracted SampleSet.
    """
    # 1. Get the bounding box of the macro polygon
    min_x, min_y = macro_poly_microns.min(axis=0)
    max_x, max_y = macro_poly_microns.max(axis=0)
    
    # 2. Crop a high-res "Viewport" directly from the master slide
    micro_img = master_sample.images[image_key].crop_to_micron_roi(
        min_x - margin_um, max_x + margin_um, 
        min_y - margin_um, max_y + margin_um
    )
    
    viewer = napari.Viewer()
    dapi_layer = viewer.add_image(micro_img.image, name="Micro View", colormap="gray")
    
    roi_layer = viewer.add_shapes(
        name="Draw Tight Crop", shape_type='polygon',
        edge_color='cyan', face_color='transparent', edge_width=3
    )

    # Center of the viewport image in pixels
    h, w = micro_img.image.shape[:2]
    cy_px, cx_px = h / 2.0, w / 2.0

    angle_slider = FloatSlider(min=-180, max=180, value=0.0, label="Rotation")
    apply_btn = PushButton(text="Apply & Extract")
    panel = Container(widgets=[angle_slider, apply_btn])
    
    result_container = []

    @angle_slider.changed.connect
    def on_rotate(angle: float):
        dapi_layer.rotate = angle                     # type: ignore
        dapi_layer.rotation_center = (cy_px, cx_px)   # type: ignore

    @apply_btn.clicked.connect
    def on_apply():
        angle = angle_slider.value

        # roi_layer.data[0] is (N,2) in Napari's (row, col) = (Y, X) order
        # Swap to (X, Y) to match platpy convention
        roi_yx = roi_layer.data[0]
        roi_px = roi_yx[:, [1, 0]]

        # Convert viewport pixels to absolute microns using the viewport ImageChannel
        # micro_img.mpp gives microns/pixel, extent[0]/[2] give the physical origin (tx, ty)
        roi_um = roi_px * micro_img.mpp + [micro_img.extent[0], micro_img.extent[2]]

        # Crop the master sample (not the viewport) to this ROI
        final_sample = master_sample.extract_roi(roi_um, new_name=new_name, rotation_deg=angle)

        result_container.append(final_sample)
        viewer.close()

    viewer.window.add_dock_widget(panel, area="right")
    
    loop = QEventLoop()
    viewer.window._qt_window.destroyed.connect(loop.quit)
    loop.exec_()
    
    return result_container[0] if result_container else None



import napari
import tifffile
import dask.array as da
import numpy as np
from pathlib import Path

class NapariSlicer:
    def __init__(self, tiff_path: str | Path, pyramid_level: int = 4, base_mpp: float = 0.2125):
        """Launches Napari asynchronously for Jupyter workflows."""
        self.base_mpp = base_mpp

        tiff_path = Path(tiff_path)
        if not tiff_path.exists():
            alt_path = tiff_path.parent / "morphology_focus.ome.tif"
            if alt_path.exists(): tiff_path = alt_path
            else: raise FileNotFoundError(f"Image not found at {tiff_path}")

        # Keep the TiffFile open for the lifetime of this object —
        # the Zarr store holds a reference to the file handle and will
        # crash with AssertionError if the file is closed under it
        self._tif = tifffile.TiffFile(tiff_path)
        try:
            zarr_store = self._tif.series[0].levels[pyramid_level].aszarr()
            img_dask = da.from_zarr(zarr_store)
            full_h = self._tif.series[0].levels[0].shape[-2]
            curr_h = img_dask.shape[-2]  # type: ignore
            scale_factor = full_h / curr_h
        except (IndexError, AttributeError):
            zarr_store = self._tif.series[0].aszarr()
            img_dask = da.from_zarr(zarr_store)
            scale_factor = 1.0

        print(f"🚀 Launching Async Napari Slicer...")
        self.viewer = napari.Viewer()
        self.viewer.add_image(
            img_dask,
            name='Xenium_Preview',
            scale=(scale_factor, scale_factor),
            colormap='gray',
            blending='opaque'
        )

        self.roi_layer = self.viewer.add_shapes(
            name='ROI_Selector',
            face_color='#00000000',
            edge_color_cycle=['cyan', 'magenta', 'yellow', 'lime', 'orange', 'red'],
            edge_width=int(max(25, 250 / scale_factor))
        )

    def harvest_rois(self) -> list[np.ndarray]:
        """Harvests shapes drawn in Napari and converts them to absolute microns."""

        if len(self.roi_layer.data) == 0:
            print("⚠️ No ROIs drawn yet! Draw shapes in Napari first.")
            return []

        micron_polygons = []
        for rect in self.roi_layer.data:
            poly_um = np.array([
                [rect[0, 1] * self.base_mpp, rect[0, 0] * self.base_mpp],
                [rect[1, 1] * self.base_mpp, rect[1, 0] * self.base_mpp],
                [rect[2, 1] * self.base_mpp, rect[2, 0] * self.base_mpp],
                [rect[3, 1] * self.base_mpp, rect[3, 0] * self.base_mpp]
            ])
            micron_polygons.append(poly_um)

        print(f"✅ Harvested {len(micron_polygons)} ROIs.")
        self.viewer.close()
        return micron_polygons

    def __del__(self):
        try:
            self._tif.close()
        except Exception:
            pass