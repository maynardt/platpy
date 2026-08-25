"""
draw_rois.py  --  Interactive ROI drawing tool for SPACKLE

Run directly from the terminal (not in Jupyter):
    python demo/draw_rois.py

Workflow:
    1. Napari opens with a downsampled DAPI image.
    2. Draw a polygon using the Shapes tool, type a name, click "Add ROI".
    3. Optionally draw an axis line (pial → WM direction) and click "Add Axis"
       using the same name as its ROI.  This direction is used by
       extract_ordered_edges() for cortical depth binning.
    4. Repeat for additional ROIs / axes.
    5. Click "Save & Close" -- writes the JSON file.

Output JSON format:
    {
      "cortex": [[x, y], ...],          <- polygon vertices in microns
      "cortex_axis": [dx, dy],          <- unit vector pial → WM (if drawn)
      "striatum": [[x, y], ...],
      ...
    }

Load into spatial analysis:
    rois = load_rois("demo/rois.json")
    cortex_poly  = np.array(rois["cortex"])
    direction    = np.array(rois["cortex_axis"])   # if axis was drawn
"""
import sys
from pathlib import Path
import numpy as np
import napari
from magicgui.widgets import Container, LineEdit, PushButton, Label
from qtpy.QtCore import QEventLoop

from spackle.io_utils import load_sample
from spackle.transform_utils import pixels_to_microns
from spackle.spatial_tools import save_rois

# =============================================================================
# CONFIG -- edit these for your dataset
# =============================================================================
SAMPLE_PATH   = Path.home() / 'spackle_data' / 'XeniumSamples' / 'WT_02'
IMAGE_KEY     = "DAPI"                        # image channel to display
OUTPUT_JSON   = Path(__file__).parent / "WT_02_rois.json"
DISPLAY_WIDTH = 2000                               # px -- lighter = faster
# =============================================================================


def main():
    print(f"Loading sample from {SAMPLE_PATH} ...")
    sample = load_sample(str(SAMPLE_PATH))

    img_ch = sample.images[IMAGE_KEY]
    min_x, max_x, min_y, max_y = img_ch.extent

    # Pick the pyramid level closest to DISPLAY_WIDTH
    full_width_px = img_ch.image.shape[1]
    target_mpp = (max_x - min_x) / DISPLAY_WIDTH
    display_ch = img_ch.get_optimal_channel(target_mpp)

    # Compute and display the image -- force out of dask if needed
    img = np.array(display_ch.image)
    if img.max() > 0:
        img = (img / img.max() * 255).astype(np.uint8)

    transform_matrix = display_ch.spatial_meta.transform_matrix
    roi_dict  = {}   # name → polygon (µm)
    axis_dict = {}   # name → unit direction vector
    cap_dict  = {}   # name → [[pial_x, pial_y], [wm_x, wm_y]] (µm)
    combined  = {}   # populated on save; returned from main()

    # -------------------------------------------------------------------------
    viewer = napari.Viewer(title="SPACKLE -- Draw ROIs")
    viewer.add_image(img, name=IMAGE_KEY, colormap="gray")

    shapes_layer = viewer.add_shapes(
        name="ROI Polygons",
        edge_color="cyan",
        face_color="transparent",
        edge_width=3,
    )
    axes_layer = viewer.add_shapes(
        name="Axis Lines",
        edge_color="yellow",
        face_color="transparent",
        edge_width=2,
    )

    # -- panel widgets --------------------------------------------------------
    name_field    = LineEdit(value="", label="Name")
    add_roi_btn   = PushButton(text="Add ROI (polygon)")
    add_axis_btn  = PushButton(text="Add Axis (line, pial→WM direction vector)")
    add_cap_btn   = PushButton(text="Add Cap (line, pial→WM endpoints)")
    status_label  = Label(value="No ROIs saved yet.")
    save_btn      = PushButton(text="Save & Close")

    panel = Container(widgets=[
        name_field, add_roi_btn, add_axis_btn, add_cap_btn, status_label, save_btn
    ])

    def _current_name():
        name = name_field.value.strip()
        if not name:
            status_label.value = "Enter a name first."
        return name

    def _update_status():
        parts = list(roi_dict.keys())
        axes  = list(axis_dict.keys())
        caps  = list(cap_dict.keys())
        msg   = f"{len(parts)} ROI(s): {', '.join(parts)}"
        if axes:
            msg += f"  |  axes: {', '.join(axes)}"
        if caps:
            msg += f"  |  caps: {', '.join(caps)}"
        status_label.value = msg

    @add_roi_btn.clicked.connect
    def on_add_roi():
        name = _current_name()
        if not name:
            return
        if len(shapes_layer.data) == 0:
            status_label.value = "Draw a polygon on the 'ROI Polygons' layer first."
            return

        # Napari gives (row, col) = (y, x) -- swap to (x, y)
        raw_yx  = shapes_layer.data[-1]
        poly_px = raw_yx[:, [1, 0]]
        poly_um = pixels_to_microns(transform_matrix, poly_px)

        roi_dict[name] = poly_um
        name_field.value = ""
        print(f"  Added ROI '{name}' ({len(poly_px)} vertices)")
        _update_status()

    @add_axis_btn.clicked.connect
    def on_add_axis():
        name = _current_name()
        if not name:
            return
        # Find the most recently drawn line on the Axes layer
        lines = [s for s in axes_layer.data if len(s) == 2]
        if not lines:
            status_label.value = "Draw a line on the 'Axis Lines' layer first."
            return

        raw_yx = lines[-1]              # shape (2, 2), row/col
        p0_px  = raw_yx[0, [1, 0]]     # swap to x, y
        p1_px  = raw_yx[1, [1, 0]]
        p0_um  = pixels_to_microns(transform_matrix, p0_px[None])[0]
        p1_um  = pixels_to_microns(transform_matrix, p1_px[None])[0]

        direction = p1_um - p0_um
        norm = np.linalg.norm(direction)
        if norm == 0:
            status_label.value = "Axis line has zero length -- redraw it."
            return

        axis_dict[name] = (direction / norm).tolist()
        name_field.value = ""
        print(f"  Added axis '{name}': direction = {axis_dict[name]}")
        _update_status()

    @add_cap_btn.clicked.connect
    def on_add_cap():
        name = _current_name()
        if not name:
            return
        lines = [s for s in axes_layer.data if len(s) == 2]
        if not lines:
            status_label.value = "Draw a line on the 'Axis Lines' layer first."
            return

        raw_yx = lines[-1]
        p0_px  = raw_yx[0, [1, 0]]
        p1_px  = raw_yx[1, [1, 0]]
        p0_um  = pixels_to_microns(transform_matrix, p0_px[None])[0]
        p1_um  = pixels_to_microns(transform_matrix, p1_px[None])[0]

        cap_dict[name] = [p0_um.tolist(), p1_um.tolist()]
        name_field.value = ""
        print(f"  Added cap '{name}': {cap_dict[name]}")
        _update_status()

    @save_btn.clicked.connect
    def on_save():
        if not roi_dict:
            status_label.value = "No ROIs to save."
            return

        # Merge polygons, axis vectors, and cap lines into one flat dict
        combined.clear()
        for name, poly in roi_dict.items():
            combined[name] = poly
            if name in axis_dict:
                combined[f"{name}_axis"] = axis_dict[name]
        for name, pts in cap_dict.items():
            combined[name] = pts   # saved as [[x0,y0],[x1,y1]]

        save_rois(OUTPUT_JSON, combined)
        print(f"Saved {len(roi_dict)} ROI(s) + {len(axis_dict)} axis(es) + {len(cap_dict)} cap(s) to {OUTPUT_JSON}")
        viewer.close()

    viewer.window.add_dock_widget(panel, area="right")

    loop = QEventLoop()
    viewer.window._qt_window.destroyed.connect(loop.quit)
    loop.exec_()

    return combined if roi_dict else {}


if __name__ == "__main__":
    main()
