# platpy
### Pipeline for Layered Analysis of Transcriptomics

platpy is a Python framework for spatial transcriptomics analysis. It allows you to work with spatial datasets in a unified space (microns), and it keeps imaging, spot-level, and transcript-level data in perfect registration regardless of the spatial platform, imaging resolution, or analysis operation applied.

---

## Key Features

- **Unified micron space** — every channel (images, spots, transcripts, ROIs) shares one absolute coordinate system. Crop, rotate, zoom, or overlay any combination of data layers without managing scale factors or alignment matrices.
- **ViewPort** — compositing engine for multi-channel spatial renders. Handles rotation, flip, zoom, and multi-layer blending. Automatically selects the appropriate image pyramid level for the render resolution. Renders are fast and accurate.
- **DataPort** — memory-efficient ROI extraction pipeline. Crop image regions, run segmentation (CellPose, StarDist), and convert results back to micron space automatically.
- **Plays well with spatial ecosystem** — designed to work seamlessly with the widely used pipelines for spatial and single cell analysis: scanpy, squidpy, etc.
- **Spatial analysis tools** — simple tools provided for common spatial tasks: cortical depth binning, contour shells, axis bins, and cortex unfolding for structure-aware expression analysis.
- **Multi-platform** — same API for Visium HD and Xenium; designed to be extensible to additional spatial platforms.
- **Memory-efficient** — designed to be quick and efficient: use of python toolkits like Dask/Zarr for lazy loading, data indexing, and image pyramids keeps the footprint low enough to run on a laptop.

---

## Installation

```bash
pip install git+https://github.com/maynardt/platpy.git
```

### Dependencies

Core dependencies are installed automatically. For segmentation vignettes, install CellPose and/or StarDist separately:

```bash
# --no-deps avoids cellpose pulling in opencv-python-headless, which
# collides with the opencv-python this project already depends on
pip install cellpose --no-deps
pip install torch torchvision fastremap imagecodecs roifile fill-voids segment_anything
pip install stardist
```

For interactive ROI drawing:

```bash
pip install napari
```

---

## Quick Start

```python
from platpy import load_sample, ViewPort

# Load a pre-processed platpy bundle
sample = load_sample('path/to/VisiumSample')

# Create a viewport and render a multi-channel view
vp = sample.viewport(center_um=(2000, 3000), fov_um=(1000, 800))
vp.add_layer(sample.images['dapi']).adjust_layer(tint='xkcd:bright blue', percentile_clip=(2, 99))
vp.add_layer(sample.omics['spots']).adjust_layer(gene_name='total_counts', cmap_name='inferno')

from platpy.plot_utils import plot_image_grid
plot_image_grid(vp.render(), width=6)
```

---

## Vignettes

Step-by-step vignettes are in the [`vignettes/`](vignettes/) directory:

| Vignette | Description |
|----------|-------------|
| [00 — Data loading](vignettes/Vignette_00_data_loading.ipynb) | Convert raw 10x outputs to platpy format (not needed to run pre-generated sample datasets) |
| [01 — Spatial registration](vignettes/Vignette_01_spatial_registration.ipynb) | The registration problem and how platpy solves it |
| [02 — Virtual views](vignettes/Vignette_02_virtual_views.ipynb) | ViewPort: compositing, orientation, Napari ROI drawing |
| [03 — DataPort](vignettes/Vignette_03_dataport.ipynb) | ROI extraction, CellPose/StarDist segmentation |
| [04 — Multi-platform](vignettes/Vignette_04_multi_platform_workflow.ipynb) | Xenium on the same API as Visium |
| [05 — Spatial to single-cell](vignettes/Vignette_05_.spatial_to_single_cell.ipynb) | Cell assignment, expression matrices, UMAP |
| [06 — Spatial analysis tools](vignettes/Vignette_06_.spatial_analysis_tools.ipynb) | Depth bins, contour shells, cortex unfolding |
| [07 — Fine alignment](vignettes/Vignette_07_fine_alignment.ipynb) | Sub-micron IF-to-transcript registration |

Sample datasets for all vignettes are downloaded automatically on first run from Zenodo (https://doi.org/10.5281/zenodo.20320692).

---

## Citation

If you use platpy in your research, please cite:

> Maynard, T. (2026). platpy: Pipeline for Layered Analysis of Transcriptomics. *bioRxiv*. [link to follow]

---

## License

MIT — see [LICENSE](LICENSE).
