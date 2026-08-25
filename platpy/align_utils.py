"""
Fine alignment utilities for platpy.

Corrects the small (~3-4 µm) residual misalignment between Visium spot positions
and the fullres morphology image that SpaceRanger's affine registration leaves behind.
Tissue physically stretches during transfer onto the capture array; SpaceRanger anchors
to the fiducial frame at the periphery, so interior drift accumulates smoothly.

Pipeline:
  1. compute_block_displacements() — phase cross-correlation on overlapping blocks
  2. fit_displacement_field()   — interpolate sparse block shifts to a smooth 2D field
  3. apply_field_to_coords()    — apply the field to spot coordinates in micron space

Top-level helpers:
  compute_fine_alignment(sample, dapi_channel) — renders both grids via ViewPort,
                                                  runs steps 1-2, stores result
  apply_fine_alignment(spots_or_adata, sample) — runs step 3 using stored field
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


# =============================================================================
# DATA STRUCTURE
# =============================================================================

@dataclass
class FineAlignment:
    """Stores a smooth 2D displacement field for sub-micron spot correction."""
    field_dy_um:          np.ndarray  # 2D float32, shape (n_rows, n_cols) — y shift in µm
    field_dx_um:          np.ndarray  # 2D float32, shape (n_rows, n_cols) — x shift in µm
    origin_um:            tuple       # (y0, x0) of the top-left grid corner in microns
    resolution_um:        float       # µm per grid cell
    method:               str   = 'pixel'  # 'pixel' = density-map alignment; future: 'tps'
    max_displacement_um:  float = 0.0      # 95th-percentile displacement magnitude across the field


# =============================================================================
# STEP 1: BLOCK-LEVEL PHASE CROSS-CORRELATION
# =============================================================================

def compute_block_displacements(
    img_ref,
    img_mov,
    block_size_px=32,
    stride_px=16,
    upsample_factor=20,
    max_shift_px=8
):
    """
    Estimates a sparse set of local displacements between two images.

    Slides a block window across both images and calls phase_cross_correlation on
    each block pair. Blank blocks (low signal variance) and implausibly large shifts
    are discarded.

    Parameters
    ----------
    img_ref        : 2D float array — reference image (DAPI), normalised to [0,1]
    img_mov        : 2D float array — moving image (density map), same shape, normalised
    block_size_px  : int — block side length in pixels
    stride_px      : int — step between block centres in pixels
    upsample_factor: int — sub-pixel precision factor (20 → 0.05 px accuracy)
    max_shift_px   : int — maximum allowed shift in either axis; blocks exceeding this
                     are discarded as likely artefacts or edge noise

    Returns
    -------
    list of (cy_px, cx_px, dy_px, dx_px)
        cy_px, cx_px : block centre in pixel coordinates of the grid
        dy_px, dx_px : shift (moving → reference), fractional pixels
    """
    from skimage.registration import phase_cross_correlation

    n_rows, n_cols = img_ref.shape
    half = block_size_px // 2
    blank_thresh = 0.02     # skip blocks with std < 2% of full range

    results = []

    for r in range(0, n_rows - block_size_px + 1, stride_px):
        for c in range(0, n_cols - block_size_px + 1, stride_px):
            ref_block = img_ref[r : r + block_size_px, c : c + block_size_px]
            mov_block = img_mov[r : r + block_size_px, c : c + block_size_px]

            if ref_block.std() < blank_thresh or mov_block.std() < blank_thresh:
                continue

            shift, _, _ = phase_cross_correlation(
                ref_block, mov_block, upsample_factor=upsample_factor
            )

            # First-pass hard cutoff — catches random correlations at tissue edges
            if abs(shift[0]) > max_shift_px or abs(shift[1]) > max_shift_px:
                continue

            cy = float(r + half)
            cx = float(c + half)
            results.append((cy, cx, float(shift[0]), float(shift[1])))

    return results


def filter_block_shifts(block_shifts, n_sigma=3.0):
    """
    Removes outlier blocks using MAD-based statistics on the shift distribution.

    Blocks whose shift deviates more than n_sigma * MAD from the median are
    discarded. This handles cases where there is a genuine systematic offset
    (e.g. 10 µm y-drift) without treating it as an outlier.

    Returns filtered list of (cy_px, cx_px, dy_px, dx_px).
    """
    if len(block_shifts) < 10:
        return block_shifts

    shifts = np.array(block_shifts)
    dy, dx = shifts[:, 2], shifts[:, 3]

    dy_med, dx_med = np.median(dy), np.median(dx)
    # MAD * 1.4826 converts to Gaussian-equivalent sigma
    dy_sigma = max(np.median(np.abs(dy - dy_med)) * 1.4826, 0.1)
    dx_sigma = max(np.median(np.abs(dx - dx_med)) * 1.4826, 0.1)

    keep = (
        (np.abs(dy - dy_med) <= n_sigma * dy_sigma) &
        (np.abs(dx - dx_med) <= n_sigma * dx_sigma)
    )
    filtered = [block_shifts[i] for i in range(len(block_shifts)) if keep[i]]
    n_removed = len(block_shifts) - len(filtered)
    if n_removed > 0:
        print(f"[fine align]   outlier filter: removed {n_removed} of {len(block_shifts)} blocks  "
              f"(dy median={dy_med:.2f}px, dx median={dx_med:.2f}px)")
    return filtered


# =============================================================================
# STEP 2: FIT SMOOTH DISPLACEMENT FIELD
# =============================================================================

def fit_displacement_field(block_shifts, grid_shape, resolution_um, smooth_sigma=2.0):
    """
    Fits a smooth 2D displacement field from sparse block measurements.

    Uses scipy griddata (linear interpolation) to fill the grid, nearest-neighbour
    to fill NaN edges, then a Gaussian smooth to suppress block-to-block noise.

    Parameters
    ----------
    block_shifts  : list of (cy_px, cx_px, dy_px, dx_px) from compute_block_displacements
    grid_shape    : (n_rows, n_cols)
    resolution_um : float — used to convert pixel shifts → micron shifts
    smooth_sigma  : float — Gaussian smoothing radius in grid pixels (default 2 = 20 µm
                    at 10 µm/px resolution). Set to 0 to skip smoothing.

    Returns
    -------
    field_dy_um : 2D float32 array, shape grid_shape
    field_dx_um : 2D float32 array, shape grid_shape
    """
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    n_rows, n_cols = grid_shape
    row_grid, col_grid = np.mgrid[0:n_rows, 0:n_cols]
    query_pts = np.c_[row_grid.ravel(), col_grid.ravel()]

    if len(block_shifts) < 4:
        print(f"[fine align] Warning: only {len(block_shifts)} valid blocks — field will be near-zero")
        return np.zeros((n_rows, n_cols), np.float32), np.zeros((n_rows, n_cols), np.float32)

    shifts = np.array(block_shifts)      # (M, 4)
    cy    = shifts[:, 0]
    cx    = shifts[:, 1]
    dy_um = shifts[:, 2] * resolution_um  # convert px → µm
    dx_um = shifts[:, 3] * resolution_um

    field_dy = griddata((cy, cx), dy_um, query_pts, method='linear').reshape(n_rows, n_cols)
    field_dx = griddata((cy, cx), dx_um, query_pts, method='linear').reshape(n_rows, n_cols)

    # Fill NaN edges (outside convex hull of block centres) with nearest-neighbour
    nan_mask = np.isnan(field_dy)
    if nan_mask.any():
        dy_nn = griddata((cy, cx), dy_um, query_pts, method='nearest').reshape(n_rows, n_cols)
        dx_nn = griddata((cy, cx), dx_um, query_pts, method='nearest').reshape(n_rows, n_cols)
        field_dy[nan_mask] = dy_nn[nan_mask]
        field_dx[nan_mask] = dx_nn[nan_mask]

    # Gaussian smooth to suppress block-to-block sampling noise
    if smooth_sigma > 0:
        field_dy = gaussian_filter(field_dy, sigma=smooth_sigma)
        field_dx = gaussian_filter(field_dx, sigma=smooth_sigma)

    return field_dy.astype(np.float32), field_dx.astype(np.float32)


def fit_displacement_field_tps(block_shifts, grid_shape, resolution_um):
    """
    Fits a Thin Plate Spline displacement field from sparse block measurements.

    Uses scipy RBFInterpolator with the thin_plate_spline kernel, which minimises
    the global bending energy of the warp. Compared to griddata + Gaussian smooth,
    TPS is naturally smooth everywhere, handles non-uniform block coverage better,
    and never produces NaN edges.

    Parameters
    ----------
    block_shifts  : list of (cy_px, cx_px, dy_px, dx_px) from compute_block_displacements
    grid_shape    : (n_rows, n_cols)
    resolution_um : float — used to convert pixel shifts → micron shifts

    Returns
    -------
    field_dy_um : 2D float32 array, shape grid_shape
    field_dx_um : 2D float32 array, shape grid_shape
    """
    from scipy.interpolate import RBFInterpolator

    n_rows, n_cols = grid_shape

    if len(block_shifts) < 4:
        print(f"[fine align] Warning: only {len(block_shifts)} valid blocks — field will be near-zero")
        return np.zeros((n_rows, n_cols), np.float32), np.zeros((n_rows, n_cols), np.float32)

    shifts   = np.array(block_shifts)           # (M, 4)
    ctrl_pts = shifts[:, :2]                    # (cy, cx) in grid-pixel space
    dy_um    = shifts[:, 2] * resolution_um     # px → µm
    dx_um    = shifts[:, 3] * resolution_um

    tps_dy = RBFInterpolator(ctrl_pts, dy_um, kernel='thin_plate_spline')
    tps_dx = RBFInterpolator(ctrl_pts, dx_um, kernel='thin_plate_spline')

    row_grid, col_grid = np.mgrid[0:n_rows, 0:n_cols]
    query_pts = np.c_[row_grid.ravel(), col_grid.ravel()]

    field_dy = tps_dy(query_pts).reshape(n_rows, n_cols).astype(np.float32)
    field_dx = tps_dx(query_pts).reshape(n_rows, n_cols).astype(np.float32)

    return field_dy, field_dx


# =============================================================================
# STEP 3: APPLY FIELD TO COORDINATES
# =============================================================================

def apply_field_to_coords(coords_um, field_dy_um, field_dx_um, origin_um, resolution_um):
    """
    Applies a displacement field to an array of spot coordinates.

    Parameters
    ----------
    coords_um     : (N, 2) float array — columns are (x, y) in microns
    field_dy_um   : 2D float32 from fit_displacement_field
    field_dx_um   : 2D float32 from fit_displacement_field
    origin_um     : (y0, x0) of grid top-left corner in microns
    resolution_um : float — µm per grid cell

    Returns
    -------
    corrected_coords_um : (N, 2) float array, same column order as input
    """
    from scipy.interpolate import RegularGridInterpolator

    y0, x0 = origin_um
    n_rows, n_cols = field_dy_um.shape

    # Axis coordinates: centre of each grid cell in microns
    row_axis = y0 + (np.arange(n_rows) + 0.5) * resolution_um
    col_axis = x0 + (np.arange(n_cols) + 0.5) * resolution_um

    interp_dy = RegularGridInterpolator(
        (row_axis, col_axis), field_dy_um,
        method='linear', bounds_error=False, fill_value=0.0
    )
    interp_dx = RegularGridInterpolator(
        (row_axis, col_axis), field_dx_um,
        method='linear', bounds_error=False, fill_value=0.0
    )

    x = coords_um[:, 0]
    y = coords_um[:, 1]

    # RegularGridInterpolator expects (axis0=row=y, axis1=col=x)
    query = np.c_[y, x]
    dy = interp_dy(query)
    dx = interp_dx(query)

    corrected = coords_um.copy().astype(np.float64)
    corrected[:, 0] = x + dx
    corrected[:, 1] = y + dy
    return corrected


# =============================================================================
# TOP-LEVEL PIPELINE HELPERS
# =============================================================================

def compute_fine_alignment(
    sample,
    dapi_channel,
    resolution_um=4.0,
    block_size_um=320.0,
    stride_um=40.0,
    upsample_factor=20,
    method='pixel' 
):
    """
    Full pipeline: bins spot counts, matches blocks to DAPI, fits displacement field.
    Stores the FineAlignment in sample.metadata['fine_spatial'] and returns it.

    Parameters
    ----------
    sample        : SampleSet with at least one omic channel ('spots' or 'transcripts')
    dapi_channel  : ImageChannel — coarsely aligned DAPI image
    resolution_um : grid cell size in µm (10 µm default works well for Visium)
    block_size_um : block side length for cross-correlation
    stride_um     : stride between block centres (50% overlap by default)
    upsample_factor: sub-pixel precision passed to phase_cross_correlation
    method        : 'pixel' (griddata + Gaussian smooth, default) or
                    'tps' (Thin Plate Spline — smoother, handles non-uniform coverage better)
    """
    omic_channel = sample.omics.get('spots') or sample.omics.get('transcripts')
    if omic_channel is None:
        raise ValueError("No omic channel found in sample.omics — expected key 'spots' or 'transcripts'.")

    # Build both grids through the ViewPort rendering pipeline so they share
    # exactly the same pixel grid, scale, and spatial origin — no manual
    # histogram edges or independent crop/resize that could introduce offsets.
    from .viewport import ViewPort

    ext = dapi_channel.spatial_meta.extent   # (min_x, max_x, min_y, max_y)
    cx  = (ext[0] + ext[1]) / 2
    cy  = (ext[2] + ext[3]) / 2
    fx  = ext[1] - ext[0]
    fy  = ext[3] - ext[2]
    target_w = max(64, int(fx / resolution_um))

    vp = ViewPort.create(
        target_width_px=target_w,
        center_um=(cx, cy),
        fov_um=(fx, fy)
    )

    print(f"[fine align] Step 1: rendering spot density via ViewPort ({target_w} px at {resolution_um} µm/px) ...")
    vp_dens = vp.clone(keep_layers=False)
    vp_dens.add_layer(omic_channel).adjust_layer(
        gene_name='total_counts', cmap_name='gray', alpha=1.0)
    density_grid = np.asarray(vp_dens.render().image, dtype=np.float32)
    if density_grid.ndim == 3:
        density_grid = density_grid[..., :3].mean(axis=2)

    print(f"[fine align] Step 2: rendering DAPI via same ViewPort ...")
    vp_dapi = vp.clone(keep_layers=False)
    vp_dapi.add_layer(dapi_channel).adjust_layer(percentile_clip=(2, 99))
    dapi_grid = np.asarray(vp_dapi.render().image, dtype=np.float32)
    if dapi_grid.ndim == 3:
        dapi_grid = dapi_grid[..., :3].mean(axis=2)

    # Both grids are now uint8 [0,255]; normalise to [0,1]
    density_norm = density_grid / max(float(density_grid.max()), 1.0)
    dapi_grid    = dapi_grid    / max(float(dapi_grid.max()),    1.0)

    # Grid origin for apply_field_to_coords: top-left corner in microns
    ve       = vp.view_extent   # (min_x, max_x, min_y, max_y)
    origin_um = (ve[2], ve[0])  # (min_y, min_x)
    print(f"[fine align]   grid shape: {density_grid.shape}  origin: {origin_um}")

    block_size_px = max(8,  int(block_size_um / resolution_um))
    stride_px     = max(4,  int(stride_um    / resolution_um))

    print(f"[fine align] Step 3: block matching  (block={block_size_px}px, stride={stride_px}px) ...")
    block_shifts = compute_block_displacements(
        dapi_grid, density_norm, block_size_px, stride_px, upsample_factor
    )
    print(f"[fine align]   {len(block_shifts)} valid blocks before outlier filter")
    block_shifts = filter_block_shifts(block_shifts)
    print(f"[fine align]   {len(block_shifts)} blocks after outlier filter")

    print(f"[fine align] Step 4: fitting displacement field (method='{method}') ...")
    if method == 'tps':
        field_dy_um, field_dx_um = fit_displacement_field_tps(block_shifts, density_grid.shape, resolution_um)
    else:
        field_dy_um, field_dx_um = fit_displacement_field(block_shifts, density_grid.shape, resolution_um)

    magnitudes = np.sqrt(field_dy_um**2 + field_dx_um**2)
    max_disp_um = float(np.percentile(magnitudes, 95))

    fa = FineAlignment(
        field_dy_um=field_dy_um,
        field_dx_um=field_dx_um,
        origin_um=origin_um,
        resolution_um=resolution_um,
        method=method,
        max_displacement_um=max_disp_um
    )

    sample.metadata['fine_spatial'] = fa
    print(f"[fine align] Done. max_displacement_um={max_disp_um:.2f} µm. Stored in sample.metadata['fine_spatial'].")
    return fa


def apply_fine_alignment(omic_channel, sample, in_place=False):
    """
    Applies the stored fine alignment field to an OmicChannel.

    Parameters
    ----------
    omic_channel : OmicChannel
    sample       : SampleSet — must have fine_spatial in metadata
    in_place     : bool
        False (default) — returns a new OmicChannel with all pyramid levels copied.
                          Original channel is not modified.  Name is suffixed '_fine'.
        True            — writes obsm['spatial_fine'] into the existing AnnData levels
                          with no copies.  Returns the same channel object.  Use when
                          you don't need the unaligned channel for comparison and want
                          to avoid duplicating the expression matrix in memory.

    In both cases obsm['spatial'] is left untouched as the raw baseline, and
    get_active_coords() will automatically prefer obsm['spatial_fine'] going forward.
    """
    import copy
    fa = sample.metadata.get('fine_spatial')
    if fa is None:
        raise ValueError("No fine alignment found. Run compute_fine_alignment() first.")

    def _add_fine_coords(adata):
        coords = adata.obsm['spatial'].copy().astype(np.float64)
        adata.obsm['spatial_fine'] = apply_field_to_coords(
            coords, fa.field_dy_um, fa.field_dx_um, fa.origin_um, fa.resolution_um
        )

    if in_place:
        _add_fine_coords(omic_channel.data)
        for mpp_key, level_adata in omic_channel.pyramid.items():
            if level_adata is not omic_channel.data:
                _add_fine_coords(level_adata)
        omic_channel.fine_alignment = fa
        return omic_channel

    # Copy path: new channel, original unchanged
    def _align_adata(adata):
        adata_new = adata.copy()
        _add_fine_coords(adata_new)
        return adata_new

    new_channel = copy.copy(omic_channel)
    new_channel.data = _align_adata(omic_channel.data)

    new_pyramid = {}
    for mpp_key, level_adata in omic_channel.pyramid.items():
        if level_adata is omic_channel.data:
            new_pyramid[mpp_key] = new_channel.data   # reuse already-corrected copy
        else:
            new_pyramid[mpp_key] = _align_adata(level_adata)
    new_channel.pyramid = new_pyramid

    new_channel.fine_alignment = fa
    new_channel.name = omic_channel.name + '_fine'
    return new_channel


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def plot_displacement_field(fa, step=15, scale=250, ax=None, figsize=(7, 9), flip_x=False, flip_y=False):
    """
    Quiver plot of a FineAlignment displacement field — standard sanity check
    after compute_fine_alignment().

    Coherent arrows (all pointing the same general direction) confirm that
    block-matching found real tissue drift rather than noise. Random arrows
    mean the alignment should not be trusted.

    Parameters
    ----------
    fa    : FineAlignment object from compute_fine_alignment()
    step  : int   — subsample every N grid cells for readability (default 15)
    scale : float — matplotlib quiver scale; smaller = longer arrows
    ax    : matplotlib Axes or None — if None, creates and shows its own figure
    """
    import matplotlib.pyplot as plt

    nr, nc = fa.field_dy_um.shape
    y0, x0 = fa.origin_um
    res    = fa.resolution_um

    cg, rg = np.meshgrid(np.arange(0, nc, step), np.arange(0, nr, step))
    x_um   = x0 + cg * res
    y_um   = y0 + rg * res
    dy_s   = fa.field_dy_um[::step, ::step]
    dx_s   = fa.field_dx_um[::step, ::step]
    mag    = np.sqrt(dy_s**2 + dx_s**2)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)

    q = ax.quiver(x_um, y_um, dx_s, dy_s, mag, cmap='viridis', scale=scale, width=0.004)
    plt.colorbar(q, ax=ax, label='Displacement magnitude (µm)', shrink=0.7)
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_title('Fine alignment displacement field')
    if flip_y: ax.invert_yaxis()
    if flip_x: ax.invert_xaxis()

    if own_fig:
        plt.tight_layout()
        plt.show()
