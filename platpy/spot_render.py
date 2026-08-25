import numpy as np
from typing import Optional, Tuple

# platpy imports
from .channels.image_channel import ImageChannel
from .channels.omic_channel import OmicChannel
from .transform_utils import SpatialMetadata, create_transform_matrix
from .render_utils import make_colormap_lut, _numba_render_spots_fast


def _estimate_grid_angle(coords_um: np.ndarray, sample_size: int = 1000) -> float:
    """
    Estimates the rotation angle of a square Visium HD spot grid in radians.

    Challenge: the spot grid is not axis-aligned, so rendering square spots
    axis-aligned produces nasty moire artifacts. We detect the grid angle,
    render into a native-aligned canvas, then let the affine warp handle
    rotating it back to true physical space.

    Uses nearest-neighbor angles + histogram mode to robustly detect the
    dominant grid direction even with diagonal neighbors present.
    """
    if len(coords_um) < 2:
        return 0.0
    
    # Random subset for speed
    n_samples = min(sample_size, len(coords_um))
    idx = np.random.choice(len(coords_um), n_samples, replace=False)
    subset = coords_um[idx]
    
    # Pairwise distances to find nearest neighbors
    diffs = subset[:, np.newaxis, :] - subset[np.newaxis, :, :]
    dists = np.sum(diffs**2, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    nearest_idx = np.argmin(dists, axis=1)
    nearest_diffs = subset[np.arange(n_samples), :] - subset[nearest_idx, :]
    
    angles = np.arctan2(nearest_diffs[:, 1], nearest_diffs[:, 0])
    
    # Modulo pi/2: squares are visually identical every 90 degrees
    angles_mod = angles % (np.pi / 2)
    
    # Histogram mode is robust against diagonal neighbor contamination
    hist, bin_edges = np.histogram(angles_mod, bins=90, range=(0, np.pi / 2))
    peak_idx = np.argmax(hist)
    return (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2.0


def render_spotset_to_image(
    omic: OmicChannel,
    base_image: ImageChannel,
    values: Optional[np.ndarray] = None,
    cmap_name: str = "plasma",
    alpha: float = 1.0,
    alpha_gamma: Optional[float] = None,
    alpha_pass: float = 1.0,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    percentile_clip: tuple = (1, 99),
    gamma: float = 1.0,
    spot_shape: str = "circle",
    return_native: bool = False,
    debug: bool = False
) -> ImageChannel:
    """
    Renders a Visium spotset (HD square grid or standard circles) onto a canvas.

    The key trick for HD square grids:
      1. Detect the grid rotation angle via nearest-neighbor histogram
      2. Render spots into a native axis-aligned canvas at that angle
      3. Attach the correct affine matrix (rotation + translation) to that canvas
      4. Let match_image_to_reference() warp it perfectly onto the target image

    This produces clean square spots with no moire and correct spatial registration.
    """
    from .render_utils import match_image_to_reference

    # --- STEP 1: Normalize values ---
    all_coords_um = omic.get_active_coords()   # prefers spatial_fine if present

    if values is None:
        values = omic.data.obs['total_counts'].values

    valid_mask = ~np.isnan(values)
    norm_vals = np.zeros_like(values, dtype=np.float32)

    if np.any(valid_mask):
        valid_values = values[valid_mask]

        if scale_min is not None or scale_max is not None:
            v_min = float(scale_min) if scale_min is not None else np.nanmin(valid_values)
            v_max = float(scale_max) if scale_max is not None else np.nanmax(valid_values)
        else:
            # v_min is always 0: expression data has no meaningful lower noise floor,
            # and deriving it from expressing spots discards real signal (e.g. count=1
            # normalizes to 0 when v_min=1). v_max uses the nonzero percentile so
            # sparse genes aren't crushed to 0 by the many zero-expression bins.
            nonzero = valid_values[valid_values > 0]
            clip_base = nonzero if len(nonzero) > 0 else valid_values
            v_min = 0.0
            v_max = np.percentile(clip_base, percentile_clip[1])

        denom = v_max - v_min if v_max > v_min else 1.0
        normalized = np.clip((valid_values - v_min) / denom, 0.0, 1.0)

        if gamma != 1.0:
            normalized = np.power(normalized, gamma)

        norm_vals[valid_mask] = normalized

    if debug:
        nonzero_in_data = (values > 0).sum()
        print(f"  [spot_render] total spots in level: {len(values):,}  |  nonzero values: {nonzero_in_data:,}")
        print(f"  [spot_render] v_min={v_min:.3f}  v_max={v_max:.3f}  nonzero norm_vals: {(norm_vals > 0).sum():,}")

    # --- STEP 2: Spatial filtering ---
    # Only render spots near the target canvas to avoid memory blowup
    left, right, bottom, top = base_image.extent
    bin_um = omic.metadata.get('bin_size_um', 8.0)
    margin = bin_um * 4  # generous margin to prevent edge clipping

    in_bounds = (
        (all_coords_um[:, 0] >= min(left, right) - margin) &
        (all_coords_um[:, 0] <= max(left, right) + margin) &
        (all_coords_um[:, 1] >= min(bottom, top) - margin) &
        (all_coords_um[:, 1] <= max(bottom, top) + margin)
    )

    coords_um = all_coords_um[in_bounds]
    render_vals = norm_vals[in_bounds]

    if debug:
        print(f"  [spot_render] proxy extent: x=[{min(left,right):.0f}, {max(left,right):.0f}]  y=[{min(bottom,top):.0f}, {max(bottom,top):.0f}]  (margin={margin:.0f}µm)")
        print(f"  [spot_render] spots in proxy AABB: {in_bounds.sum():,}  |  nonzero after filter: {(render_vals > 0).sum():,}")
    
    if len(coords_um) == 0:
        h, w = base_image.image.shape[:2]
        return ImageChannel(
            name=f"render_{omic.name}",
            image=np.zeros((h, w, 4), dtype=np.uint8),
            spatial_meta=base_image.spatial_meta,
            metadata={"source_omic": omic.name, "renderer": "empty"}
        )

    # --- STEP 3: Grid alignment ---
    is_circle = 1 if spot_shape in ['spot', 'circle'] else 0
    
    if is_circle == 0:
        # Estimate angle using ALL coords for global accuracy
        theta = _estimate_grid_angle(all_coords_um)
        upsample = 2.0  # 2x super-resolution for clean square edges
    else:
        theta = 0.0
        upsample = 1.0
        
    # Rotate physical coords to be axis-aligned with the spot grid
    center = np.mean(all_coords_um, axis=0)  # global center for consistency
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    R_neg = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    coords_centered = coords_um - center
    coords_rot = (R_neg @ coords_centered.T).T
    
    # Bounding box of the rotated filtered subset
    min_x_rot, min_y_rot = np.min(coords_rot, axis=0) - bin_um
    max_x_rot, max_y_rot = np.max(coords_rot, axis=0) + bin_um
    
    # Native canvas resolution
    render_mpp = base_image.mpp / upsample
    
    # Map rotated coords to native canvas pixels
    x_px = (coords_rot[:, 0] - min_x_rot) / render_mpp
    y_px = (coords_rot[:, 1] - min_y_rot) / render_mpp
    
    w = int(np.ceil((max_x_rot - min_x_rot) / render_mpp))
    h = int(np.ceil((max_y_rot - min_y_rot) / render_mpp))
    radius_px = int(max(1, (bin_um / 2) / render_mpp))

    if debug:
        print(f"  [spot_render] bin_um={bin_um}  render_mpp={render_mpp:.3f}  radius_px={radius_px}  native canvas={w}x{h}")

    # --- STEP 4: Numba render into native grid ---
    lut = make_colormap_lut(cmap_name)
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    
    gamma_val = -1.0 if alpha_gamma is None else float(alpha_gamma)
    pass_val  = float(alpha_pass) if alpha_pass is not None else 1.0

    _numba_render_spots_fast(
        x_px.astype(np.float32), y_px.astype(np.float32), render_vals, 
        canvas, lut, radius_px, float(alpha), gamma_val, pass_val, is_circle
    )
    
    # --- STEP 5: Attach spatial truth to the native canvas ---
    # M_pix_to_rot: native pixels -> rotated micron space
    M_pix_to_rot = np.array([
        [render_mpp, 0,          min_x_rot],
        [0,          render_mpp, min_y_rot],
        [0,          0,          1        ]
    ])
    
    # M_rot_to_true: rotated microns -> absolute microns (undoes the rotation)
    cos_p, sin_p = np.cos(theta), np.sin(theta)
    M_rot_to_true = np.array([
        [cos_p, -sin_p, center[0]],
        [sin_p,  cos_p, center[1]],
        [0,      0,     1        ]
    ])
    
    # Chain: pixels -> rotated space -> true absolute space
    M_native = M_rot_to_true @ M_pix_to_rot
    
    native_channel = ImageChannel(
        name=f"native_render_{omic.name}", 
        image=canvas, 
        spatial_meta=SpatialMetadata(
            transform_matrix=M_native,
            unit=base_image.spatial_meta.unit
        ),
        metadata={"source_omic": omic.name, "renderer": "visium_spots_native"}
    )
    
    # return_native skips the ViewPort warp and hands back the axis-aligned
    # canvas directly — same pixel grid Cellpose and the DataPort will consume.
    if return_native:
        return native_channel

    # Warp native canvas onto the target image grid via affine transform.
    # This handles rotation correction and downsampling in one clean step.
    matched_channel = match_image_to_reference(native_channel, base_image)
    matched_channel.name = f"render_{omic.name}"

    return matched_channel


def render_spotset_to_density_map(
    omic: OmicChannel,
    extent: Tuple[float, float, float, float],
    bin_um: float = 4.0,
    values: Optional[np.ndarray] = None,
) -> ImageChannel:
    """
    Renders a Visium spotset into a uint16 density map at native bin resolution.

    One pixel = one bin_um × bin_um physical area. Pixel values are raw counts
    (not normalized), making this suitable for direct Cellpose input or any
    quantitative per-bin analysis.

    Grid rotation is handled identically to render_spotset_to_image: the native
    canvas is axis-aligned to the spot grid, with the correct affine matrix attached
    so spatial registration is preserved in absolute micron space.

    The pyramid level is chosen automatically via get_optimal_level(bin_um), so
    passing bin_um=4.0 renders from the 4µm pyramid level if it exists.

    Args:
        omic:    OmicChannel containing the spot data (with pyramid).
        extent:  ROI in microns as (min_x, max_x, min_y, max_y).
        bin_um:  Bin / pixel size in microns. Selects pyramid level automatically.
        values:  Optional per-spot value array. Must match the selected pyramid
                 level's obs count. Defaults to total_counts.
    """
    # Select the appropriate pyramid level
    _, adata = omic.get_optimal_level(bin_um)
    actual_bin_um = float(adata.uns.get('bin_size_um', bin_um))

    all_coords_um = adata.obsm['spatial_fine'] if 'spatial_fine' in adata.obsm else adata.obsm['spatial']
    vals_arr: np.ndarray = adata.obs['total_counts'].values if values is None else values

    # Filter to ROI + small margin so edge bins aren't clipped
    min_x, max_x, min_y, max_y = extent
    margin = actual_bin_um * 2
    in_bounds = (
        (all_coords_um[:, 0] >= min_x - margin) &
        (all_coords_um[:, 0] <= max_x + margin) &
        (all_coords_um[:, 1] >= min_y - margin) &
        (all_coords_um[:, 1] <= max_y + margin)
    )
    coords_um = all_coords_um[in_bounds]
    vals      = vals_arr[in_bounds]

    if len(coords_um) == 0:
        n_cols = int(round((max_x - min_x) / actual_bin_um))
        n_rows = int(round((max_y - min_y) / actual_bin_um))
        return ImageChannel(
            name=f"density_{omic.name}",
            image=np.zeros((n_rows, n_cols), dtype=np.uint16),
            spatial_meta=SpatialMetadata(
                transform_matrix=create_transform_matrix(actual_bin_um, actual_bin_um, min_x, min_y),
                unit=omic.spatial_meta.unit
            ),
            metadata={"source_omic": omic.name, "renderer": "density_map", "bin_um": actual_bin_um}
        )

    # Estimate grid rotation using the full dataset for global consistency
    theta  = _estimate_grid_angle(all_coords_um)
    center = np.mean(all_coords_um, axis=0)

    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    R_neg = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    coords_rot = (R_neg @ (coords_um - center).T).T

    min_x_rot = np.min(coords_rot[:, 0]) - actual_bin_um
    min_y_rot = np.min(coords_rot[:, 1]) - actual_bin_um
    max_x_rot = np.max(coords_rot[:, 0]) + actual_bin_um
    max_y_rot = np.max(coords_rot[:, 1]) + actual_bin_um

    n_cols = int(np.ceil((max_x_rot - min_x_rot) / actual_bin_um))
    n_rows = int(np.ceil((max_y_rot - min_y_rot) / actual_bin_um))

    # Direct grid fill: one bin → one pixel, no interpolation
    ix = np.floor((coords_rot[:, 0] - min_x_rot) / actual_bin_um).astype(int)
    iy = np.floor((coords_rot[:, 1] - min_y_rot) / actual_bin_um).astype(int)
    valid = (ix >= 0) & (ix < n_cols) & (iy >= 0) & (iy < n_rows)

    grid = np.zeros((n_rows, n_cols), dtype=np.float32)
    np.add.at(grid, (iy[valid], ix[valid]), vals[valid])
    density = np.clip(grid, 0, 65535).astype(np.uint16)

    # Affine: pixels → rotated space → absolute micron space
    M_pix_to_rot = np.array([
        [actual_bin_um, 0,             min_x_rot],
        [0,             actual_bin_um, min_y_rot],
        [0,             0,             1        ]
    ])
    cos_p, sin_p = np.cos(theta), np.sin(theta)
    M_rot_to_true = np.array([
        [cos_p, -sin_p, center[0]],
        [sin_p,  cos_p, center[1]],
        [0,      0,     1        ]
    ])
    M_native = M_rot_to_true @ M_pix_to_rot

    return ImageChannel(
        name=f"density_{omic.name}",
        image=density,
        spatial_meta=SpatialMetadata(
            transform_matrix=M_native,
            unit=omic.spatial_meta.unit
        ),
        metadata={"source_omic": omic.name, "renderer": "density_map", "bin_um": actual_bin_um}
    )
