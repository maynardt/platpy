import numpy as np
import matplotlib.pyplot as plt
import cv2
from numba import njit, prange
from typing import Union, Optional, Tuple, List, Dict
import dask.array as da

# platpy imports
from .channels.image_channel import ImageChannel
from .channels.omic_channel import OmicChannel
from .transform_utils import SpatialMetadata

# =============================================================================
# CORE NORMALIZATION UTILITIES
# =============================================================================

def normalize_array(arr: np.ndarray, percentile_clip: Tuple[float, float] = (0, 100)) -> np.ndarray:
    """
    Normalizes an array to 0.0 - 1.0 floats.
    Crucially ignores absolute zero background when calculating percentiles
    so sparse spatial images aren't washed out.
    """
    arr_f = np.array(arr, dtype=np.float32)
    valid_pixels = arr_f[arr_f > 0] if np.any(arr_f > 0) else arr_f

    if len(valid_pixels) == 0:
        return np.zeros_like(arr_f)

    p_lo, p_hi = np.percentile(valid_pixels, percentile_clip)

    if abs(p_hi - p_lo) < 1e-5:
        return np.where(arr_f > 0, 1.0, 0.0).astype(np.float32)

    arr_f = np.clip(arr_f, p_lo, p_hi)
    return (arr_f - p_lo) / (p_hi - p_lo)


def make_colormap_lut(cmap_name: str = "plasma", lut_size: int = 256) -> np.ndarray:
    import matplotlib.colors as mcolors
    # Custom IF colormaps: 'if_<any color name>' e.g. 'if_xkcd:neon green', 'if_red'
    if cmap_name.startswith("if_"):
        color_name = cmap_name[3:]  # strip 'if_' prefix
        try:
            base_color = np.array(mcolors.to_rgb(color_name))
        except ValueError:
            base_color = np.array([0.0, 1.0, 0.0])  # fallback green
        lut = np.zeros((lut_size, 4), dtype=np.uint8)
        factors = np.linspace(0, 1, lut_size)
        lut[:, 0] = (base_color[0] * factors * 255).astype(np.uint8)
        lut[:, 1] = (base_color[1] * factors * 255).astype(np.uint8)
        lut[:, 2] = (base_color[2] * factors * 255).astype(np.uint8)
        lut[:, 3] = (factors * 255).astype(np.uint8)
        return lut
    # Standard matplotlib colormaps
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        cmap = plt.get_cmap("plasma")
    lut = (cmap(np.linspace(0, 1, lut_size)) * 255).astype(np.uint8)
    lut[0, 3] = 0  # transparent background bin
    return lut


def create_new_image(
    size: Tuple[float, float, float, float],
    resolution: float,
    background_color: str = 'black',
    name: str = "new_image"
) -> ImageChannel:
    """
    Creates a blank ImageChannel canvas for a given physical extent and resolution.
    size: (min_x, max_x, min_y, max_y) in microns
    resolution: microns per pixel
    """
    from .transform_utils import create_transform_matrix

    min_x, max_x, min_y, max_y = size
    width_px  = int(np.ceil((max_x - min_x) / resolution))
    height_px = int(np.ceil((max_y - min_y) / resolution))

    if background_color is None:
        background_color = 'transparent'

    color_map = {'black': 0, 'white': 255}
    fill_val = color_map.get(background_color.lower(), 0)
    canvas = np.full((height_px, width_px, 4), fill_val, dtype=np.uint8)
    canvas[..., 3] = 255  # Full alpha

    spatial = SpatialMetadata(
        transform_matrix=create_transform_matrix(
            scale_x=resolution, scale_y=resolution, tx=min_x, ty=min_y
        ),
        local_bounds=(0.0, float(width_px), 0.0, float(height_px)),
        unit="microns"
    )

    return ImageChannel(name=name, image=canvas, spatial_meta=spatial)


# =============================================================================
# HARDWARE-ACCELERATED BACKENDS
# =============================================================================

@njit(parallel=True)
def _numba_render_spots_fast(x_px, y_px, values, rgba_out, colormap_lut, spot_radius_px, alpha, alpha_gamma, alpha_pass, is_circle):
    """(Type A Backend) RGBA spot rendering with thresholded dynamic alpha."""
    H, W = rgba_out.shape[:2]
    r2 = spot_radius_px * spot_radius_px

    for i in prange(x_px.shape[0]):
        x, y, v = x_px[i], y_px[i], values[i]
        if np.isnan(v): continue

        xi, yi = int(x), int(y)
        val_idx = min(max(int(v * 255), 0), 255)

        # Calculate dynamic alpha
        if alpha_gamma < 0.0:
            spot_alpha = alpha
        else:
            if v >= alpha_pass: spot_alpha = alpha
            else: spot_alpha = alpha * ((v / alpha_pass if alpha_pass > 0.0 else 1.0) ** alpha_gamma)

        alpha_val = int(255 * spot_alpha)

        for dy in range(-spot_radius_px, spot_radius_px + 1):
            for dx in range(-spot_radius_px, spot_radius_px + 1):
                if is_circle == 1 and (dx*dx + dy*dy > r2): continue
                yy, xx = yi + dy, xi + dx
                if 0 <= yy < H and 0 <= xx < W:
                    color = colormap_lut[val_idx]
                    rgba_out[yy, xx, 0] = color[0]
                    rgba_out[yy, xx, 1] = color[1]
                    rgba_out[yy, xx, 2] = color[2]
                    rgba_out[yy, xx, 3] = alpha_val

@njit(parallel=True)
def _numba_render_spots_gray_fast(x_px, y_px, values, gray_out, spot_radius_px, is_circle):
    """(Type B/C Backend) Fast raw 2D grayscale intensity mapping."""
    H, W = gray_out.shape
    r2 = spot_radius_px * spot_radius_px

    for i in prange(x_px.shape[0]):
        x, y, v = x_px[i], y_px[i], values[i]
        if np.isnan(v): continue

        xi, yi = int(x), int(y)
        for dy in range(-spot_radius_px, spot_radius_px + 1):
            for dx in range(-spot_radius_px, spot_radius_px + 1):
                if is_circle == 1 and (dx*dx + dy*dy > r2): continue
                yy, xx = yi + dy, xi + dx
                if 0 <= yy < H and 0 <= xx < W:
                    if v > gray_out[yy, xx]:
                        gray_out[yy, xx] = v


def _render_xenium_to_array(data, extent, shape, gene_name=None, bin_um=None) -> np.ndarray:
    """Fast 2D histograms for point clouds.

    bin_um: histogram bin size in microns. Larger values accumulate more transcripts
    per bin, giving brighter signal for sparse genes. The coarsened histogram is
    bilinearly upsampled back to shape before returning.
    """
    from .lazy_transcripts import LazyTranscripts

    left, right, bottom, top = extent
    h, w = shape[:2]
    q_min_y, q_max_y = min(bottom, top), max(bottom, top)
    gene_col = 'feature'

    if isinstance(data, LazyTranscripts):
        # A render is almost always a small viewport out of a multi-million-row
        # section -- resolve to the overlapping blocks and materialize only
        # those rows, rather than ever touching the rest of the point cloud.
        hit_ranges = data.row_ranges_for_block_index(left, right, q_min_y, q_max_y)
        df = data.materialize(hit_ranges)
    else:
        # Non-lazy fallback (in-memory-constructed data, older bundles, etc.)
        df = data.obs if hasattr(data, 'obs') else data
        gene_col = 'feature' if 'feature' in df.columns else 'feature_name'
        block_index = getattr(data, 'uns', {}).get('block_index') if hasattr(data, 'uns') else None
        if block_index:
            from .io_utils import query_block_index
            hit_ranges = query_block_index(block_index, left, right, q_min_y, q_max_y)
            if not hit_ranges:
                df = df.iloc[0:0]
            else:
                row_idx = np.concatenate([np.arange(r0, r1) for r0, r1 in hit_ranges])
                df = df.iloc[row_idx]

    subset = df if gene_name is None else df[df[gene_col] == gene_name]

    if bin_um is not None and bin_um > 0:
        w_bins = max(1, int(abs(right - left)  / bin_um))
        h_bins = max(1, int(abs(top   - bottom) / bin_um))
    else:
        w_bins, h_bins = w, h

    x_bins = np.linspace(left, right, w_bins + 1)
    y_bins = np.linspace(min(bottom, top), max(bottom, top), h_bins + 1)
    H, _, _ = np.histogram2d(subset['x'], subset['y'], bins=[x_bins, y_bins])

    img_array = H.T.astype(np.float32)
    if top < bottom:
        img_array = np.flipud(img_array)

    if (w_bins, h_bins) != (w, h):
        img_array = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_LINEAR)

    return img_array


# =============================================================================
# EXPLICIT OMIC RENDERING PATHWAYS
# =============================================================================

def render_omic_flat(
    omic: OmicChannel, target_extent: Tuple[float, float, float, float], target_shape: Tuple[int, int],
    gene_name: str = None, bin_um: float = None   #type:ignore
) -> np.ndarray:
    """
    TYPE C: Flat analytical render.
    Returns RAW data mapping (no clipping, no gamma, no alpha).
    Used purely for computation (e.g. raw input for clustering).
    """
    geom_type = omic.metadata.get('geometry_type', 'point')

    if geom_type == 'point':
        return _render_xenium_to_array(omic.data, target_extent, target_shape, gene_name, bin_um=bin_um)
    else:
        h, w = target_shape[:2]
        out_arr = np.zeros((h, w), dtype=np.float32)
        return out_arr

def render_omic_grayscale(
    omic: OmicChannel, target_extent: Tuple[float, float, float, float], target_shape: Tuple[int, int],
    gene_name: str = None, percentile_clip: Tuple[float, float] = (1, 99), gamma: float = 1.0  #type:ignore
) -> ImageChannel:
    """
    TYPE B: Grayscale visual render.
    Generates a scaled 0-255 uint8 grayscale image. Excludes zeros for clipping.
    Excellent for CellPose segmentation or IF-style colorization.
    """
    raw_array = render_omic_flat(omic, target_extent, target_shape, gene_name)

    norm_array = normalize_array(raw_array, percentile_clip=percentile_clip)

    if gamma != 1.0:
        norm_array = np.power(norm_array, 1.0 / gamma)

    gray_8bit = (np.clip(norm_array, 0, 1) * 255).astype(np.uint8)

    return ImageChannel(
        name=f"{omic.name}_{gene_name}_gray",
        image=gray_8bit,
        spatial_meta=SpatialMetadata()
    )

def render_omic_color_mapped(
    omic: OmicChannel, target_extent: Tuple[float, float, float, float], target_shape: Tuple[int, int],
    gene_name: str = None, cmap_name: str = 'plasma', percentile_clip: Tuple[float, float] = (1, 99),   #type:ignore
    gamma: float = 1.0, alpha: float = 1.0, alpha_gamma: float = -1.0, alpha_pass: float = 1.0,
    bin_um: float = None   #type:ignore
) -> ImageChannel:
    """TYPE A: Full color-mapped RGBA render with dynamic alpha."""
    raw_array = render_omic_flat(omic, target_extent, target_shape, gene_name, bin_um=bin_um)

    # Count histograms always start from zero — normalize_array's lower percentile
    # clip would otherwise map single-count bins (the most common value in sparse data)
    # to the same value as empty bins, making sparse genes appear blank.
    valid = raw_array[raw_array > 0]
    if valid.size == 0:
        norm_array = np.zeros_like(raw_array)
    else:
        p_hi = float(np.percentile(valid, percentile_clip[1]))
        p_hi = max(p_hi, 1.0)
        norm_array = np.clip(raw_array / p_hi, 0.0, 1.0).astype(np.float32)
        if gamma != 1.0:
            norm_array = np.power(norm_array, 1.0 / gamma)

    h, w = target_shape[:2]
    rgba_out = np.zeros((h, w, 4), dtype=np.uint8)
    lut = make_colormap_lut(cmap_name)

    indices = (norm_array * 255).astype(np.uint8)
    rgba_out[..., :3] = lut[indices, :3]

    if alpha_gamma < 0:
        rgba_out[..., 3] = np.where(norm_array > 0, int(255 * alpha), 0)
    else:
        alpha_map = np.where(
            norm_array >= alpha_pass,
            alpha,
            alpha * ((norm_array / alpha_pass) ** alpha_gamma)
        )
        rgba_out[..., 3] = (np.clip(alpha_map, 0, 1) * 255).astype(np.uint8)

    return ImageChannel(
        name=f"{omic.name}_{gene_name}_rgba",
        image=rgba_out,
        spatial_meta=SpatialMetadata()
    )


_SPOT_RENDER_KWARGS = {
    'values', 'cmap_name', 'alpha', 'alpha_gamma', 'alpha_pass',
    'scale_min', 'scale_max', 'percentile_clip', 'gamma', 'spot_shape'
}

def _extract_gene_values(omic: 'OmicChannel', gene_name: Optional[str]) -> Optional[np.ndarray]:
    """Pull per-spot values for a gene (or obs column) from an AnnData OmicChannel."""
    if gene_name is None:
        return None
    if gene_name in omic.data.obs:
        return omic.data.obs[gene_name].values
    if gene_name in omic.data.var_names:
        return np.array(omic.data[:, gene_name].X.todense()).flatten()
    return None


def render_omic_to_image(
    omic: 'OmicChannel',
    base_image: 'ImageChannel',
    gene_name: str = None,                          #type: ignore
    debug: bool = False,
    **kwargs
) -> 'ImageChannel':
    """
    EASY MODE: Renders an OmicChannel to perfectly match the bounds and
    resolution of an existing canvas ImageChannel.
    Routes to the grid-aligned Visium spot renderer or Xenium histogram renderer.
    Auto-selects the optimal pyramid level from the OmicChannel based on the
    canvas resolution so the caller never needs to specify a bin size explicitly.
    """
    # Auto-select the best pyramid level for this canvas resolution.
    # For Xenium (continuous), get_optimal_level always returns the base data unchanged.
    # For Visium HD (binned), use 8× the canvas mpp so that each rendered spot spans
    # ~4 pixels — fine bins (2-4µm) would otherwise produce sub-pixel dots that are
    # invisible at typical viewport scales even though they render correctly.
    _, level_data = omic.get_optimal_level(base_image.mpp * 8)
    if level_data is not omic.data:
        from .channels.omic_channel import OmicChannel as _OmicChannel
        level_bin_um = getattr(level_data, 'uns', {}).get('bin_size_um', omic.metadata.get('bin_size_um'))
        level_meta = {**omic.metadata, 'bin_size_um': level_bin_um} if level_bin_um is not None else omic.metadata.copy()
        omic = _OmicChannel(name=omic.name, data=level_data, spatial_meta=omic.spatial_meta, metadata=level_meta)

    if debug:
        print(f"  [render_omic] canvas mpp={base_image.mpp:.3f}  level selected: bin_size_um={omic.metadata.get('bin_size_um')}  n_obs={omic.data.n_obs:,}  gene='{gene_name}'")

    # --- Visium path: fast grid-aligned spot renderer ---
    if "bin_size_um" in omic.metadata:
        from platpy.spot_render import render_spotset_to_image
        values = _extract_gene_values(omic, gene_name)
        spot_shape = omic.metadata.get('geometry_type', 'square')
        spot_kwargs = {k: v for k, v in kwargs.items() if k in _SPOT_RENDER_KWARGS}
        if spot_kwargs.get('alpha') is None:
            spot_kwargs['alpha'] = 1.0
        rendered = render_spotset_to_image(
            omic=omic, base_image=base_image, values=values,
            spot_shape=spot_shape, debug=debug, **spot_kwargs
        )
        rendered.spatial_meta = base_image.spatial_meta

        # Composite over background_color if specified (Porter-Duff over)
        bg = kwargs.get('background_color', 'transparent')
        if bg and bg.lower() not in ('transparent', 'none'):
            import matplotlib.colors as mcolors
            rgb = np.array(mcolors.to_rgb(bg), dtype=np.float32)
            img = rendered.image.astype(np.float32) / 255.0
            if img.ndim == 3 and img.shape[-1] == 4:
                a = img[..., 3:4]
                img[..., :3] = img[..., :3] * a + rgb * (1.0 - a)
                img[..., 3] = 1.0
            rendered.image = (img * 255).astype(np.uint8)

        return rendered

    # --- Xenium path: histogram point-cloud renderer ---
    target_extent = base_image.spatial_meta.extent
    target_shape = base_image.image.shape[:2]
    rendered_channel = render_omic_color_mapped(
        omic=omic,
        target_extent=target_extent,
        target_shape=target_shape,                  #type: ignore
        gene_name=gene_name,
        **{k: v for k, v in kwargs.items() if k != 'background_color'}
    )
    rendered_channel.spatial_meta = base_image.spatial_meta
    return rendered_channel


# =============================================================================
# IMAGE MANIPULATION UTILITIES
# =============================================================================

def render_image_to_bounds(source_channel, min_x_um, max_x_um, min_y_um, max_y_um, target_width_px):
    """
    Generates a new lightweight ImageChannel for a specific physical ROI.
    Crops natively (preserving sub-pixel origins) then resamples to the
    requested pixel width.
    """
    roi_poly_um = np.array([
        [min_x_um, min_y_um],
        [max_x_um, min_y_um],
        [max_x_um, max_y_um],
        [min_x_um, max_y_um]
    ])

    cropped = source_channel.crop_to_micron_roi(roi_poly_um)

    src_img = cropped.image
    if isinstance(src_img, da.Array):
        src_img = src_img.compute()

    micron_width  = float(max_x_um - min_x_um)
    micron_height = float(max_y_um - min_y_um)

    mpp = micron_width / target_width_px
    target_height_px = int(micron_height / mpp)

    resampled_img = cv2.resize(
        src_img,
        (target_width_px, target_height_px),
        interpolation=cv2.INTER_LANCZOS4
    )

    target_spatial = cropped.spatial_meta
    M = target_spatial.transform_matrix
    M[0, 0] = mpp
    M[1, 1] = mpp

    new_meta = cropped.metadata.copy() if cropped.metadata else {}
    new_meta['resampled_width_px'] = target_width_px
    new_meta['roi_microns'] = (min_x_um, max_x_um, min_y_um, max_y_um)

    return ImageChannel(
        name=f"resampled_{target_width_px}px_{source_channel.name}",
        image=resampled_img,
        spatial_meta=target_spatial,
        metadata=new_meta
    )


def match_image_to_reference(source_channel, reference_channel):
    """
    Forces a source image to perfectly match the bounding box, pixel dimensions,
    and spatial metadata of a reference image using pure affine transformation.
    Pre-crops before warping to avoid loading massive Dask arrays into RAM.
    """
    from numpy.linalg import inv

    ref_extent = reference_channel.spatial_meta.extent
    try:
        source_cropped = source_channel.crop_to_micron_roi(
            min_x_um=ref_extent[0],
            max_x_um=ref_extent[1],
            min_y_um=ref_extent[2],
            max_y_um=ref_extent[3]
        )
    except ValueError:
        # No overlap — fall back to original (will render as blank)
        source_cropped = source_channel

    M_src = source_cropped.spatial_meta.transform_matrix
    M_ref = reference_channel.spatial_meta.transform_matrix

    M_warp = inv(M_ref) @ M_src
    M_warp_2x3 = M_warp[:2, :]

    ref_h, ref_w = reference_channel.image.shape[:2] if hasattr(reference_channel, 'image') else reference_channel.shape

    src_img = source_cropped.image
    if isinstance(src_img, da.Array):
        src_img = src_img.compute()

    matched_img = cv2.warpAffine(
        src_img,
        M_warp_2x3,
        (ref_w, ref_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    new_meta = source_channel.metadata.copy() if source_channel.metadata else {}
    new_meta['matched_to'] = getattr(reference_channel, 'name', 'viewport')
    new_meta['resample_method'] = 'affine_lanczos4'

    return ImageChannel(
        name=f"matched_{source_channel.name}",
        image=matched_img,
        spatial_meta=reference_channel.spatial_meta,
        metadata=new_meta
    )


def composite_overlay(
    base_channel: 'ImageChannel',
    overlay_channel: Union['ImageChannel', 'OmicChannel'],
    blend_mode: str = 'normal',
    **kwargs
) -> 'ImageChannel':
    """
    EASY MODE: Composites an overlay image or omic layer onto a base image.
    Automatically handles spatial registration and alpha blending.
    Useful before the caller knows how to use ViewPorts directly.

    Blend modes:
    - 'normal':  Standard alpha transparency layering.
    - 'lighten': Maximum intensity (best for stacking IF fluorescence).
    - 'darken':  Minimum intensity (best for stacking ink/stains).
    """
    from .viewport import ViewPort

    bg_color = kwargs.pop('background_color', 'black')
    mpp = abs(base_channel.spatial_meta.transform_matrix[0, 0])

    # Union extent so staggered/offset channels aren't clipped to the base boundary
    base_ext = base_channel.spatial_meta.extent
    if isinstance(overlay_channel, OmicChannel):
        ov_ext = base_ext  # omics don't have a fixed image extent
    else:
        ov_ext = overlay_channel.spatial_meta.extent

    union_size = (
        min(base_ext[0], ov_ext[0]),
        max(base_ext[1], ov_ext[1]),
        min(base_ext[2], ov_ext[2]),
        max(base_ext[3], ov_ext[3]),
    )

    union_canvas = create_new_image(size=union_size, resolution=mpp,
                                    background_color=bg_color, name="composite_canvas")

    vp = ViewPort.from_image(union_canvas)
    vp.add_layer(union_canvas, blend_mode='normal')
    vp.add_layer(base_channel, blend_mode='normal')
    vp.add_layer(overlay_channel, blend_mode=blend_mode, **kwargs)

    return vp.render(output_name=f"{base_channel.name}_composite")
