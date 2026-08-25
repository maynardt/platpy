import logging
import matplotlib.pyplot as plt
import numpy as np
import dask.array as da
from dataclasses import dataclass, field
from typing import List, Union, Optional, Any

# platpy imports
from .channels import ImageChannel

@dataclass
class PlotObject:
    """
    Bundles a plottable object with its display metadata so plot_image_grid
    can render scatter, UMAP, and histogram panels alongside images without
    per-call matplotlib boilerplate.

    Args:
        data:      AnnData, ImageChannel, np.ndarray, or pd.Series.
        plot_type: 'scatter' | 'umap' | 'histogram' | 'image' | 'auto'.
                   'auto' resolves to 'image' for ImageChannel / ndarray inputs.
        color:     obs column name for scatter/umap coloring, or a matplotlib
                   color string for histogram bars.
        palette:   dict {category: color} or list of colors for categorical data.
                   Falls back to adata.uns['{color}_colors'] if present, then tab20.
        title:     Panel title (overrides auto-generated titles).
        xlabel:    x-axis label override.
        ylabel:    y-axis label override.
        s:         Scatter point size.
        alpha:     Scatter / histogram bar transparency.
        rois:      Dict of ROI polygons {name: [(x, y), ...]} overlaid on scatter.
        bins:      Number of histogram bins.
        x_col:     AnnData obs column for scatter x coordinates.
        y_col:     AnnData obs column for scatter y coordinates.
        legend:    Whether to show a categorical legend on scatter panels.
        vlines:    List of vertical-line annotations for histogram panels.
                   Each entry is a dict with keys: x (required), color, ls, label.
                   Example: [{'x': 0.3, 'color': 'red', 'ls': '--', 'label': 'threshold'}]
        label:     Legend label for this series when overlaying multiple PlotObjects
                   on the same axes (pass a tuple to plot_image_grid).
        annotations: Reserved for future arrow / callout overlays.
    """
    data:        Any
    plot_type:   str            = 'auto'
    color:       Optional[str]  = None
    palette:     Optional[Any]  = None
    title:       Optional[str]  = None
    xlabel:      Optional[str]  = None
    ylabel:      Optional[str]  = None
    s:           Optional[float] = None
    alpha:       float          = 0.4
    base_color:  str            = 'steelblue'
    rois:        Optional[dict] = None
    roi_colors:  Optional[dict] = None
    bins:        int            = 50
    x_col:       str            = 'x_micron'
    y_col:       str            = 'y_micron'
    legend:      bool           = True
    legend_loc:  str            = 'lower left'
    vlines:      Optional[list] = None
    label:       Optional[str]  = None
    annotations: Optional[list] = None

    def __post_init__(self):
        if self.plot_type == 'auto':
            if hasattr(self.data, 'spatial_meta') or isinstance(self.data, (np.ndarray, da.Array)):
                self.plot_type = 'image'


def _render_plot_object(ax: Any, po: PlotObject, fs: dict) -> None:
    """Render a non-image PlotObject onto an existing matplotlib Axes."""
    data = po.data

    if po.plot_type == 'scatter':
        x = data.obs[po.x_col].values
        y = data.obs[po.y_col].values

        if po.color:
            col_vals = data.obs[po.color]
            cats = (col_vals.cat.categories.tolist() if hasattr(col_vals, 'cat')
                    else sorted(col_vals.dropna().unique().tolist()))

            # Palette priority: explicit > adata.uns > tab20 fallback
            palette = po.palette
            if palette is None:
                uns_key = f'{po.color}_colors'
                if uns_key in getattr(data, 'uns', {}):
                    palette = dict(zip(data.obs[po.color].cat.categories.tolist(),
                                       data.uns[uns_key]))
            if palette is None:
                raw = plt.cm.tab20(np.linspace(0, 1, len(cats)))
                palette = {str(c): tuple(raw[k]) for k, c in enumerate(cats)}

            s = po.s if po.s is not None else 1.0
            for cat in cats:
                mask = (col_vals == cat).values
                c = palette.get(str(cat), (0.5, 0.5, 0.5, 1.0))
                ax.scatter(x[mask], y[mask],
                           c=[c] if isinstance(c, tuple) else c,
                           s=s, alpha=po.alpha, label=str(cat), rasterized=True)
        else:
            ax.scatter(x, y, s=po.s if po.s is not None else 1.0,
                       alpha=po.alpha, c=po.base_color, rasterized=True)

        ax.set_aspect('equal')
        ax.invert_yaxis()

        if po.rois:
            for roi_name, coords in po.rois.items():
                xs = [pt[0] for pt in coords] + [coords[0][0]]
                ys = [pt[1] for pt in coords] + [coords[0][1]]
                c  = (po.roi_colors or {}).get(roi_name, None)
                ax.plot(xs, ys, lw=1.5, zorder=5, label=roi_name,
                        **({'color': c} if c else {}))

        if po.legend:
            if po.legend_loc == 'outside right':
                ax.legend(markerscale=5, fontsize=fs['tick'],
                          bbox_to_anchor=(1.01, 1), loc='upper left')
            else:
                ax.legend(markerscale=5, fontsize=fs['tick'], loc=po.legend_loc)

    elif po.plot_type == 'umap':
        try:
            import scanpy as sc
            sc.pl.umap(data, color=po.color, ax=ax, show=False,
                       palette=po.palette, title='', size=po.s,
                       legend_loc='none' if not po.legend else po.legend_loc)
        except ImportError:
            ax.text(0.5, 0.5, 'scanpy required for UMAP plots',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=fs['axis'])

    elif po.plot_type == 'histogram':
        if hasattr(data, 'obs') and po.color and po.color in data.obs.columns:
            vals  = data.obs[po.color].dropna().values
            bar_c = 'steelblue'
        else:
            vals  = np.asarray(data).ravel()
            bar_c = po.color if isinstance(po.color, str) else 'steelblue'
        hist_kwargs = {'label': po.label} if po.label else {}
        ax.hist(vals, bins=po.bins, color=bar_c, alpha=po.alpha, edgecolor='none', **hist_kwargs)
        if po.vlines:
            for vl in po.vlines:
                ax.axvline(vl['x'], color=vl.get('color', 'red'),
                           ls=vl.get('ls', '--'), lw=1.5, label=vl.get('label'))
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=fs['tick'])

    # Shared label / title overrides (applied after all plot types)
    if po.title:
        ax.set_title(po.title, fontsize=fs['title'], fontweight='bold')
    if po.plot_type == 'scatter':
        ax.set_xlabel(po.xlabel or 'x (µm)', fontsize=fs['axis'])
        ax.set_ylabel(po.ylabel or 'y (µm)', fontsize=fs['axis'])
    else:
        if po.xlabel:
            ax.set_xlabel(po.xlabel, fontsize=fs['axis'])
        if po.ylabel:
            ax.set_ylabel(po.ylabel, fontsize=fs['axis'])
    ax.tick_params(axis='both', labelsize=fs['tick'])


def fig_to_image(fig):
    """
    Capture a matplotlib Figure as a float32 RGB array [0, 1], then close it.

    Use this to convert any matplotlib-based output (seaborn, squidpy, etc.)
    into a raw image array that can be passed directly to plot_image_grid.

    Uses savefig → PNG → imread to guarantee correct alpha compositing.
    buffer_rgba() can behave inconsistently with Jupyter's inline backend;
    savefig forces a clean Agg render that preserves scatter/patch transparency.
    Alpha channel is dropped because point alphas are already baked into the
    RGB values during rendering.
    """
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=fig.dpi)
    buf.seek(0)
    img = plt.imread(buf)   # float32 RGBA [0, 1]
    plt.close(fig)
    return img[:, :, :3]    # drop alpha — already composited into RGB


def _normalize_plot_items(*args):
    """Helper to coerce and flatten *args into a single list of items."""
    flat_list = []
    for arg in args:
        if isinstance(arg, list):
            flat_list.extend(arg)
        else:
            flat_list.append(arg)  # Keep None values to allow empty spots in grid
    return flat_list

def plot_image_grid(
    *items, 
    columns: Optional[int] = None,
    title: Union[List[Optional[str]], str, bool, None] = 'auto',
    titles: Union[List[Optional[str]], str, bool, None] = 'auto',
    show_axis: bool = True, 
    width: float = 16.0, 
    dpi: int = 150, 
    cmap: str = 'gray', 
    supertitle: Optional[str] = None, 
    font_size: str = 'medium', 
    use_microns: Union[str, bool] = 'auto',
    uniform_size: bool = False,
    debug: bool = False
):
    """
    Renders a proportional 2D grid of image channels or raw arrays.
    
    Items can be passed individually as positional arguments, or as a list.
    Automatically distributes the figure width proportionally based on the true 
    physical width (or pixel width) of each image across the specified columns.
    
    Args:
        *items: ImageChannels, raw image arrays, or None (positional or list).
            Passing `None` will leave an empty spot in the grid.
        columns (int, optional): Number of columns to wrap the grid. Defaults to 1 row.
        title (list, str, bool, None, optional): Alias for `titles`.
        titles (list, str, bool, None, optional): Title behavior. 
            - 'auto' (default): Generates titles with name and dimensions.
            - None or False: Suppresses all individual image titles.
            - ["Title 1", None]: Applies "Title 1" to the first image, leaves second blank.
        show_axis (bool): Toggle all axis rendering (ticks and labels).
        width (float): Total figure width in inches.
        dpi (int): Figure DPI.
        cmap (str): Colormap for single-channel (2D) arrays.
        supertitle (str, optional): Overarching figure title.
        font_size (str): Base font size ('small', 'medium', 'large', 'xl', or 'xxl').
        use_microns (str or bool): 
            - 'auto': Microns for channels, pixels for arrays. If mixed, arrays are blank.
            - True: Force microns everywhere.
            - False: Force pixels everywhere (ignores spatial metadata).
        debug (bool): Print layout diagnostic info.
        
    Returns:
        tuple: (matplotlib.figure.Figure, list of matplotlib.axes.Axes)
    """
    font_scales = {
        'small':  {'super': 10, 'title': 8,  'axis': 7,  'tick': 7},
        'medium': {'super': 14, 'title': 11, 'axis': 9,  'tick': 8},
        'large':  {'super': 16, 'title': 12, 'axis': 10, 'tick': 9},
        'xl':     {'super': 20, 'title': 16, 'axis': 13, 'tick': 12},
        'xxl':    {'super': 26, 'title': 22, 'axis': 18, 'tick': 16},
    }
    fs = font_scales.get(font_size, font_scales['medium'])

    items = _normalize_plot_items(*items)
    
    if len(items) == 0 or all(item is None for item in items):
        if debug:
            print("No valid items provided to plot_image_grid.")
        fig, ax = plt.subplots(1, 1, figsize=(width, width))
        ax.axis('off')
        return fig, [ax]

    n_items_raw = len(items)
    n_cols = columns if columns and columns > 0 else n_items_raw
    n_rows = (n_items_raw + n_cols - 1) // n_cols  # integer ceiling division
    
    # Pad items with None to complete the grid
    items = items + [None] * (n_rows * n_cols - n_items_raw)

    valid_items = [item for item in items if item is not None]

    # Unwrap PlotObjects to detect image-type mixture for 'auto' axis scaling
    unwrapped    = [item.data if isinstance(item, PlotObject) else item for item in valid_items]
    has_channels = any(hasattr(i, 'spatial_meta') for i in unwrapped)
    has_raw      = any(isinstance(i, (np.ndarray, da.Array)) for i in unwrapped)

    if debug:
        print(f"--- Plotting Grid Diagnostic ---")
        print(f"Total Spots: {len(items)} ({n_rows} rows x {n_cols} cols) | Has Channels: {has_channels} | Has Raw: {has_raw}")
        print(f"Scale Mode: {use_microns}")

    plot_data = []
    
    # 1. Standardize and Extract Data
    for i, item in enumerate(items):
        if item is None:
            plot_data.append({
                'is_empty': True,
                'width_metric': 0.0,
                'height_metric': 0.0,
                'item_mode': 'blank'
            })
            continue

        # Tuple of PlotObjects → overlay on same axes (first item sets panel size)
        if isinstance(item, tuple) and item and isinstance(item[0], PlotObject):
            primary = item[0]
            pdat = primary.data
            if (primary.plot_type == 'scatter' and hasattr(pdat, 'obs')
                    and primary.x_col in pdat.obs.columns and primary.y_col in pdat.obs.columns):
                w = float(pdat.obs[primary.x_col].max() - pdat.obs[primary.x_col].min())
                h = float(pdat.obs[primary.y_col].max() - pdat.obs[primary.y_col].min())
            elif primary.plot_type == 'umap':
                w, h = 1.0, 1.0
            elif primary.plot_type == 'histogram':
                w, h = 1.6, 1.0
            else:
                w, h = 1.0, 1.0
            plot_data.append({
                'is_empty':        False,
                'arr':             None,
                'plot_obj':        primary,
                'overlay_objects': list(item[1:]),
                'width_metric':    w,
                'height_metric':   h,
                'item_mode':       primary.plot_type,
                'name':            primary.title or f'Plot {i}',
                'extent':          (0, w, h, 0),
                'xlabel':          primary.xlabel or '',
                'ylabel':          primary.ylabel or '',
                'show_ticks':      True,
            })
            continue

        # Non-image PlotObject: compute panel size and defer all rendering
        if isinstance(item, PlotObject) and item.plot_type not in ('auto', 'image'):
            po   = item
            pdat = po.data
            if (po.plot_type == 'scatter' and hasattr(pdat, 'obs')
                    and po.x_col in pdat.obs.columns and po.y_col in pdat.obs.columns):
                w = float(pdat.obs[po.x_col].max() - pdat.obs[po.x_col].min())
                h = float(pdat.obs[po.y_col].max() - pdat.obs[po.y_col].min())
            elif po.plot_type == 'umap':
                w, h = 1.0, 1.0
            elif po.plot_type == 'histogram':
                w, h = 1.6, 1.0
            else:
                w, h = 1.0, 1.0
            plot_data.append({
                'is_empty':      False,
                'arr':           None,
                'plot_obj':      po,
                'width_metric':  w,
                'height_metric': h,
                'item_mode':     po.plot_type,
                'name':          po.title or f'Plot {i}',
                'extent':        (0, w, h, 0),
                'xlabel':        po.xlabel or '',
                'ylabel':        po.ylabel or '',
                'show_ticks':    True,
            })
            continue

        # Image PlotObject: unwrap to underlying data, retain wrapper for title/annotations
        po_wrapper = None
        if isinstance(item, PlotObject):
            po_wrapper = item
            item = item.data

        is_chan = hasattr(item, 'spatial_meta')

        # Extract underlying array and name
        if is_chan:
            img_arr = item.image.compute() if isinstance(item.image, da.Array) else item.image
            name = item.name
        else:
            img_arr = item.compute() if isinstance(item, da.Array) else item
            name = f"Image {i}"

        # Ensure numpy array for plotting
        img_arr = np.array(img_arr)
        h, w = img_arr.shape[:2]

        # Determine if this specific item should use microns, pixels, or blank axes
        if use_microns is True:
            item_mode = 'microns' if is_chan else 'blank'
        elif use_microns is False:
            item_mode = 'pixels'
        else:  # 'auto'
            if has_channels and has_raw:
                item_mode = 'microns' if is_chan else 'blank'
            elif has_channels:
                item_mode = 'microns'
            else:
                item_mode = 'pixels'

        # Set extents and labels based on the mode
        if item_mode == 'microns' and is_chan:
            T = item.spatial_meta.transform_matrix  # type: ignore

            # Compute fov as vector distances between edge midpoints.
            # Using midpoints (not corners) makes this rotation-invariant — opposite
            # corners of a rotated rectangle can have identical X or Y coordinates,
            # which collapses the imshow extent and causes extreme stretching.
            left_mid  = (T @ np.array([0,   h/2, 1.0]))[:2]
            right_mid = (T @ np.array([w,   h/2, 1.0]))[:2]
            top_mid   = (T @ np.array([w/2, 0,   1.0]))[:2]
            bot_mid   = (T @ np.array([w/2, h,   1.0]))[:2]
            ctr       = (T @ np.array([w/2, h/2, 1.0]))[:2]

            fov_x = float(np.linalg.norm(right_mid - left_mid))
            fov_y = float(np.linalg.norm(bot_mid   - top_mid))
            cx, cy = float(ctr[0]), float(ctr[1])

            # Rotation detection: if moving right across the image has a meaningful
            # y-component in world space, the view is rotated. Threshold ~0.5°.
            is_rotated = abs(right_mid[1] - left_mid[1]) / fov_x > 0.01

            if is_rotated:
                # Absolute coordinates are meaningless on a rotated canvas — axis X
                # is no longer world X. Show relative microns from 0 instead.
                ext = (0, fov_x, fov_y, 0)
                xlabel, ylabel = "Relative Microns (X)", "Relative Microns (Y)"
            else:
                # Flip direction from signed component of the edge vector.
                # Falls back gracefully to normal orientation near 90° where the
                # component is near zero and direction is genuinely ambiguous.
                x_sign = -1 if (right_mid[0] - left_mid[0]) < 0 else 1
                y_sign = -1 if (bot_mid[1]   - top_mid[1])  < 0 else 1

                left   = cx - x_sign * fov_x / 2
                right  = cx + x_sign * fov_x / 2
                top    = cy - y_sign * fov_y / 2
                bottom = cy + y_sign * fov_y / 2
                ext = (left, right, bottom, top)
                xlabel, ylabel = "Microns (X)", "Microns (Y)"
            show_ticks = True
        elif item_mode == 'pixels':
            ext = (0, w, h, 0)              # Origin upper-left means y goes H -> 0
            xlabel, ylabel = "Pixels (X)", "Pixels (Y)"
            show_ticks = True
        else:  # 'blank'
            ext = (0, w, h, 0)
            xlabel, ylabel = "", ""
            show_ticks = False

        plot_data.append({
            'is_empty':      False,
            'arr':           img_arr,
            'extent':        ext,
            'name':          name,
            'width_metric':  abs(ext[1] - ext[0]),
            'height_metric': abs(ext[3] - ext[2]),
            'xlabel':        xlabel,
            'ylabel':        ylabel,
            'show_ticks':    show_ticks,
            'item_mode':     item_mode,
            'plot_obj':      po_wrapper,
        })

    # 2. Calculate Layout Dimensions per column and row
    if uniform_size:
        col_widths  = [1.0] * n_cols
        row_heights = [1.0] * n_rows
    else:
        col_widths = []
        for c in range(n_cols):
            widths = [plot_data[r * n_cols + c]['width_metric'] for r in range(n_rows)]
            max_w = max(widths)
            col_widths.append(max_w if max_w > 0 else 1.0)

        row_heights = []
        for r in range(n_rows):
            heights = [plot_data[r * n_cols + c]['height_metric'] for c in range(n_cols)]
            max_h = max(heights)
            row_heights.append(max_h if max_h > 0 else 1.0)
        
    total_width = sum(col_widths)
    total_height = sum(row_heights)
    
    # Scale figure height proportionally to total width
    fig_height = width * (total_height / total_width) if total_width > 0 else width

    # 3. Create Grid
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(width, fig_height), 
        dpi=dpi,
        gridspec_kw={'width_ratios': col_widths, 'height_ratios': row_heights}
    )

    axes_flat = np.atleast_1d(axes).flatten()

    # Resolve titles parameter alias
    resolved_titles = titles if titles != 'auto' else (title if title != 'auto' else 'auto')

    # 4. Plot each item
    for i, (ax, data) in enumerate(zip(axes_flat, plot_data)):
        if data['is_empty']:
            ax.axis('off')
            continue

        # Non-image PlotObject — fully delegated to _render_plot_object
        if data.get('plot_obj') is not None and data['arr'] is None:
            _render_plot_object(ax, data['plot_obj'], fs)
            for extra_po in data.get('overlay_objects', []):
                _render_plot_object(ax, extra_po, fs)
            continue

        img_arr = data['arr']
        left, right, bottom, top = data['extent']

        # Fix array datatypes for smooth imshow rendering
        if img_arr.ndim == 3 and img_arr.shape[-1] in (3, 4) and img_arr.dtype == np.float32:
            img_arr = np.clip(img_arr, 0, 1)
        elif img_arr.dtype == np.uint16:
            img_arr = (img_arr / 256.0).astype(np.uint8)  # Downcast visually
        elif img_arr.dtype == np.float64:
            img_arr = img_arr.astype(np.float32)

        # Plot with explicit mapping
        ax.imshow(img_arr, extent=[left, right, bottom, top], cmap=cmap, origin='upper')

        # Lock axes to physical/pixel footprint. Note: top is smaller than bottom in image coords.
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top) 

        # Format Titles based on resolved_titles preference
        title_text = ""
        if resolved_titles is None or resolved_titles is False:
            title_text = ""
        elif resolved_titles == 'auto':
            title_text = f"{data['name']}\n({img_arr.shape[1]}x{img_arr.shape[0]} px)"
        else:
            # Handle explicit list or single string
            res_list = [resolved_titles] if isinstance(resolved_titles, str) else resolved_titles
            if i < len(res_list):      #type: ignore
                t_val = res_list[i]    #type: ignore
                if t_val is not None:
                    title_text = str(t_val)
            else:
                # If they passed a list but ran out of items, fallback to default
                title_text = f"{data['name']}\n({img_arr.shape[1]}x{img_arr.shape[0]} px)"

        # Image PlotObject: ROI polygon overlays and title override
        if data.get('plot_obj') is not None:
            po = data['plot_obj']
            if po.rois:
                for roi_name, coords in po.rois.items():
                    xs = [pt[0] for pt in coords] + [coords[0][0]]
                    ys = [pt[1] for pt in coords] + [coords[0][1]]
                    c  = (po.roi_colors or {}).get(roi_name, None)
                    ax.plot(xs, ys, lw=1.5, zorder=5, label=roi_name,
                            **({'color': c} if c else {}))
                if po.legend:
                    if po.legend_loc == 'outside right':
                        ax.legend(fontsize=fs['tick'],
                                  bbox_to_anchor=(1.01, 1), loc='upper left')
                    else:
                        ax.legend(fontsize=fs['tick'], loc=po.legend_loc)

        if data.get('plot_obj') is not None and data['plot_obj'].title:
            title_text = data['plot_obj'].title

        if title_text:
            ax.set_title(title_text, fontsize=fs['title'], fontweight='bold')

        # Format Axes
        if not show_axis:
            ax.axis('off')
        else:
            if data['show_ticks']:
                ax.set_xlabel(data['xlabel'], fontsize=fs['axis'])
                ax.tick_params(axis='both', labelsize=fs['tick'])
                
                # Check previous item in the row to see if we need a Y-label
                is_first_in_row = (i % n_cols == 0)
                prev_mode = plot_data[i-1]['item_mode'] if i > 0 and not plot_data[i-1]['is_empty'] else None
                
                # Only show Y label on the far left, OR if the unit mode just changed
                if is_first_in_row or prev_mode != data['item_mode']:
                    ax.set_ylabel(data['ylabel'], fontsize=fs['axis'])
            else:
                # 'Blank' mode: Remove ticks but keep the spine for visual bounds
                ax.set_xticks([])
                ax.set_yticks([])

    if supertitle:
        fig.suptitle(supertitle, fontsize=fs['super'], fontweight='bold')

    plt.tight_layout()
    return fig, axes