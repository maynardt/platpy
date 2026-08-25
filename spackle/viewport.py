from dataclasses import dataclass, field
import numpy as np
import cv2
from typing import List, Dict, Tuple, Union, Optional
import copy

# Spackle imports
from .channels.image_channel import ImageChannel
from .channels.omic_channel import OmicChannel
from .channels.roi_channel import ROIChannel
from .transform_utils import SpatialMetadata, create_transform_matrix, pixels_to_microns, microns_to_pixels


def _draw_dashed_polyline(canvas: np.ndarray, pts: np.ndarray, color: tuple,
                          thickness: int, dash_len: int = 15, gap_len: int = 8,
                          closed: bool = True):
    """Draw a dashed polygon or open path onto a uint8 RGBA canvas."""
    n = len(pts)
    end = n if closed else n - 1
    for i in range(end):
        p1 = pts[i].astype(float)
        p2 = pts[(i + 1) % n].astype(float)
        seg_len = np.linalg.norm(p2 - p1)
        if seg_len < 1:
            continue
        direction = (p2 - p1) / seg_len
        pos, drawing = 0.0, True
        while pos < seg_len:
            next_pos = pos + (dash_len if drawing else gap_len)
            if drawing:
                start = p1 + direction * pos
                end   = p1 + direction * min(next_pos, seg_len)
                cv2.line(canvas, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color, thickness)
            pos, drawing = next_pos, not drawing


@dataclass
class ViewPort:
    """
    A specialized engine for compositing spatial data into a fixed-grid array.
    Uses Lazy Evaluation: stacks layers and only crunches the math when .render() is called.

    Physical space is always in absolute microns. The ViewPort stores a camera
    (center_um, fov_um, rotation, flips) and recomputes the affine transform
    matrix whenever the camera is updated, so every render is spatially correct.
    """
    extent: Tuple[float, float, float, float]  # min_x, max_x, min_y, max_y
    shape: Tuple[int, int]                     # height, width in pixels
    spatial_meta: SpatialMetadata
    layers: List[Dict] = field(default_factory=list)

    # --- Camera State ---
    center_um: Optional[Tuple[float, float]] = None
    fov_um: Optional[Tuple[float, float]] = None
    rotation_deg: float = 0.0
    flip_x: bool = False
    flip_y: bool = False
    background_color: str = 'black'

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def mpp(self) -> float:
        """
        Microns per pixel of this viewport.
        abs() handles flipped viewports where the diagonal goes negative.
        """
        return float(abs(self.spatial_meta.transform_matrix[0, 0]))

    @property
    def target_mpp(self) -> float:
        """Resolution of the current camera FOV in microns per pixel."""
        if self.fov_um is None:
            return self.mpp
        return self.fov_um[0] / self.shape[1]

    @property
    def view_extent(self) -> Tuple[float, float, float, float]:
        """Returns (min_x, max_x, min_y, max_y) of the current camera FOV in microns."""
        if self.center_um is None or self.fov_um is None:
            return self.extent
        cx, cy = self.center_um
        fx, fy = self.fov_um
        return (cx - fx / 2, cx + fx / 2, cy - fy / 2, cy + fy / 2)

    # =========================================================================
    # CONSTRUCTORS
    # =========================================================================

    @classmethod
    def create(
        cls,
        target_width_px: int,
        center_um: Tuple[float, float],
        fov_um: Tuple[float, float],
        flip_x: bool = False,
        flip_y: bool = False,
        rotation_deg: float = 0.0,
        background_color: str = 'black'
    ) -> 'ViewPort':
        """
        Creates a ViewPort anchored in absolute micron space from scratch.
        Use this when you know where you want to look before you have an image.
        """
        from .render_utils import create_new_image

        cx, cy = center_um
        fx, fy = fov_um
        min_x, max_x = cx - (fx / 2.0), cx + (fx / 2.0)
        min_y, max_y = cy - (fy / 2.0), cy + (fy / 2.0)
        mpp_x = fx / target_width_px

        master_canvas = create_new_image(
            size=(min_x, max_x, min_y, max_y),
            resolution=mpp_x,
            background_color=background_color,
            name="viewport_canvas"
        )

        vp = cls.from_image(master_canvas)
        vp.background_color = background_color
        vp.update_view(
            center_um=center_um,
            fov_um=fov_um,
            rotation_deg=rotation_deg,
            flip_x=flip_x,
            flip_y=flip_y
        )
        return vp

    @classmethod
    def from_image(cls, reference_image: ImageChannel) -> 'ViewPort':
        """
        Creates a ViewPort perfectly matched to an existing ImageChannel's canvas.
        Deep-copies spatial_meta so camera updates don't mutate the source image.
        """
        return cls(
            extent=reference_image.spatial_meta.extent,
            shape=reference_image.image.shape[:2],  # type: ignore
            spatial_meta=copy.deepcopy(reference_image.spatial_meta)
        )

    @classmethod
    def match_view(cls, source: Union['ViewPort', 'ImageChannel']) -> 'ViewPort':
        """
        Creates a new empty ViewPort with the same camera settings as source.
        Accepts either a ViewPort (copies all camera state) or an ImageChannel
        (matches its spatial bounds and pixel dimensions).
        No layers are pre-loaded — add whatever you need after calling this.

        Typical use: make two side-by-side views of the same region, or add
        new layers on top of an already-rendered composite:

            omic_view = ViewPort.match_view(view)
            omic_view.add_layer(previous_render)
            omic_view.add_layer(sample.omics['spots']).adjust_omic(gene_name='Snap25')
        """
        if isinstance(source, ViewPort):
            return cls(
                extent=source.extent,
                shape=source.shape,
                spatial_meta=copy.deepcopy(source.spatial_meta),
                center_um=source.center_um,
                fov_um=source.fov_um,
                rotation_deg=source.rotation_deg,
                flip_x=source.flip_x,
                flip_y=source.flip_y,
                background_color=source.background_color
            )
        else:
            # ImageChannel: match its spatial bounds and pixel dimensions
            return cls.from_image(source)
        
    # =========================================================================
    # CAMERA CONTROLS
    # =========================================================================

    def update_view(
        self,
        center_um: Optional[Tuple[float, float]] = None,
        fov_um: Optional[Tuple[float, float]] = None,
        rotation_deg: Optional[float] = None,
        flip_x: Optional[bool] = None,
        flip_y: Optional[bool] = None,
        background_color: Optional[str] = None
    ) -> 'ViewPort':
        """
        Adjusts the camera window in absolute micron space and recomputes the
        affine transform matrix. Returns self so calls can be chained.
        """
        if center_um is not None:        self.center_um = center_um
        if fov_um is not None:           self.fov_um = fov_um
        if rotation_deg is not None:     self.rotation_deg = float(rotation_deg)
        if flip_x is not None:           self.flip_x = bool(flip_x)
        if flip_y is not None:           self.flip_y = bool(flip_y)
        if background_color is not None: self.background_color = background_color

        if self.center_um is not None and self.fov_um is not None:
            _, W = self.shape  # width stays constant; height tracks FOV aspect ratio
            cx_um, cy_um = self.center_um
            fx, fy = self.fov_um
            H = max(1, int(round(W * fy / fx)))
            self.shape = (H, W)

            # Proper camera matrix: rotate/flip around the viewport center, not pixel (0,0).
            # T1: center pixels -> S: scale to microns -> F: flip -> R: rotate -> T2: translate to micron center
            T1 = np.array([[1, 0, -W/2], [0, 1, -H/2], [0, 0, 1]], dtype=np.float64)
            S  = np.array([[fx/W, 0, 0], [0, fy/H, 0], [0, 0, 1]], dtype=np.float64)

            F = np.eye(3, dtype=np.float64)
            if self.flip_x: F[0, 0] = -1.0
            if self.flip_y: F[1, 1] = -1.0

            theta = np.radians(self.rotation_deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            R  = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float64)

            T2 = np.array([[1, 0, cx_um], [0, 1, cy_um], [0, 0, 1]], dtype=np.float64)

            self.spatial_meta.transform_matrix = T2 @ R @ F @ S @ T1

            # Extent = AABB of the rotated viewport corners in micron space
            corners_px = np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float64)
            ones = np.ones((4, 1))
            M = self.spatial_meta.transform_matrix
            corners_um = (M @ np.hstack([corners_px, ones]).T).T[:, :2]
            self.extent = (
                float(corners_um[:, 0].min()), float(corners_um[:, 0].max()),
                float(corners_um[:, 1].min()), float(corners_um[:, 1].max())
            )

        return self

    def clone(self, keep_layers: bool = True) -> 'ViewPort':
        """Creates a duplicate of this ViewPort with independent layer configs but shared channel data.

        Each layer dict is copied independently so visibility/display changes in the clone
        don't affect the original. The channel objects themselves are shared (not copied),
        so no image data is duplicated.
        """
        new_layers = []
        if keep_layers:
            for layer in self.layers:
                new_layer = layer.copy()                        # independent top-level dict
                new_layer['display'] = layer['display'].copy()  # independent display settings
                new_layers.append(new_layer)
        return ViewPort(
            extent=self.extent,
            shape=self.shape,
            spatial_meta=copy.deepcopy(self.spatial_meta),
            center_um=self.center_um,
            fov_um=self.fov_um,
            rotation_deg=self.rotation_deg,
            flip_x=self.flip_x,
            flip_y=self.flip_y,
            background_color=self.background_color,
            layers=new_layers
        )

    # =========================================================================
    # LAYER MANAGEMENT
    # =========================================================================

    def add_layer(self, channel: Union[ImageChannel, OmicChannel, ROIChannel],
                  name: str = None, overwrite: bool = False) -> 'ViewPort':
        """
        Queues a channel as a layer. Returns self for chaining.
        Use adjust_layer() / adjust_omic() / adjust_roi() to configure display.
        Any styles pre-set via channel.set_style() are picked up automatically.

        overwrite=False (default): raises ValueError if a layer with the same name
        already exists — duplicate layers are almost always a mistake.
        overwrite=True: replaces the existing layer in-place, preserving stack order.
        """
        layer_name = name if name is not None else channel.name
        defaults   = channel.metadata.get('layer_defaults', {})

        existing_idx = next(
            (i for i, l in enumerate(self.layers) if l.get('name') == layer_name), None
        )

        if existing_idx is not None:
            if not overwrite:
                raise ValueError(
                    f"A layer named '{layer_name}' already exists. "
                    "Pass overwrite=True to replace it, or call remove_layer() first."
                )
            self.layers[existing_idx] = {
                'channel': channel,
                'name': layer_name,
                'blend_mode': 'normal',
                'visible': True,
                'display': defaults.copy()
            }
            return self

        self.layers.append({
            'channel': channel,
            'name': layer_name,
            'blend_mode': 'normal',
            'visible': True,
            'display': defaults.copy()
        })
        return self

    def adjust_layer(self, name: str = None, blend_mode: str = None, visible: bool = None, **display_kwargs) -> 'ViewPort':
        """
        Configures display settings for a layer. Returns self for chaining.
        name=None targets the last-added layer, enabling method chaining after add_layer().
        Re-calling merges new settings without clearing existing ones.

        Image options:  tint, percentile_clip, gamma
        Omic options:   gene_name, cmap_name, alpha, alpha_gamma, alpha_pass,
                        percentile_clip, gamma, background_color
        Both:           blend_mode
        """
        if not self.layers:
            raise ValueError("No layers to adjust — call add_layer() first.")

        if name is None:
            layer = self.layers[-1]
        else:
            matches = [l for l in self.layers if l.get('name') == name]
            if not matches:
                available = [l.get('name') for l in self.layers]
                raise ValueError(f"No layer named '{name}'. Available: {available}")
            layer = matches[-1]

        if visible is not None:
            layer['visible'] = visible
        if blend_mode is not None:
            layer['blend_mode'] = blend_mode
        layer['display'].update(display_kwargs)
        return self

    def adjust_omic(self, name: str = None, blend_mode: str = None,
                    omic_percentile_clip: tuple = None, omic_alpha: float = None,
                    **omic_kwargs) -> 'ViewPort':
        """
        Configures display settings for an OmicChannel layer. Returns self for chaining.
        name=None targets the last-added layer.

        Options: gene_name, cmap_name, gamma, omic_percentile_clip, omic_alpha,
                 alpha_gamma, alpha_pass, background_color, blend_mode
        """
        import warnings

        if not self.layers:
            raise ValueError("No layers to adjust — call add_layer() first.")

        if name is None:
            layer = self.layers[-1]
        else:
            matches = [l for l in self.layers if l.get('name') == name]
            if not matches:
                available = [l.get('name') for l in self.layers]
                raise ValueError(f"No layer named '{name}'. Available: {available}")
            layer = matches[-1]

        if not isinstance(layer['channel'], OmicChannel):
            warnings.warn(
                f"adjust_omic() called on layer '{layer['name']}' which is not an OmicChannel "
                "— settings stored but ignored at render time.",
                UserWarning, stacklevel=2
            )

        # Map omic-prefixed names to internal render pipeline names
        if omic_percentile_clip is not None:
            omic_kwargs['percentile_clip'] = omic_percentile_clip
        if omic_alpha is not None:
            omic_kwargs['alpha'] = omic_alpha

        if blend_mode is not None:
            layer['blend_mode'] = blend_mode
        layer['display'].update(omic_kwargs)
        return self

    def adjust_roi(self, name: str = None, blend_mode: str = None, **roi_kwargs) -> 'ViewPort':
        """
        Configures display settings for an ROIChannel layer. Returns self for chaining.
        name=None targets the last-added layer.

        Options: line_color, line_width, fill, fill_color, line_style
        """
        import warnings

        if not self.layers:
            raise ValueError("No layers to adjust — call add_layer() first.")

        if name is None:
            layer = self.layers[-1]
        else:
            matches = [l for l in self.layers if l.get('name') == name]
            if not matches:
                available = [l.get('name') for l in self.layers]
                raise ValueError(f"No layer named '{name}'. Available: {available}")
            layer = matches[-1]

        if not isinstance(layer['channel'], ROIChannel):
            warnings.warn(
                f"adjust_roi() called on layer '{layer['name']}' which is not an ROIChannel "
                "— settings stored but ignored at render time.",
                UserWarning, stacklevel=2
            )

        if blend_mode is not None:
            layer['blend_mode'] = blend_mode
        layer['display'].update(roi_kwargs)
        return self

    def remove_layer(self, name: str) -> 'ViewPort':
        """Removes a layer from the stack by name. Returns self for chaining."""
        before = len(self.layers)
        self.layers = [l for l in self.layers if l.get('name') != name]
        if len(self.layers) == before:
            available = [l.get('name') for l in self.layers]
            raise ValueError(f"No layer named '{name}'. Available: {available}")
        return self

    def layer_to_front(self, name: str) -> 'ViewPort':
        """Moves a layer to the top of the stack (rendered last, appears in front). Returns self."""
        matches = [l for l in self.layers if l.get('name') == name]
        if not matches:
            raise ValueError(f"No layer named '{name}'. Available: {[l.get('name') for l in self.layers]}")
        layer = matches[-1]
        self.layers.remove(layer)
        self.layers.append(layer)
        return self

    def layer_to_back(self, name: str) -> 'ViewPort':
        """Moves a layer to the bottom of the stack (rendered first, appears behind). Returns self."""
        matches = [l for l in self.layers if l.get('name') == name]
        if not matches:
            raise ValueError(f"No layer named '{name}'. Available: {[l.get('name') for l in self.layers]}")
        layer = matches[-1]
        self.layers.remove(layer)
        self.layers.insert(0, layer)
        return self

    def list_layers(self, verbose: bool = False) -> 'ViewPort':
        """Prints the current layer stack in render order (bottom to top).
        verbose=True also shows per-layer display settings."""
        if not self.layers:
            print("No layers.")
            return self
        print(f"Layer stack (bottom → top), {len(self.layers)} layer(s):")
        for i, layer in enumerate(self.layers):
            name      = layer.get('name', '?')
            blend     = layer.get('blend_mode', 'normal')
            vis       = 'visible' if layer.get('visible', True) else 'not_visible'
            chan_type = type(layer['channel']).__name__
            if verbose:
                display = layer.get('display', {})
                disp_str = '  '.join(f"{k}={v}" for k, v in display.items()) or '(no display settings)'
                print(f"  [{i}] '{name}'  ({chan_type})  blend={blend}  {vis}  {disp_str}")
            else:
                print(f"  [{i}] '{name}'  ({chan_type})  blend={blend}  {vis}")
        return self


    # =========================================================================
    # COORDINATE CONVERSION
    # =========================================================================

    def pixels_to_microns(self, px_coords: np.ndarray) -> np.ndarray:
        """Convert (N, 2) viewport pixel coordinates to absolute micron coordinates."""
        return pixels_to_microns(self.spatial_meta.transform_matrix, px_coords)

    def microns_to_pixels(self, um_coords: np.ndarray) -> np.ndarray:
        """Convert (N, 2) absolute micron coordinates to viewport pixel coordinates."""
        return microns_to_pixels(self.spatial_meta.transform_matrix, um_coords)

    def screen_direction_to_angle(self, direction) -> float:
        """
        Convert a screen-space direction to a micron-space angle in degrees.
        Use the result as label_anchor in bins_to_roi_channel() so labels
        appear on the correct side of the image regardless of rotation/flip.

        Parameters
        ----------
        direction : str or float
            'right', 'left', 'top', 'bottom' — sides of the rendered image.
            float — angle in degrees in screen space (0 = right, CCW).

        Returns
        -------
        float — angle in micron space (degrees), ready for label_anchor.

        Example
        -------
        shells_chan = bins_to_roi_channel(
            shell_outlines, ...,
            label_anchor=vp_sh.screen_direction_to_angle('right')
        )
        """
        _screen_dirs = {
            'right':  np.array([ 1.0,  0.0]),
            'left':   np.array([-1.0,  0.0]),
            'top':    np.array([ 0.0, -1.0]),  # y=0 is top in image/screen space
            'bottom': np.array([ 0.0,  1.0]),
        }
        if isinstance(direction, str):
            if direction not in _screen_dirs:
                raise ValueError(f"direction must be one of {list(_screen_dirs.keys())} or a float angle")
            d_screen = _screen_dirs[direction]
        else:
            theta_s = np.radians(float(direction))
            d_screen = np.array([np.cos(theta_s), np.sin(theta_s)])

        # Direction-only part of the pixel→micron transform is R @ F.
        F = np.diag([-1.0 if self.flip_x else 1.0,
                     -1.0 if self.flip_y else 1.0])
        t = np.radians(self.rotation_deg)
        R = np.array([[np.cos(t), -np.sin(t)],
                      [np.sin(t),  np.cos(t)]])
        d_micron = R @ F @ d_screen
        return float(np.degrees(np.arctan2(d_micron[1], d_micron[0])))

    def overview(self, name: str = None, target_width_px: int = 512,
                 show_footprint: bool = True) -> 'ImageChannel':
        """
        Render a named layer (default: topmost visible ImageChannel) axis-aligned at
        a quick resolution. Useful for spatial navigation when the main viewport is
        rotated or zoomed into a small region.

        If show_footprint=True, the current viewport's actual rotated footprint is
        overlaid as a yellow box so you can see exactly where in the tissue you are.

        Args:
            name:             Layer name to use as the overview canvas.
                              Default: topmost visible ImageChannel.
            target_width_px:  Output resolution. Default 512.
            show_footprint:   Overlay the current viewport rectangle. Default True.

        Returns:
            ImageChannel suitable for passing directly to plot_image_grid().

        Example:
            plot_image_grid(vp.overview(), width=4)
        """
        from .channels.roi_channel import ROIChannel

        visible_img = [l for l in self.layers
                       if l.get('visible', True) and isinstance(l['channel'], ImageChannel)]
        if not visible_img:
            raise ValueError("No visible ImageChannel layers to build overview from.")

        if name is not None:
            matches = [l for l in visible_img if l['name'] == name]
            if not matches:
                avail = [l['name'] for l in visible_img]
                raise ValueError(f"No visible image layer named '{name}'. Available: {avail}")
            layer = matches[-1]
        else:
            layer = visible_img[-1]

        chan = layer['channel']
        ext  = chan.spatial_meta.extent          # (min_x, max_x, min_y, max_y)
        cx   = (ext[0] + ext[1]) / 2
        cy   = (ext[2] + ext[3]) / 2
        fx   = ext[1] - ext[0]
        fy   = ext[3] - ext[2]

        nav = ViewPort.create(target_width_px=target_width_px,
                              center_um=(cx, cy), fov_um=(fx, fy))
        nav.add_layer(chan, name=layer['name'])
        nav.layers[-1]['display'].update(layer.get('display', {}))

        if show_footprint and self.shape is not None:
            H, W = self.shape
            T    = self.spatial_meta.transform_matrix
            corners_px = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float64)
            ones       = np.ones((4, 1))
            corners_um = (T @ np.hstack([corners_px, ones]).T).T[:, :2]
            box = ROIChannel(name='viewport_footprint',
                             polygons={'footprint': corners_um})
            box.set_style(line_color='xkcd:bright yellow', line_width=3)
            nav.add_layer(box)

        return nav.render(output_name='overview')

    # =========================================================================
    # RENDERING
    # =========================================================================

    def _ensure_rgba_float(self, img: np.ndarray) -> np.ndarray:
        """Standardizes any image array to float32 RGBA in range [0.0, 1.0].
        For non-uint8 inputs, uses percentile normalization rather than dividing by
        the dtype max — microscopy images use a small fraction of 16-bit range."""
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif np.issubdtype(img.dtype, np.floating):
            img = img.astype(np.float32)
        else:
            # High bit-depth (uint16, uint32, etc.): normalize via percentile clip
            # so actual signal fills the [0,1] range instead of being nearly invisible.
            from .render_utils import normalize_array
            img = normalize_array(img)

        if img.ndim == 2:
            return np.stack([img, img, img, np.ones_like(img)], axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 3:
            return np.concatenate([img, np.ones_like(img[..., :1])], axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 4:
            return img
        raise ValueError(f"Unsupported image shape for RGBA conversion: {img.shape}")

    def render(self, output_name: str = "Viewport_Render", debug: bool = False) -> ImageChannel:
        """
        Executes the render pipeline: for each layer, crop -> warp -> blend.
        Returns a single composited ImageChannel at the viewport's resolution.

        Memory note: ImageChannel layers are pre-cropped to the viewport extent
        before the Dask compute() call. This prevents accidentally loading a full
        40k x 40k source image into RAM when only a small FOV is needed.
        """
        H, W = self.shape
        canvas = np.zeros((H, W, 4), dtype=np.float32)

        for layer in self.layers:
            if not layer.get('visible', True):
                continue
            chan = layer['channel']

            # --- ROI vector overlay ---
            if isinstance(chan, ROIChannel):
                import matplotlib.colors as mcolors
                d = layer.get('display', {})
                line_color  = d.get('line_color', 'white')
                lw          = d.get('line_width', 2)
                do_fill     = d.get('fill', False)
                fill_color  = d.get('fill_color', None)
                line_style  = d.get('line_style', 'solid')

                color_rgb = mcolors.to_rgb(line_color)
                cv_color  = (int(color_rgb[0]*255), int(color_rgb[1]*255), int(color_rgb[2]*255), 255)

                if fill_color is not None:
                    fc_rgb    = mcolors.to_rgb(fill_color)
                    cv_fill   = (int(fc_rgb[0]*255), int(fc_rgb[1]*255), int(fc_rgb[2]*255), 100)
                else:
                    cv_fill   = (cv_color[0], cv_color[1], cv_color[2], 100)  # line color @ ~40% alpha

                ov = np.zeros((H, W, 4), dtype=np.uint8)

                # Closed polygons — per-polygon style overrides fall back to channel defaults
                for poly_name, poly_um in (chan.polygons or {}).items():
                    ps          = (chan.polygon_styles or {}).get(poly_name, {})
                    poly_fill   = ps.get('fill', do_fill)
                    poly_lc_str = ps.get('line_color', line_color)
                    poly_fc_str = ps.get('fill_color', fill_color)

                    p_rgb   = mcolors.to_rgb(poly_lc_str)
                    p_color = (int(p_rgb[0]*255), int(p_rgb[1]*255), int(p_rgb[2]*255), 255)
                    if poly_fc_str is not None:
                        fc_rgb = mcolors.to_rgb(poly_fc_str)
                        p_fill = (int(fc_rgb[0]*255), int(fc_rgb[1]*255), int(fc_rgb[2]*255), 100)
                    else:
                        p_fill = (p_color[0], p_color[1], p_color[2], 100)

                    px  = self.microns_to_pixels(np.array(poly_um)).astype(np.int32)
                    pts = px.reshape((-1, 1, 2))
                    if poly_fill:
                        cv2.fillPoly(ov, [pts], p_fill)
                    if line_style == 'dashed':
                        _draw_dashed_polyline(ov, px, p_color, lw, closed=True)
                    else:
                        cv2.polylines(ov, [pts], isClosed=True, color=p_color, thickness=lw)

                # Open features (axis lines, arrows, etc.)
                for feat_um in (chan.features or {}).values():
                    px  = self.microns_to_pixels(np.array(feat_um)).astype(np.int32)
                    pts = px.reshape((-1, 1, 2))
                    if line_style == 'dashed':
                        _draw_dashed_polyline(ov, px, cv_color, lw, closed=False)
                    else:
                        cv2.polylines(ov, [pts], isClosed=False, color=cv_color, thickness=lw)

                # Text labels — centred on transformed anchor point.
                # cv2.putText on a 4-channel BGRA image does not reliably write the
                # alpha channel, leaving text pixels transparent after compositing.
                # Work around by rendering to a 3-channel temp buffer and manually
                # promoting written pixels to alpha=255 in ov.
                if chan.labels:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # label_size is a relative scale: 1.0 → text ~H/500 px tall,
                    # which stays legible regardless of canvas resolution.
                    font_scale = d.get('label_size', 1.0) * H / 500
                    thickness  = max(1, int(font_scale * d.get('label_weight', 0.12)))
                    lc_rgb     = mcolors.to_rgb(d.get('label_color', 'white'))
                    cv_lc_3    = (int(lc_rgb[0]*255), int(lc_rgb[1]*255), int(lc_rgb[2]*255))
                    # Render in white to get a reliable non-zero mask (black text on a
                    # black buffer passes the > 0 check), then paint with the real color.
                    tmp = np.zeros((H, W, 3), dtype=np.uint8)
                    for text, (x_um, y_um) in chan.labels.items():
                        px_pt = self.microns_to_pixels(np.array([[x_um, y_um]]))[0].astype(int)
                        (tw, th), _ = cv2.getTextSize(str(text), font, font_scale, thickness)
                        cv2.putText(tmp, str(text), (px_pt[0] - tw//2, px_pt[1] + th//2),
                                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_8)
                    mask = tmp[:, :, 0] > 0
                    ov[mask, :3] = cv_lc_3
                    ov[mask, 3]  = 255

                ov_f  = ov.astype(np.float32) / 255.0
                top_a = ov_f[..., 3:4]
                bot_a = canvas[..., 3:4]
                out_a = top_a + bot_a * (1.0 - top_a)
                safe_a = np.where(out_a == 0, 1.0, out_a)
                canvas[..., :3] = (ov_f[..., :3] * top_a + canvas[..., :3] * bot_a * (1.0 - top_a)) / safe_a
                canvas[..., 3:4] = out_a
                continue

            mode = layer['blend_mode']
            display = layer.get('display', {})

            # --- 1. EXTRACT & WARP ---
            if isinstance(chan, OmicChannel):
                from .render_utils import render_omic_to_image

                # Render omics to an axis-aligned proxy canvas covering the viewport's
                # micron extent, then warp to viewport space the same way images are.
                # Passing the rotated viewport directly causes the spot renderer's
                # internal M_warp to compose with the grid rotation in a way that
                # produces a streak for any non-zero viewport rotation.
                ex = self.extent   # AABB of viewport in microns (min_x, max_x, min_y, max_y)
                true_mpp = self.fov_um[0] / self.shape[1] if self.fov_um else abs(self.spatial_meta.transform_matrix[0, 0])
                proxy_w = max(1, int((ex[1] - ex[0]) / true_mpp))
                proxy_h = max(1, int((ex[3] - ex[2]) / true_mpp))
                proxy_meta = SpatialMetadata(
                    transform_matrix=create_transform_matrix(
                        scale_x=true_mpp, scale_y=true_mpp, tx=ex[0], ty=ex[2]
                    ),
                    local_bounds=(0.0, float(proxy_w), 0.0, float(proxy_h)),
                    unit=self.spatial_meta.unit
                )
                proxy = ImageChannel(
                    name="proxy_canvas",
                    image=np.zeros((proxy_h, proxy_w, 4), dtype=np.uint8),
                    spatial_meta=proxy_meta
                )
                rendered = render_omic_to_image(chan, base_image=proxy, debug=debug, **display)  # type: ignore

                # Warp the rendered omic into viewport pixel space (handles rotation/flip).
                src_rgba = self._ensure_rgba_float(rendered.image)                  # type: ignore
                M_warp = np.linalg.inv(self.spatial_meta.transform_matrix) @ rendered.spatial_meta.transform_matrix
                src_mpp  = abs(rendered.spatial_meta.transform_matrix[0, 0])
                dst_mpp  = abs(self.spatial_meta.transform_matrix[0, 0])
                interp   = cv2.INTER_AREA if src_mpp < dst_mpp * 0.9 else cv2.INTER_LINEAR
                layer_rgba = cv2.warpAffine(
                    src_rgba, M_warp[:2, :], (W, H),
                    flags=interp,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0)
                )

            elif isinstance(chan, ImageChannel):
                # Pick the pyramid level that best matches the viewport's render resolution.
                chan = chan.get_optimal_channel(self.target_mpp)
                # Pre-crop to viewport extent BEFORE computing the Dask array.
                # This is the critical memory fix: we only pull the pixels we need.
                try:
                    chan_cropped = chan.crop_to_micron_roi(
                        min_x_um=self.extent[0],
                        max_x_um=self.extent[1],
                        min_y_um=self.extent[2],
                        max_y_um=self.extent[3]
                    )
                except ValueError as e:
                    # No overlap between this layer and the viewport — skip it
                    print(f"ViewPort.render: skipping layer '{chan.name}' — {e}")
                    continue

                # Apply display settings (tint, percentile_clip, gamma) at render time.
                # adjust_display() materializes only the cropped region — memory safe.
                img_display = {k: v for k, v in display.items()
                               if k in ('tint', 'percentile_clip', 'gamma')}
                if img_display:
                    chan_cropped = chan_cropped.adjust_display(**img_display)

                src_arr = chan_cropped.image.compute() if hasattr(chan_cropped.image, 'compute') else chan_cropped.image  # type: ignore
                src_rgba = self._ensure_rgba_float(src_arr)  # type: ignore

                # Warp source pixels → microns → viewport pixels (forward mapping for cv2.warpAffine).
                # Uses chan_cropped.spatial_meta because the crop updates the origin.
                M_warp = np.linalg.inv(self.spatial_meta.transform_matrix) @ chan_cropped.spatial_meta.transform_matrix

                # INTER_AREA averages all contributing source pixels — correct for downsampling.
                # INTER_LINEAR is fine for upsampling or roughly 1:1 scale.
                src_mpp = abs(chan_cropped.spatial_meta.transform_matrix[0, 0])
                dst_mpp = abs(self.spatial_meta.transform_matrix[0, 0])
                interp = cv2.INTER_AREA if src_mpp < dst_mpp * 0.9 else cv2.INTER_LINEAR

                layer_rgba = cv2.warpAffine(
                    src_rgba, M_warp[:2, :], (W, H),
                    flags=interp,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0)
                )

            else:
                print(f"ViewPort.render: unknown layer type {type(chan)}, skipping.")
                continue

            # --- 2. COMPOSITE ---
            top_a = layer_rgba[..., 3:4]
            bot_a = canvas[..., 3:4]

            if mode == 'lighten':
                canvas[..., :3] = np.maximum(canvas[..., :3], layer_rgba[..., :3] * top_a)
                canvas[..., 3:4] = np.maximum(bot_a, top_a)

            elif mode == 'darken':
                canvas[..., :3] = np.minimum(canvas[..., :3], layer_rgba[..., :3] + (1.0 - top_a))
                canvas[..., 3:4] = np.maximum(bot_a, top_a)

            else:  # 'normal' Porter-Duff Over
                out_a = top_a + bot_a * (1.0 - top_a)
                out_a_safe = np.where(out_a == 0, 1.0, out_a)
                canvas[..., :3] = (layer_rgba[..., :3] * top_a + canvas[..., :3] * bot_a * (1.0 - top_a)) / out_a_safe
                canvas[..., 3:4] = out_a

        # --- 3. FINALIZE ---
        # Composite background color into transparent areas (Porter-Duff Over with bg).
        # Transparent pixels (no data) become the background; opaque pixels are unchanged.
        if self.background_color and self.background_color.lower() not in ('transparent', 'none'):
            import matplotlib.colors as mcolors
            bg = np.array(mcolors.to_rgba(self.background_color), dtype=np.float32)
            fg_a = canvas[..., 3:4]
            canvas[..., :3] = canvas[..., :3] * fg_a + bg[:3] * (1.0 - fg_a)
            canvas[..., 3] = 1.0

        final_uint8 = (np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8)
        return ImageChannel(
            name=output_name,
            image=final_uint8,
            spatial_meta=self.spatial_meta
        )
