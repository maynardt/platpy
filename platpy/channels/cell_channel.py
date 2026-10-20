import copy
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from .base_channel import BaseChannel


@dataclass
class CellChannel(BaseChannel):
    """
    Single-cell data channel — holds a cells×genes AnnData and provides
    two render modes that return ImageChannels registerable in ViewPort:

      render_to_canvas(canvas, color_col, ...)
          Categorical identity plot: each cell drawn as a filled circle
          coloured by an obs column (cluster label, cell type, etc.).

      to_expression_image(gene_name, canvas, ...)
          Continuous expression plot: each cell coloured by its normalised
          expression level for a single gene.

    Both methods take a canvas ImageChannel that sets the spatial frame of
    reference — the returned ImageChannel has the same spatial_meta, so it
    layers directly into ViewPort alongside DAPI / NeuN renders.
    """
    data:       Optional[Any] = None   # cells×genes AnnData; obs needs x_micron, y_micron
    boundaries: Any           = None   # polygon boundaries (optional; from mask_to_roi_channel)
    summary:    Any           = None   # per-cell DataFrame (index=cell_id) derived from boundaries:
                                        # true polygon centroid (x, y), area_um2, n_vertices, label_id.
                                        # Not a second source of truth -- always recomputable from
                                        # `boundaries`; kept alongside it so callers get an O(1)
                                        # per-cell lookup instead of a groupby every time.

    def __getattribute__(self, name):
        # .data and .boundaries can be a lazy_value.Deferred (load_sample()
        # defers both -- see its cells-loading section) rather than the real
        # AnnData/DataFrame, so that opening a bundle doesn't pay for a
        # per-vertex polygon table or a cached cell x gene matrix nobody
        # asked for. Resolving (and caching the result in place) here, on
        # *any* access to these two attributes, makes that transparent --
        # code reading cell_chan.data/.boundaries directly (every existing
        # caller, including anything outside platpy itself) gets the real
        # object automatically, not a wrapper it has to know to unwrap.
        value = object.__getattribute__(self, name)
        if name in ('data', 'boundaries'):
            from ..lazy_value import Deferred
            if isinstance(value, Deferred):
                value = value.resolve()
                object.__setattr__(self, name, value)
        return value

    # ------------------------------------------------------------------
    # Identity plot
    # ------------------------------------------------------------------

    def render_to_canvas(self, canvas, color_col: str,
                         palette=None, dot_radius_um: float = 3.0,
                         alpha: float = 0.8):
        """
        Render cells as filled circles coloured by obs[color_col].

        Parameters
        ----------
        canvas : ImageChannel
            Sets the spatial frame and output resolution.  Cells outside
            the canvas extent are silently clipped.
        color_col : str
            obs column name.  Must be categorical or string-valued.
        palette : dict, optional
            {category: color} mapping.  Any matplotlib color string or hex
            accepted.  Falls back to adata.uns['{color_col}_colors'] if
            present (set by scanpy), then tab20.
        dot_radius_um : float
            Cell dot radius in microns.  Scales to pixels via canvas.mpp.
        alpha : float
            Opacity of the dot layer (0–1).

        Returns
        -------
        ImageChannel  (RGBA, same spatial_meta as canvas)
        """
        from .image_channel import ImageChannel
        from ..transform_utils import microns_to_pixels

        coords_um  = self.data.obs[['x_micron', 'y_micron']].values
        color_vals = self.data.obs[color_col]
        cats = (color_vals.cat.categories.tolist()
                if hasattr(color_vals, 'cat')
                else sorted(color_vals.dropna().unique().tolist()))

        # Resolve palette
        if palette is None:
            uns_key = f'{color_col}_colors'
            if uns_key in getattr(self.data, 'uns', {}):
                palette = dict(zip(
                    self.data.obs[color_col].cat.categories.tolist(),
                    self.data.uns[uns_key]
                ))
        if palette is None:
            raw = plt.cm.tab20(np.linspace(0, 1, len(cats)))
            palette = {str(c): tuple(raw[k]) for k, c in enumerate(cats)}

        h, w       = canvas.image.shape[:2]
        T          = canvas.spatial_meta.transform_matrix
        radius_px  = max(1, int(round(dot_radius_um / canvas.mpp)))
        alpha_u8   = int(np.clip(alpha, 0, 1) * 255)
        overlay    = np.zeros((h, w, 4), dtype=np.uint8)

        # Convert all coords at once, then split per category
        px_all = microns_to_pixels(T, coords_um)          # (N, 2) float

        for cat in cats:
            mask     = (color_vals == cat).values
            rgb_f    = mcolors.to_rgb(palette.get(str(cat), 'gray'))
            bgr_u8   = (int(rgb_f[2]*255), int(rgb_f[1]*255), int(rgb_f[0]*255))
            color_cv = (*bgr_u8, alpha_u8)

            px_cat = px_all[mask].astype(int)
            in_bounds = ((px_cat[:, 0] >= 0) & (px_cat[:, 0] < w) &
                         (px_cat[:, 1] >= 0) & (px_cat[:, 1] < h))
            for cx, cy in px_cat[in_bounds]:
                cv2.circle(overlay, (cx, cy), radius_px, color_cv, -1)

        # cv2 uses BGRA internally — swap to RGBA before wrapping
        overlay_rgba         = overlay.copy()
        overlay_rgba[..., 0] = overlay[..., 2]
        overlay_rgba[..., 2] = overlay[..., 0]

        return ImageChannel(
            name=f'{self.name}_{color_col}',
            image=overlay_rgba,
            spatial_meta=copy.deepcopy(canvas.spatial_meta),
        )

    # ------------------------------------------------------------------
    # Expression plot
    # ------------------------------------------------------------------

    def to_expression_image(self, gene_name: str, canvas,
                             cmap: str = 'inferno',
                             dot_radius_um: float = 3.0,
                             alpha: float = 0.8):
        """
        Render cells as filled circles coloured by expression of gene_name.

        Assumes data.X holds log1p-normalised counts (standard scanpy
        preprocessing).  Colours scale from 0 to the 99th-percentile of
        non-zero expression.

        Parameters
        ----------
        gene_name : str
            Must be in data.var.index.
        canvas : ImageChannel
            Sets the spatial frame and output resolution.
        cmap : str
            Any matplotlib colormap name.
        dot_radius_um, alpha : float
            Same meaning as render_to_canvas.

        Returns
        -------
        ImageChannel  (RGBA, same spatial_meta as canvas)
        """
        from .image_channel import ImageChannel
        from ..transform_utils import microns_to_pixels

        gene_idx = self.data.var.index.get_loc(gene_name)
        expr     = np.asarray(self.data.X[:, gene_idx]).ravel().astype(float)

        nonzero = expr[expr > 0]
        vmax    = float(np.percentile(nonzero, 99)) if len(nonzero) else 1.0
        expr_n  = np.clip(expr / vmax, 0.0, 1.0)

        cmap_fn     = plt.cm.get_cmap(cmap)
        colors_rgba = (cmap_fn(expr_n) * 255).astype(np.uint8)   # (N, 4) RGBA

        coords_um = self.data.obs[['x_micron', 'y_micron']].values
        h, w      = canvas.image.shape[:2]
        T         = canvas.spatial_meta.transform_matrix
        radius_px = max(1, int(round(dot_radius_um / canvas.mpp)))
        alpha_u8  = int(np.clip(alpha, 0, 1) * 255)
        overlay   = np.zeros((h, w, 4), dtype=np.uint8)

        px_all = microns_to_pixels(T, coords_um).astype(int)
        in_bounds = ((px_all[:, 0] >= 0) & (px_all[:, 0] < w) &
                     (px_all[:, 1] >= 0) & (px_all[:, 1] < h))

        for i in np.where(in_bounds)[0]:
            cx, cy   = px_all[i]
            r, g, b  = int(colors_rgba[i, 0]), int(colors_rgba[i, 1]), int(colors_rgba[i, 2])
            cv2.circle(overlay, (cx, cy), radius_px, (b, g, r, alpha_u8), -1)

        overlay_rgba         = overlay.copy()
        overlay_rgba[..., 0] = overlay[..., 2]
        overlay_rgba[..., 2] = overlay[..., 0]

        return ImageChannel(
            name=f'{self.name}_{gene_name}',
            image=overlay_rgba,
            spatial_meta=copy.deepcopy(canvas.spatial_meta),
        )
