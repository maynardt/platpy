from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Optional, Tuple, Union
from shapely.geometry import Polygon, Point
from .base_channel import BaseChannel

@dataclass
class ROIChannel(BaseChannel):
    """
    Contains named spatial geometries for ROI definition and analysis.

    polygons:  Closed boundary shapes — ROI outlines, spatial bins, tissue regions.
    features:  Open or auxiliary geometries — axis lines, arrows, labels, crosshairs.
               Rendered as open paths (no fill, no closure). Open bucket for future
               geometry types (lines, annotations, etc.).
    """
    polygons: Dict[str, np.ndarray] = field(default_factory=dict)
    features: Dict[str, np.ndarray] = field(default_factory=dict)
    polygon_styles: Dict[str, dict] = field(default_factory=dict)
    # Per-polygon style overrides keyed by polygon name.
    # Supported keys: 'line_color', 'fill_color', 'fill' (bool).
    # Unset keys fall back to the channel-level style from set_style().
    labels: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Text annotations: {display_text: (x_um, y_um)}.
    # Anchor is centred on the pixel that (x_um, y_um) maps to at render time.

    def _boolean_operation(self, other_roi: 'ROIChannel', operation: str) -> 'ROIChannel':
        """Handles Shapely boolean geometry math on polygons only."""
        self_poly = Polygon()
        for pts in self.polygons.values():
            self_poly = self_poly.union(Polygon(pts))

        other_poly = Polygon()
        for pts in other_roi.polygons.values():
            other_poly = other_poly.union(Polygon(pts))

        if operation == 'union':        result = self_poly.union(other_poly)
        elif operation == 'difference': result = self_poly.difference(other_poly)
        elif operation == 'intersection': result = self_poly.intersection(other_poly)
        else: raise ValueError(f"Unknown operation: {operation}")

        new_polys = {}
        if result.geom_type == 'Polygon' and not result.is_empty:
            new_polys[f"{self.name}_{operation}"] = np.array(result.exterior.coords)   #type: ignore
        elif result.geom_type == 'MultiPolygon':
            for idx, geom in enumerate(result.geoms):                                   #type: ignore
                new_polys[f"{self.name}_part{idx+1}"] = np.array(geom.exterior.coords)

        meta_copy = self.metadata.copy()
        meta_copy['boolean_history'] = f"{self.name} {operation} {other_roi.name}"

        return ROIChannel(
            name=f"{self.name}_{operation}_{other_roi.name}",
            polygons=new_polys,
            spatial_meta=self.spatial_meta,
            metadata=meta_copy
        )

    def __add__(self, other): return self._boolean_operation(other, 'union')
    def __sub__(self, other): return self._boolean_operation(other, 'difference')
    def __and__(self, other): return self._boolean_operation(other, 'intersection')

    # ------------------------------------------------------------------
    # Per-polygon mask operations  (preserve bin structure / styles)
    # ------------------------------------------------------------------

    def _resolve_mask(self, mask) -> Polygon:
        """Convert mask to a single Shapely Polygon for use in boolean ops."""
        if isinstance(mask, np.ndarray):
            return Polygon(mask)
        if isinstance(mask, ROIChannel):
            combined = Polygon()
            for pts in mask.polygons.values():
                combined = combined.union(Polygon(pts))
            return combined
        if hasattr(mask, 'geom_type'):
            return mask
        raise TypeError("mask must be a numpy array, ROIChannel, or Shapely geometry")

    def _apply_mask_op(self, mask, operation: str, name_suffix: str) -> 'ROIChannel':
        """
        Apply a Shapely boolean op to each polygon individually, preserving
        polygon names, per-polygon styles, and labels.  Polygons that become
        empty after the operation are dropped.  If a MultiPolygon results,
        only the largest piece is kept.
        """
        mask_poly = self._resolve_mask(mask)
        new_polys  = {}
        new_styles = {}

        for poly_name, coords in self.polygons.items():
            if operation == 'difference':
                result = Polygon(coords).difference(mask_poly)
            else:
                result = Polygon(coords).intersection(mask_poly)

            if result.is_empty:
                continue
            if result.geom_type == 'MultiPolygon':
                result = max(result.geoms, key=lambda g: g.area)
            if result.geom_type != 'Polygon':
                continue

            new_polys[poly_name] = np.array(result.exterior.coords)
            if poly_name in self.polygon_styles:
                new_styles[poly_name] = self.polygon_styles[poly_name]

        # Re-place labels: keep original position if still inside the polygon,
        # otherwise snap to centroid of the matching polygon (bin_N convention).
        new_labels = {}
        for label_text, (lx, ly) in self.labels.items():
            pt = Point(lx, ly)
            poly_key = f'bin_{label_text}'
            if poly_key in new_polys:
                poly_sh = Polygon(new_polys[poly_key])
                if poly_sh.contains(pt):
                    new_labels[label_text] = (lx, ly)
                else:
                    c = poly_sh.centroid
                    new_labels[label_text] = (c.x, c.y)
            else:
                new_labels[label_text] = (lx, ly)

        return ROIChannel(
            name=f"{self.name}_{name_suffix}",
            polygons=new_polys,
            polygon_styles=new_styles,
            labels=new_labels,
            spatial_meta=self.spatial_meta,
            metadata=self.metadata.copy()   # carries set_style() defaults
        )

    def crop(self, mask) -> 'ROIChannel':
        """
        Remove mask from each polygon individually.  Preserves per-polygon
        names, colormap fills, and labels.  Useful for cutting out tissue-edge
        artifacts while keeping a binned structure intact.

        Parameters
        ----------
        mask : np.ndarray [N, 2], ROIChannel, or Shapely geometry
            The region to remove.  If a ROIChannel, the union of its polygons
            is used as the mask.

        Returns
        -------
        New ROIChannel with the mask subtracted from each polygon.
        """
        return self._apply_mask_op(mask, 'difference', 'cropped')

    def intersect(self, mask) -> 'ROIChannel':
        """
        Clip each polygon to the region inside mask.  Useful for deriving a
        spatially specific sub-channel (e.g. cortex bins clipped to a 'medial'
        ROI) while keeping all bin labels, styles, and the original channel name.

        Parameters
        ----------
        mask : np.ndarray [N, 2], ROIChannel, or Shapely geometry
            The region to keep.  If a ROIChannel, the union of its polygons
            is used as the mask.

        Returns
        -------
        New ROIChannel with each polygon clipped to the mask.
        """
        return self._apply_mask_op(mask, 'intersection', 'clipped')
