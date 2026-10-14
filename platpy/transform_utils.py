import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional

# =============================================================================
# CORE MATRIX MATH
# =============================================================================

def create_transform_matrix(scale_x: float, scale_y: float, tx: float = 0.0, ty: float = 0.0, rotation_deg: float = 0.0) -> np.ndarray:
    """Creates a 3x3 affine matrix supporting scaling, translation, and rotation."""
    theta = np.radians(rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    return np.array([
        [scale_x * cos_t, -scale_y * sin_t, tx],
        [scale_x * sin_t,  scale_y * cos_t, ty],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

def create_scale_matrix(sx: float, sy: float) -> np.ndarray:
    """Helper for pure scaling (legacy fallback)."""
    return create_transform_matrix(sx, sy, 0.0, 0.0)

def apply_affine(matrix: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Applies a 3x3 affine matrix to an array of (N, 2) coordinates."""
    coords = np.atleast_2d(coords)
    coords_homo = np.hstack([coords, np.ones((coords.shape[0], 1))])
    transformed = (matrix @ coords_homo.T).T
    return transformed[:, :2]

def pixels_to_microns(transform_matrix: np.ndarray, px_coords: np.ndarray) -> np.ndarray:
    """Convert (N, 2) pixel coordinates to absolute micron coordinates."""
    return apply_affine(transform_matrix, px_coords)

def microns_to_pixels(transform_matrix: np.ndarray, um_coords: np.ndarray) -> np.ndarray:
    """Convert (N, 2) absolute micron coordinates to pixel coordinates."""
    return apply_affine(np.linalg.inv(transform_matrix), um_coords)

# =============================================================================
# GEOMETRY & BOUNDING BOX UTILITIES
# =============================================================================

def aabb_to_poly(min_x: float, min_y: float, max_x: float, max_y: float) -> np.ndarray:
    """
    Converts an Axis-Aligned Bounding Box (AABB) into a 4-point polygon ROI.
    Returns a numpy array of shape (4, 2) in [X, Y] format.
    """
    return np.array([
        [min_x, min_y], # Top-Left (or Bottom-Left depending on origin)
        [max_x, min_y], # Top-Right
        [max_x, max_y], # Bottom-Right
        [min_x, max_y]  # Bottom-Left
    ], dtype=np.float64)

def get_poly_bounds(poly_pts: np.ndarray) -> Tuple[float, float, float, float]:
    """Standardized AABB extractor for any polygon. Returns: (min_x, max_x, min_y, max_y)"""
    min_x, min_y = np.nanmin(poly_pts, axis=0)
    max_x, max_y = np.nanmax(poly_pts, axis=0)
    return float(min_x), float(max_x), float(min_y), float(max_y)

def is_poly_axis_aligned(poly_pts: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Checks if a polygon is a rectangle aligned to the X/Y axes.

    Area-based, not a count of unique x/y values: a right triangle with
    axis-aligned legs (e.g. [[10,10],[60,10],[10,60]]) also has exactly 2
    unique x's and 2 unique y's, so that count alone misclassifies it as a
    rectangle. True only when the polygon's own area equals its bounding
    box's area -- i.e. it actually fills its AABB, not just shares it.
    `tolerance` is a fraction of the AABB's area.
    """
    from shapely.geometry import Polygon
    min_x, max_x, min_y, max_y = get_poly_bounds(poly_pts)
    aabb_area = (max_x - min_x) * (max_y - min_y)
    if aabb_area <= 0:
        return True  # degenerate -- nothing meaningful to mask either way
    poly_area = Polygon(poly_pts).area
    return abs(poly_area - aabb_area) < tolerance * aabb_area

# =============================================================================
# SPATIAL METADATA (THE ABSOLUTE TRUTH)
# =============================================================================

@dataclass
class SpatialMetadata:
    """The Single Source of Truth for where data lives in physical space."""
    transform_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    
    # local_bounds represent the internal size BEFORE transformations.
    local_bounds: Optional[Tuple[float, float, float, float]] = None
    unit: str = "microns"

    # Placeholder for sub-field registration correction (v0.1.1).
    # Will hold a thin-plate spline or sparse offset matrix to correct the small
    # (~few µm) residual misalignment between Visium spot grid and fullres images.
    # None = ignored. Only applied at high-resolution renders/extractions (< ~2 µm/px).
    fine_spatial: Optional[object] = None

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """Dynamically calculates the absolute physical bounding box (AABB)."""
        if self.local_bounds is None:
            # Fallback to prevent crashes in plotting utilities
            return (0.0, 1.0, 0.0, 1.0) 
            
        min_x, max_x, min_y, max_y = self.local_bounds
        
        # Matrix math to transform local pixel bounds to absolute micron space
        corners = np.array([
            [min_x, min_y], [max_x, min_y],
            [max_x, max_y], [min_x, max_y]
        ])
        transformed = apply_affine(self.transform_matrix, corners)
        
        return (
            float(np.min(transformed[:, 0])),
            float(np.max(transformed[:, 0])),
            float(np.min(transformed[:, 1])),
            float(np.max(transformed[:, 1]))
        )

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Alias for extent — compatibility with SpatialData / scverse conventions."""
        return self.extent