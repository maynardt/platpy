import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon as SPoly, box as SBox
from shapely.affinity import rotate as srotate
from matplotlib.path import Path as MplPath


def resample_by_arclength(curve, n_points):
    """Resample a curve to n_points equally spaced along its arc length."""
    deltas = np.diff(curve, axis=0)
    dists = np.sqrt((deltas ** 2).sum(axis=1))
    cumulative = np.insert(np.cumsum(dists), 0, 0)
    t = cumulative / cumulative[-1]
    new_t = np.linspace(0, 1, n_points)
    return np.stack([
        np.interp(new_t, t, curve[:, 0]),
        np.interp(new_t, t, curve[:, 1])
    ], axis=1)


def _sort_by_angle(points, reference, axis):
    orth = np.array([-axis[1], axis[0]])
    projections = np.dot(points - reference, orth)
    return points[np.argsort(projections)]


def _seg_intersect(p1, p2, p3, p4):
    """Parametric intersection of segment p1-p2 with p3-p4.
    Returns (t, point) where t is position along p1-p2, or None if no hit."""
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-10:
        return None
    diff = p3 - p1
    t = (diff[0] * d2[1] - diff[1] * d2[0]) / cross
    u = (diff[0] * d1[1] - diff[1] * d1[0]) / cross
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t, p1 + t * d1
    return None


def _find_poly_intersections(polygon, line_p0, line_p1):
    """Find where line segment line_p0→line_p1 crosses the polygon boundary.
    Returns list of (arc_pos, point) sorted by arc_pos.
    arc_pos is a float in [0, n_vertices): vertex index + fraction along that edge."""
    n = len(polygon)
    results = []
    for i in range(n):
        hit = _seg_intersect(polygon[i], polygon[(i + 1) % n], line_p0, line_p1)
        if hit is not None:
            t, pt = hit
            results.append((i + t, pt))
    results.sort(key=lambda x: x[0])
    return results


def _extract_arc(polygon, pos_start, pos_end, pt_start, pt_end):
    """Extract the polygon arc going forward (increasing index, with wrap)
    from pos_start to pos_end, including interpolated endpoints."""
    n = len(polygon)
    forward_end = (pos_end - pos_start) % n
    pts = [pt_start]
    i = (int(pos_start) + 1) % n
    count = 0
    while count < n:
        forward_i = (float(i) - pos_start) % n
        if forward_i >= forward_end:
            break
        pts.append(polygon[i])
        i = (i + 1) % n
        count += 1
    pts.append(pt_end)
    return np.array(pts)


def extract_edges_from_endcaps(polygon, cap1_pts, cap2_pts):
    """
    Extract ordered pial and WM edge curves from a cortex polygon using
    two end cap line segments.

    Each cap runs pial→WM across one end of the cortex strip.  The polygon
    boundary is split at the four cap-intersection points; the two arcs that
    connect same-side endpoints are returned.

    Parameters
    ----------
    polygon : np.ndarray [N, 2]
        Closed cortex polygon in micron space.
    cap1_pts : np.ndarray [2, 2]
        First end cap: [[pial_x, pial_y], [wm_x, wm_y]].
    cap2_pts : np.ndarray [2, 2]
        Second end cap: [[pial_x, pial_y], [wm_x, wm_y]].

    Returns
    -------
    pial_edge : np.ndarray [M, 2]
    wm_edge   : np.ndarray [M, 2]
    """
    polygon  = np.asarray(polygon,  dtype=float)
    cap1_pts = np.asarray(cap1_pts, dtype=float)
    cap2_pts = np.asarray(cap2_pts, dtype=float)
    n = len(polygon)

    isects1 = _find_poly_intersections(polygon, cap1_pts[0], cap1_pts[1])
    isects2 = _find_poly_intersections(polygon, cap2_pts[0], cap2_pts[1])

    if len(isects1) != 2:
        raise ValueError(f"Cap 1 intersects polygon {len(isects1)} time(s); expected 2. "
                         "Check that the line crosses the polygon boundary twice.")
    if len(isects2) != 2:
        raise ValueError(f"Cap 2 intersects polygon {len(isects2)} time(s); expected 2. "
                         "Check that the line crosses the polygon boundary twice.")

    def assign_pial_wm(isects, pial_end):
        pos0, pt0 = isects[0]
        pos1, pt1 = isects[1]
        if np.linalg.norm(pt0 - pial_end) < np.linalg.norm(pt1 - pial_end):
            return (pos0, pt0), (pos1, pt1)
        return (pos1, pt1), (pos0, pt0)

    cap1_pial, cap1_wm = assign_pial_wm(isects1, cap1_pts[0])
    cap2_pial, cap2_wm = assign_pial_wm(isects2, cap2_pts[0])

    # Going forward from cap1_pial, which do we hit first: cap2_pial or a WM point?
    f_p2p = (cap2_pial[0] - cap1_pial[0]) % n
    f_p1w = (cap1_wm[0]   - cap1_pial[0]) % n
    f_p2w = (cap2_wm[0]   - cap1_pial[0]) % n

    if f_p2p < f_p1w and f_p2p < f_p2w:
        pial_edge = _extract_arc(polygon, cap1_pial[0], cap2_pial[0], cap1_pial[1], cap2_pial[1])
        wm_edge   = _extract_arc(polygon, cap2_wm[0],   cap1_wm[0],   cap2_wm[1],   cap1_wm[1])
    else:
        pial_edge = _extract_arc(polygon, cap2_pial[0], cap1_pial[0], cap2_pial[1], cap1_pial[1])
        wm_edge   = _extract_arc(polygon, cap1_wm[0],   cap2_wm[0],   cap1_wm[1],   cap2_wm[1])

    # Pial and WM are on opposite sides of the closed polygon so _extract_arc
    # always traverses them in opposite directions.  Reverse WM so both edges
    # run cap1-end → cap2-end; build_depth_bins needs that correspondence.
    return pial_edge, wm_edge[::-1]


def unfold_cortex(adata, pial_edge, wm_edge, n_lat=200):
    """
    Compute curvilinear (unfolded) coordinates for each cell in a cortex strip.

    Resamples the pial and WM edges into n_lat paired points ("ribs").  For
    each cell, the nearest rib is found and the cell is projected onto it,
    giving a lateral coordinate (position along the cortex surface) and a
    depth coordinate (fractional depth pial → WM).

    Parameters
    ----------
    adata : AnnData
        Must have obsm['spatial'] as (N, 2) micron coordinates.
    pial_edge : np.ndarray [M, 2]
        Ordered pial surface points from extract_edges_from_endcaps.
    wm_edge : np.ndarray [M, 2]
        Ordered WM boundary points (same direction as pial_edge).
    n_lat : int
        Number of ribs — lateral sampling resolution.

    Returns
    -------
    AnnData copy with obs['lat_coord'] and obs['depth_coord'] added.
        lat_coord   : float [0, 1] — position along cortex (cap1 → cap2 end)
        depth_coord : float [0, 1] — depth from pial (0) to WM (1)
    """
    pial = resample_by_arclength(pial_edge, n_lat)   # [n_lat, 2]
    wm   = resample_by_arclength(wm_edge,   n_lat)   # [n_lat, 2]
    ribs        = wm - pial                           # [n_lat, 2]
    rib_len_sq  = (ribs ** 2).sum(axis=1)             # [n_lat]

    coords = np.asarray(adata.obsm['spatial'])        # [N, 2]

    # Vector from each pial point to each cell: [N, n_lat, 2]
    v = coords[:, np.newaxis, :] - pial[np.newaxis, :, :]

    # Projection parameter t along each rib: [N, n_lat]
    t = (v * ribs[np.newaxis, :, :]).sum(axis=2) / np.maximum(rib_len_sq, 1e-10)
    t_clamped = np.clip(t, 0.0, 1.0)

    # Closest point on each rib to each cell: [N, n_lat, 2]
    closest = pial[np.newaxis, :, :] + t_clamped[:, :, np.newaxis] * ribs[np.newaxis, :, :]

    # Distance from each cell to each rib: [N, n_lat]
    dists = np.linalg.norm(coords[:, np.newaxis, :] - closest, axis=2)

    nearest      = np.argmin(dists, axis=1)                          # [N]
    lat_coords   = nearest / (n_lat - 1)
    depth_coords = t_clamped[np.arange(len(coords)), nearest]

    adata = adata.copy()
    adata.obs['lat_coord']   = lat_coords
    adata.obs['depth_coord'] = depth_coords
    return adata


def extract_ordered_edges(polygon, direction):
    """
    Split a polygon into two ordered edge curves along a given direction.

    Useful for cortical ROIs where 'direction' is roughly normal to the layers
    (e.g. direction=[0,-1] for a dorsal-to-ventral axis).

    Returns (top_edge, bottom_edge) as ordered [N, 2] arrays.
    """
    direction = direction / np.linalg.norm(direction)
    origin = polygon.mean(axis=0)
    projections = np.dot(polygon - origin, direction)
    threshold = (projections.min() + projections.max()) / 2
    top = polygon[projections < threshold]
    bottom = polygon[projections >= threshold]
    nearest = np.unravel_index(np.argmin(cdist(top, bottom)), (len(top), len(bottom)))
    top_sorted = _sort_by_angle(top, top[nearest[0]], direction)
    bottom_sorted = _sort_by_angle(bottom, bottom[nearest[1]], direction)
    return top_sorted, bottom_sorted


def segment_curve_by_arclength(curve, n_segments):
    """
    Split a curve into n_segments of equal arc length, preserving original shape.

    Returns a list of n_segments arrays, each a contiguous piece of the curve.
    """
    deltas = np.diff(curve, axis=0)
    dists = np.sqrt((deltas ** 2).sum(axis=1))
    cumulative = np.insert(np.cumsum(dists), 0, 0)
    targets = np.linspace(0, cumulative[-1], n_segments + 1)

    segments = []
    points = [curve[0]]
    i = 0
    for t in targets[1:]:
        while i < len(cumulative) - 2 and cumulative[i + 1] < t:
            points.append(curve[i + 1])
            i += 1
        seg_len = cumulative[i + 1] - cumulative[i]
        frac = (t - cumulative[i]) / seg_len if seg_len > 0 else 0
        interp_pt = curve[i] + frac * (curve[i + 1] - curve[i])
        points.append(interp_pt)
        segments.append(np.array(points))
        points = [interp_pt]
    return segments


def _segments_from_markers(curve, markers, n_interp):
    """Cut arc segments from curve at positions nearest to each marker point."""
    deltas = np.diff(curve, axis=0)
    dists = np.sqrt((deltas ** 2).sum(axis=1))
    cumulative = np.insert(np.cumsum(dists), 0, 0)
    t = cumulative / cumulative[-1]

    def at(s):
        return np.array([np.interp(s, t, curve[:, dim]) for dim in range(2)]).T

    marker_t = np.array([
        t[np.argmin(np.linalg.norm(curve - pt, axis=1))]
        for pt in markers
    ])

    return [at(np.linspace(marker_t[i], marker_t[i + 1], n_interp))
            for i in range(len(marker_t) - 1)]


def build_depth_bins(top_curve, bottom_curve, n_bins=6, n_interp=100):
    """
    Build depth-wise bin polygons by interpolating between two edge curves.

    Divides the cortex into bands parallel to the edges, approximating
    cytoarchitectural depth (e.g. layers I–VI). Complementary to
    build_layer_bins, which cuts perpendicular to the layers.

    Parameters
    ----------
    top_curve : np.ndarray [N, 2]
        Ordered points along the superficial edge (e.g. pial surface).
    bottom_curve : np.ndarray [N, 2]
        Ordered points along the deep edge (e.g. white matter boundary).
    n_bins : int
        Number of depth bins (6 is a natural default for cortical layers).
    n_interp : int
        Number of points per isopleth curve.

    Returns
    -------
    list of np.ndarray [M, 2]
        One closed polygon per bin, ordered superficial to deep.
    """
    top = resample_by_arclength(top_curve, n_interp)
    bottom = resample_by_arclength(bottom_curve, n_interp)
    isopleths = [(1 - f) * top + f * bottom
                 for f in np.linspace(0, 1, n_bins + 1)]
    return [
        np.vstack([isopleths[i], isopleths[i + 1][::-1]])
        for i in range(n_bins)
    ]


def build_layer_bins(top_curve, bottom_curve, n_bins=4, n_interp=100):
    """
    Build cortical layer bin polygons between two ordered edge curves.

    Each bin follows the actual curvature of the top and bottom edges rather
    than using straight-line interpolation between them.

    Parameters
    ----------
    top_curve : np.ndarray [N, 2]
        Ordered points along one edge of the cortical ROI (e.g. pial surface).
    bottom_curve : np.ndarray [N, 2]
        Ordered points along the opposite edge (e.g. white matter boundary).
    n_bins : int
        Number of bins to create.
    n_interp : int
        Number of points per arc segment in the output polygons.

    Returns
    -------
    list of np.ndarray [M, 2]
        One closed polygon per bin, in the same coordinate space as the inputs.
    """
    top_markers = resample_by_arclength(top_curve, n_bins + 1)
    bottom_markers = resample_by_arclength(bottom_curve, n_bins + 1)
    top_arcs = _segments_from_markers(top_curve, top_markers, n_interp)
    bottom_arcs = _segments_from_markers(bottom_curve, bottom_markers, n_interp)
    return [
        np.vstack([t, b[::-1], t[0:1]])
        for t, b in zip(top_arcs, bottom_arcs)
    ]


def build_axis_bins(polygon, n_bins, direction=None, angle_deg=None, line=None):
    """
    Slice a polygon into n_bins strips along a given axis.

    Cuts are made perpendicular to `direction`, dividing the polygon into
    equal-width bands. Useful for simple positional grids (e.g. anterior-
    posterior or medial-lateral) without needing a traced edge curve.

    Parameters
    ----------
    polygon : np.ndarray [N, 2]
        Closed polygon in micron space.
    n_bins : int
        Number of strips to create.
    direction : array-like [2], optional
        Unit vector defining the binning axis (cuts are perpendicular to this).
    angle_deg : float, optional
        Angle in degrees (0 = along x, 90 = along y). Alternative to direction.
    line : array-like [(x1,y1), (x2,y2)], optional
        Two points defining the binning axis. Direction is computed as p2 - p1.
        Alternative to direction and angle_deg.

    Returns
    -------
    list of np.ndarray [M, 2]
        One polygon per bin, in the same coordinate space as the input.
    """
    if line is not None:
        p1, p2 = np.array(line[0], dtype=float), np.array(line[1], dtype=float)
        direction = p2 - p1
    elif angle_deg is not None:
        rad = np.deg2rad(angle_deg)
        direction = np.array([np.cos(rad), np.sin(rad)])
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)

    poly_sh = SPoly(polygon)
    cx, cy = poly_sh.centroid.coords[0]

    # Rotate so direction aligns with x-axis, then slice by y
    rot_angle = np.degrees(np.arctan2(direction[1], direction[0]))
    rotated = srotate(poly_sh, -rot_angle, origin=(cx, cy))

    x_min, y_min, x_max, y_max = rotated.bounds
    margin = max(x_max - x_min, y_max - y_min)
    cuts = np.linspace(y_min, y_max, n_bins + 1)

    bins = []
    for i in range(n_bins):
        clip = SBox(x_min - margin, cuts[i], x_max + margin, cuts[i + 1])
        strip = rotated.intersection(clip)
        if strip.is_empty:
            continue
        strip_back = srotate(strip, rot_angle, origin=(cx, cy))
        if strip_back.geom_type == 'MultiPolygon':
            strip_back = max(strip_back.geoms, key=lambda g: g.area)
        bins.append(np.array(strip_back.exterior.coords))

    return bins


def build_contour_shells(polygon, n_shells, step_um=None):
    """
    Build onion-skin layers by progressively eroding a polygon inward.

    Each shell is the annular region between successive erosions, returned
    as an (outer, inner) pair of coordinate arrays. Useful for proximity-to-
    boundary analyses or as a geometry-agnostic control for layer binning.

    Parameters
    ----------
    polygon : np.ndarray [N, 2]
        Closed polygon in micron space.
    n_shells : int
        Number of shells to generate (inward from the boundary).
    step_um : float, optional
        Erosion distance per shell in microns. If None, auto-computed as
        an equal division of the polygon's approximate inradius.

    Returns
    -------
    list of (np.ndarray [M, 2], np.ndarray [K, 2] or None)
        One (outer_coords, inner_coords) tuple per shell, ordered outermost first.
        The final entry has inner_coords = None (solid innermost region).
    """
    poly_sh = SPoly(polygon)

    if step_um is None:
        # Inradius = shortest distance from centroid to boundary.
        # After (n_shells - 1) erosions of step_um each, the remaining radius is
        # inradius - (n_shells-1)*step_um = step_um, so the solid center is the
        # same visual width as every annular ring.
        inradius = poly_sh.centroid.distance(poly_sh.exterior)
        step_um  = inradius / n_shells

    shells = []
    outer = poly_sh
    for _ in range(n_shells):
        inner = outer.buffer(-step_um)
        if inner.is_empty:
            break
        if inner.geom_type == 'MultiPolygon':
            inner = max(inner.geoms, key=lambda g: g.area)
        shells.append((
            np.array(outer.exterior.coords),
            np.array(inner.exterior.coords)
        ))
        outer = inner

    if not outer.is_empty and outer.geom_type == 'Polygon':
        shells.append((np.array(outer.exterior.coords), None))

    return shells


def assign_contour_shells(adata, shells):
    """
    Assign each cell to its contour shell. Adds 'shell' column to adata.obs.
    Shell 0 is outermost; cells outside all shells get shell = -1.

    Parameters
    ----------
    adata  : AnnData with obsm['spatial'] = (N, 2) micron coordinates
    shells : list of (outer, inner) coordinate pairs from build_contour_shells()

    Returns
    -------
    adata with obs['shell'] added in-place.
    """
    cell_coords = adata.obsm['spatial']
    shell_ids = np.full(adata.n_obs, -1, dtype=int)
    for i, (outer, inner) in enumerate(shells):
        in_outer = MplPath(outer).contains_points(cell_coords)
        if inner is not None:
            in_inner = MplPath(inner).contains_points(cell_coords)
            mask = in_outer & ~in_inner
        else:
            mask = in_outer
        shell_ids[mask] = i
    adata.obs['shell'] = shell_ids
    return adata


def assign_bins(adata, bin_polys, col='bin'):
    """
    Assign each cell to the first bin polygon that contains it.
    Adds an integer column to adata.obs; cells outside all bins get -1.

    Parameters
    ----------
    adata     : AnnData with obsm['spatial'] = (N, 2) micron coordinates
    bin_polys : list of np.ndarray [M, 2] — ordered bin polygons
    col       : str — obs column name to write (default 'bin')

    Returns
    -------
    adata with obs[col] added in-place.
    """
    cell_coords = adata.obsm['spatial']
    bin_ids = np.full(adata.n_obs, -1, dtype=int)
    for i, poly in enumerate(bin_polys):
        poly_closed = np.vstack([poly, poly[0]])
        mask = MplPath(poly_closed).contains_points(cell_coords)
        bin_ids[mask] = i
    adata.obs[col] = bin_ids
    return adata


def bins_to_roi_channel(bin_polys, name='bins', cmap_name='RdYlBu',
                        line_color='black', line_width=1,
                        label_color='black', label_size=1.0, label_weight=0.12,
                        label_anchor=None, label_frac=0.75, label_line=None):
    """
    Convert a list of bin polygons into a ROIChannel with per-polygon colormap
    fill and centered numeric labels. Compatible with output from build_axis_bins(),
    build_depth_bins(), or [outer for outer, _ in build_contour_shells()].

    Parameters
    ----------
    bin_polys     : list of np.ndarray [N, 2] — ordered bin polygons in micron space
    name          : str — name for the ROIChannel
    cmap_name     : str — matplotlib colormap name applied across bins (index 0 → n-1)
    line_color, line_width, label_color, label_size — passed to set_style()
    label_line    : [(x1,y1), (x2,y2)], optional
        Two points in micron space. Labels are placed at evenly-spaced positions
        from p1 (bin 0) to p2 (bin n-1) along this line. Takes precedence over
        label_anchor when both are supplied.
    label_anchor  : None, float, or str
        Where to place each label relative to its polygon's centroid.
        None   — at the centroid (default; works well for axis bins)
        float  — angle in degrees, counterclockwise from +x axis
        str    — 'right', 'top', 'left', 'bottom' (image-space aliases;
                 assumes y increases downward as in Xenium micron space,
                 so 'top' means the smallest-y direction)
    label_frac    : float (0–1)
        Fraction of the way from centroid to polygon boundary where the
        label is placed (default 0.75). Increase toward 1.0 to push
        labels closer to the edge.

    Returns
    -------
    ROIChannel ready to add to a ViewPort as a layer
    """
    import matplotlib as mpl
    from shapely.geometry import LineString
    from .channels import ROIChannel
    cmap = mpl.colormaps[cmap_name]
    n = len(bin_polys)

    # Resolve label_anchor string to angle (y-down image convention).
    _anchor_map = {'right': 0.0, 'bottom': 90.0, 'left': 180.0, 'top': 270.0}
    if isinstance(label_anchor, str):
        if label_anchor not in _anchor_map:
            raise ValueError(f"label_anchor must be one of {list(_anchor_map.keys())} or a float angle")
        angle_deg = _anchor_map[label_anchor]
    elif label_anchor is not None:
        angle_deg = float(label_anchor)
    else:
        angle_deg = None

    # Use the outermost polygon's centroid as the common origin for all ray casts.
    # This keeps all labels on a single clean radial line regardless of how the
    # inner polygon centroids drift as the shells erode inward.
    if angle_deg is not None:
        outer_sh = SPoly(bin_polys[0])
        cx0, cy0 = outer_sh.centroid.x, outer_sh.centroid.y
        theta = np.radians(angle_deg)
        dx, dy = np.cos(theta), np.sin(theta)

    def _ray_hit(poly_coords):
        """Return the nearest boundary intersection of a ray from (cx0,cy0)."""
        sh = SPoly(poly_coords)
        diag = np.hypot(sh.bounds[2] - sh.bounds[0], sh.bounds[3] - sh.bounds[1]) * 2
        ray = LineString([(cx0, cy0), (cx0 + dx * diag, cy0 + dy * diag)])
        hit = ray.intersection(sh.exterior)
        if hit.is_empty:
            return None
        if hit.geom_type == 'MultiPoint':
            hit = min(hit.geoms, key=lambda p: (p.x - cx0)**2 + (p.y - cy0)**2)
        return (hit.x, hit.y)

    def _label_pos(i):
        poly = bin_polys[i]
        if angle_deg is None:
            return (float(poly[:, 0].mean()), float(poly[:, 1].mean()))
        outer_hit = _ray_hit(poly)
        if outer_hit is None:
            return (cx0, cy0)
        # Place label at midpoint between this shell's boundary and the next one's
        # (i.e. centred inside the annular ring).  For the innermost shell use
        # label_frac toward its own boundary instead.
        if i + 1 < len(bin_polys):
            inner_hit = _ray_hit(bin_polys[i + 1])
            if inner_hit is not None:
                return ((outer_hit[0] + inner_hit[0]) / 2,
                        (outer_hit[1] + inner_hit[1]) / 2)
        return (cx0 + (outer_hit[0] - cx0) * label_frac,
                cy0 + (outer_hit[1] - cy0) * label_frac)

    polygons = {f'bin_{i}': poly for i, poly in enumerate(bin_polys)}
    polygon_styles = {
        f'bin_{i}': {
            'fill': True,
            'fill_color': mpl.colors.to_hex(cmap(i / max(n - 1, 1)))
        }
        for i in range(n)
    }
    if label_line is not None:
        _p1 = np.array(label_line[0], dtype=float)
        _p2 = np.array(label_line[1], dtype=float)
        def _label_pos(i):
            t = i / max(n - 1, 1)
            pt = _p1 + t * (_p2 - _p1)
            return (float(pt[0]), float(pt[1]))

    labels = {str(i): _label_pos(i) for i in range(n)}
    return ROIChannel(
        name=name, polygons=polygons,
        polygon_styles=polygon_styles, labels=labels
    ).set_style(line_color=line_color, line_width=line_width,
                label_color=label_color, label_size=label_size, label_weight=label_weight)


def save_rois(path, roi_dict):
    """
    Save a roi_dict of {name: np.ndarray [N, 2]} to a JSON file.

    Parameters
    ----------
    path : str or Path
        Output file path (e.g. './demo/rois.json').
    roi_dict : dict of str -> np.ndarray [N, 2]
        Named polygons in micron space.
    """
    import json
    import numpy as np
    def _to_list(v):
        return v.tolist() if isinstance(v, np.ndarray) else v
    serializable = {name: _to_list(val) for name, val in roi_dict.items()}
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)


def load_rois(path):
    """
    Load a roi_dict from a JSON file written by save_rois().

    Parameters
    ----------
    path : str or Path
        JSON file path.

    Returns
    -------
    dict of str -> np.ndarray [N, 2]
        Named polygons in micron space, ready for annotate_roi_membership().
    """
    import json
    with open(path, 'r') as f:
        data = json.load(f)
    return {name: np.array(pts) for name, pts in data.items()}


def get_cells_in_roi(adata, roi_poly_um):
    """
    Return a copy of adata containing only cells within a polygon ROI.

    Parameters
    ----------
    adata : AnnData
        Must have adata.obsm['spatial'] as (N, 2) micron-space coordinates.
    roi_poly_um : np.ndarray [M, 2]
        Polygon vertices in micron space.

    Returns
    -------
    AnnData subset (copy) of cells inside the polygon.
    """
    from matplotlib.path import Path
    coords = adata.obsm['spatial']
    poly_closed = np.vstack([roi_poly_um, roi_poly_um[0]])
    mask = Path(poly_closed).contains_points(coords)
    return adata[mask].copy()


def annotate_roi_membership(adata, roi_dict, col='roi',
                            cell_duplicates='first_match',
                            unassigned_label='unassigned'):
    """
    Annotate each cell in adata.obs with the name of the ROI it falls within.

    Parameters
    ----------
    adata : AnnData
        Must have adata.obsm['spatial'] as (N, 2) micron-space coordinates.
    roi_dict : dict of str -> np.ndarray [M, 2], or list of np.ndarray [M, 2]
        Named polygons in micron space. A list is auto-converted to
        {'bin_0': ..., 'bin_1': ...}. Dict order determines priority for
        'first_match' and 'overwrite' modes.
    col : str
        Name of the obs column to create.
    cell_duplicates : str
        How to handle cells inside multiple ROIs:
        'first_match'  — first ROI in dict order wins (default)
        'overwrite'    — last ROI in dict order wins
        'ambiguous'    — label conflicted cells as 'ambiguous'
        'undefined'    — label conflicted cells as NaN
        'concatenate'  — join all matching ROI names e.g. 'cortex striatum'
    unassigned_label : str or None
        Label for cells outside all ROIs. None keeps them as NaN (default: 'unassigned').

    Returns
    -------
    AnnData with obs[col] added as a Categorical.
    """
    if isinstance(roi_dict, list):
        roi_dict = {f'bin_{i}': poly for i, poly in enumerate(roi_dict)}
    import pandas as pd
    from matplotlib.path import Path

    valid_modes = ('first_match', 'overwrite', 'ambiguous', 'undefined', 'concatenate')
    if cell_duplicates not in valid_modes:
        raise ValueError(f"cell_duplicates must be one of {valid_modes}")

    coords = adata.obsm['spatial']
    labels = np.full(adata.n_obs, None, dtype=object)

    if cell_duplicates == 'concatenate':
        match_lists = [[] for _ in range(adata.n_obs)]
        for roi_name, poly in roi_dict.items():
            poly_closed = np.vstack([poly, poly[0]])
            in_roi = Path(poly_closed).contains_points(coords)
            for idx in np.where(in_roi)[0]:
                match_lists[idx].append(roi_name)
        labels = np.array([' '.join(m) if m else None for m in match_lists], dtype=object)
    else:
        assigned = np.zeros(adata.n_obs, dtype=bool)
        for roi_name, poly in roi_dict.items():
            poly_closed = np.vstack([poly, poly[0]])
            in_roi = Path(poly_closed).contains_points(coords)
            conflict = in_roi & assigned
            new      = in_roi & ~assigned

            if cell_duplicates == 'first_match':
                labels[new] = roi_name
                assigned[new] = True
            elif cell_duplicates == 'overwrite':
                labels[in_roi] = roi_name
                assigned[in_roi] = True
            elif cell_duplicates == 'ambiguous':
                labels[conflict] = 'ambiguous'
                labels[new] = roi_name
                assigned[new] = True
            elif cell_duplicates == 'undefined':
                labels[conflict] = None
                labels[new] = roi_name
                assigned[new] = True

    if unassigned_label is not None:
        labels[labels == None] = unassigned_label  # noqa: E711

    adata = adata.copy()
    adata.obs[col] = pd.Categorical(labels)
    return adata


def polygon_area_um2(pts):
    """
    Area of a polygon in µm² using the shoelace formula.
    pts : np.ndarray [N, 2] — vertices in micron space (open or closed).
    """
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def bin_expression_matrix(adata, markers, bin_col='bin', bin_keys=None):
    """
    Compute mean expression of a marker set per spatial bin.

    Works with any binning column produced by annotate_roi_membership(),
    assign_contour_shells(), or similar — axis bins, depth bins, shells, etc.

    Parameters
    ----------
    adata     : AnnData with obs[bin_col] containing bin labels
    markers   : list of gene names (genes absent from adata.var_names are skipped)
    bin_col   : obs column with bin assignments (default 'bin')
    bin_keys  : ordered list of bin labels defining row order. If None, inferred
                from unique values of obs[bin_col] (excluding 'unassigned'), sorted
                numerically by trailing integer for 'prefix_N' labels or
                lexicographically otherwise.

    Returns
    -------
    expr     : np.ndarray [n_bins, n_markers]  — mean transcripts per cell
    bin_keys : list of str — bin labels in row order
    """
    if bin_keys is None:
        all_keys = [k for k in adata.obs[bin_col].unique()
                    if k != 'unassigned' and k != -1]
        try:
            bin_keys = sorted(all_keys, key=lambda k: int(str(k).rsplit('_', 1)[-1]))
        except (ValueError, IndexError, AttributeError):
            bin_keys = sorted(all_keys)

    markers = [m for m in markers if m in adata.var_names]
    expr = np.zeros((len(bin_keys), len(markers)), dtype=float)
    for i, key in enumerate(bin_keys):
        cells = adata[adata.obs[bin_col] == key]
        if cells.n_obs > 0:
            expr[i] = np.asarray(cells[:, markers].X.mean(axis=0)).flatten()
    return expr, bin_keys
