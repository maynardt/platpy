import numpy as np
import pandas as pd
import cv2
import time
from typing import Dict, List, Optional, Tuple

# SPACKLE imports
from channels.image_channel import ImageChannel
from transform_utils import SpatialMetadata, create_transform_matrix
from viewport import ViewPort


# =============================================================================
# COLOUR SCHEME
# 7 pure RGB combinations that survive resampling. Any nonzero R channel
# means the spot was red-component; presence/absence of each channel is
# unambiguous even after bilinear or area-averaging interpolation smears the
# spot slightly. Colour ID is the unique spot identity — no KD-tree needed.
# =============================================================================

MAX_VAL   = 65535
COLORS: Dict[int, Tuple[int, int, int]] = {
    1: (MAX_VAL,       0,       0),   # R
    2: (      0, MAX_VAL,       0),   # G
    3: (MAX_VAL, MAX_VAL,       0),   # RG  (yellow)
    4: (      0,       0, MAX_VAL),   # B
    5: (MAX_VAL,       0, MAX_VAL),   # RB  (magenta)
    6: (      0, MAX_VAL, MAX_VAL),   # GB  (cyan)
    7: (MAX_VAL, MAX_VAL, MAX_VAL),   # RGB (white)
}

def _color_id(r: bool, g: bool, b: bool) -> int:
    """Map RGB presence flags → 1-7 (0 = background, discard)."""
    return (1 if r else 0) + (2 if g else 0) + (4 if b else 0)


# =============================================================================
# CANVAS GENERATION  (called fresh each trial)
# =============================================================================

def _draw_7_spots(
    canvas:        np.ndarray,
    mpp:           float,
    w_um:          float,
    h_um:          float,
    spot_radius_um: float,
    min_sep_um:    float = 100.0,
    margin_um:     float = 50.0,
    max_attempts:  int   = 100_000,
) -> Dict[int, Tuple[float, float]]:
    """
    Place exactly 7 spots (one per pure colour) at random positions on `canvas`.
    Positions are rejection-sampled so no two spots are closer than min_sep_um.
    Returns {color_id: (x_um, y_um)} — the unambiguous ground truth for this trial.
    Canvas is modified in-place and should be zeroed before calling.
    """
    spot_r_px = max(1, int(spot_radius_um / mpp))
    positions: List[Tuple[float, float]] = []
    attempts  = 0

    while len(positions) < 7:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Could not place all 7 spots after {max_attempts} attempts. "
                "Increase canvas size or reduce min_sep_um."
            )
        x_um = float(np.random.uniform(margin_um, w_um - margin_um))
        y_um = float(np.random.uniform(margin_um, h_um - margin_um))
        too_close = any(
            np.hypot(x_um - px, y_um - py) < min_sep_um
            for px, py in positions
        )
        if not too_close:
            positions.append((x_um, y_um))
        attempts += 1

    ground_truth: Dict[int, Tuple[float, float]] = {}
    for cid, (x_um, y_um) in zip(range(1, 8), positions):
        x_px = int(round(x_um / mpp))
        y_px = int(round(y_um / mpp))
        cv2.circle(canvas, (x_px, y_px), spot_r_px, COLORS[cid], -1)
        # Ground truth is the ACTUAL rendered pixel centre (in microns), not the
        # original float — the rounding to integer pixels when drawing is not a
        # transform error and should not be charged against the measurement.
        ground_truth[cid] = (float(x_px) * mpp, float(y_px) * mpp)

    return ground_truth


def _make_channel(canvas: np.ndarray, mpp: float) -> ImageChannel:
    """Wrap a canvas array in an ImageChannel with correct spatial metadata."""
    h_px, w_px = canvas.shape[:2]
    return ImageChannel(
        name="synthetic_spots",
        image=canvas,
        spatial_meta=SpatialMetadata(
            transform_matrix=create_transform_matrix(mpp, mpp),
            local_bounds=(0.0, float(w_px), 0.0, float(h_px)),
            unit="microns"
        )
    )


# =============================================================================
# SPOT DETECTION
# Colour is the unique identity — no KD-tree, no ambiguous nearest-neighbour.
# =============================================================================

def _detect_spots(
    img:            np.ndarray,
    vp:             ViewPort,
    ground_truth:   Dict[int, Tuple[float, float]],
    edge_margin_px: float = 0.0,
) -> List[Tuple]:
    """
    Find lit blobs in a rendered RGBA image, identify each by colour,
    map centroid to microns via the viewport transform, compute error.
    edge_margin_px: discard any centroid this many pixels from the image boundary.
    Prevents partial-disc arc centroids (clipped spots) from inflating error stats.
    Returns list of (px_f, py_f, obs_x, obs_y, true_x, true_y, c_id, error_um).
    """
    H, W = img.shape[:2]
    bright = (np.max(img[..., :3], axis=2) > 50).astype(np.uint8) * 255
    n_lab, _, _, centroids = cv2.connectedComponentsWithStats(bright)
    results = []
    for j in range(1, n_lab):
        px_f, py_f = centroids[j]

        # Skip blobs whose centroid is too close to the viewport edge —
        # the spot circle is likely partially clipped, giving a biased centroid.
        if (px_f < edge_margin_px or px_f > W - edge_margin_px or
                py_f < edge_margin_px or py_f > H - edge_margin_px):
            continue

        px_i = int(np.clip(round(px_f), 0, W - 1))
        py_i = int(np.clip(round(py_f), 0, H - 1))
        rgb  = img[py_i, px_i, :3]
        c_id = _color_id(rgb[0] > 100, rgb[1] > 100, rgb[2] > 100)
        if c_id == 0 or c_id not in ground_truth:
            continue
        obs_um       = vp.pixels_to_microns(np.array([[px_f, py_f]]))[0]
        obs_x, obs_y = float(obs_um[0]), float(obs_um[1])
        true_x, true_y = ground_truth[c_id]
        error_um     = float(np.hypot(obs_x - true_x, obs_y - true_y))
        results.append((px_f, py_f, obs_x, obs_y, true_x, true_y, c_id, error_um))
    return results


# =============================================================================
# VIEWPORT PLACEMENT
# Pre-flight check before paying the render cost.
# =============================================================================

def _cast_viewport(
    vp:             ViewPort,
    fov_um:         float,
    w_um:           float,
    h_um:           float,
    ground_truth:   Dict[int, Tuple[float, float]],
    edge_margin_px: float = 0.0,
    max_retries:    int   = 100,
) -> Optional[Tuple]:
    """
    Randomly place and orient the viewport, retrying until at least one spot
    projects with its centre at least edge_margin_px pixels inside all four edges.
    This ensures the full spot disc is visible, avoiding partial-arc centroids.
    Returns (cx, cy, rot, flip_x, flip_y, rendered_img) or None if all retries blank.
    """
    margin = fov_um / 2.0

    for _ in range(max_retries):
        cx     = float(np.random.uniform(margin, w_um - margin))
        cy     = float(np.random.uniform(margin, h_um - margin))
        rot    = float(np.random.uniform(-180.0, 180.0))
        flip_x = bool(np.random.choice([True, False]))
        flip_y = bool(np.random.choice([True, False]))

        vp.update_view(center_um=(cx, cy), fov_um=(fov_um, fov_um),
                       rotation_deg=rot, flip_x=flip_x, flip_y=flip_y)

        H, W      = vp.shape
        spot_pts  = np.array([[x, y] for x, y in ground_truth.values()])
        px_coords = vp.microns_to_pixels(spot_pts)
        m         = edge_margin_px
        visible   = np.any(
            (px_coords[:, 0] >= m) & (px_coords[:, 0] <= W - m) &
            (px_coords[:, 1] >= m) & (px_coords[:, 1] <= H - m)
        )
        if visible:
            img = np.asarray(vp.render().image)
            return (cx, cy, rot, flip_x, flip_y, img)

    return None


# =============================================================================
# VARIANT 1: SCALE SWEEP  →  scatterplot of error_px vs scale_fraction
# =============================================================================

def run_scale_sweep(
    mpp            : float = 0.33,
    w_px           : int   = 8192,
    h_px           : int   = 8192,
    spot_radius_um : float = 2.0,
    min_sep_um     : float = 100.0,
    vp_width_px    : int   = 512,
    min_trials     : int   = 10,
    min_obs        : int   = 100,
    max_trials     : int   = 500,
    output_csv     : str   = "viewport_scale_sweep.csv",
) -> pd.DataFrame:
    """
    Sweep scale_fraction from 1.0 (1:1) down to ~0.05 (1:20) in 0.05-wide bands.
    Each band runs until BOTH min_trials renders AND min_obs observations are met.
    Each trial generates a FRESH random canvas so spatial bias is tested across
    all canvas regions, not just a fixed set of spots.

    scale_fraction = source_mpp / render_mpp  (1.0 = native, 0.05 = 20x downscale)
    """
    print("==============================================")
    print(" SPACKLE: Scale Sweep Accuracy Test")
    print("==============================================")

    w_um = w_px * mpp
    h_um = h_px * mpp

    # Build scale-fraction bands, high→low, filtered to bands that fit the canvas
    min_valid_sf = (mpp * vp_width_px) / (min(w_um, h_um) * 0.90)
    breakpoints  = np.round(np.arange(0.05, 1.01, 0.05), 2)
    raw_ranges   = list(zip(breakpoints[:-1], breakpoints[1:]))[::-1]
    scale_ranges = [(max(float(lo), round(min_valid_sf, 3)), float(hi))
                    for lo, hi in raw_ranges if float(hi) > min_valid_sf]

    if not scale_ranges:
        print("[!] Canvas too small for any valid scale range.")
        return pd.DataFrame()

    print(f"[*] Canvas: {w_px}x{h_px} px  ({w_um:.0f}x{h_um:.0f} µm)  mpp={mpp}")
    print(f"[*] Spots:  7 per trial, r={spot_radius_um}µm, min_sep={min_sep_um}µm")
    print(f"[*] {len(scale_ranges)} scale bands  "
          f"(stop: trials≥{min_trials} AND N≥{min_obs})\n")

    # Create a viewport — canvas updated each trial via overwrite=True
    vp = ViewPort.create(target_width_px=vp_width_px,
                         center_um=(w_um / 2, h_um / 2), fov_um=(500.0, 500.0))

    log_data   = []
    start_time = time.time()

    for lo, hi in scale_ranges:
        trials = obs_count = 0
        print(f"  [{lo:.2f}, {hi:.2f}]  ", end='', flush=True)

        while not (trials >= min_trials and obs_count >= min_obs):
            if trials >= max_trials:
                print(f"  [!] max_trials={max_trials} reached.")
                break

            # Fresh canvas and random spot positions every trial
            sf         = float(np.random.uniform(lo, hi))
            render_mpp = mpp / sf
            fov_um     = render_mpp * vp_width_px

            canvas = np.zeros((h_px, w_px, 3), dtype=np.uint16)
            gt     = _draw_7_spots(canvas, mpp, w_um, h_um, spot_radius_um, min_sep_um)
            chan   = _make_channel(canvas, mpp)

            if len(vp.layers) == 0:
                vp.add_layer(chan).adjust_layer(blend_mode='normal')
            else:
                vp.add_layer(chan, overwrite=True)

            # Margin: spot must be fully inside the viewport — no partial arcs.
            edge_m = float(np.ceil(spot_radius_um / render_mpp)) + 1.0

            result = _cast_viewport(vp, fov_um, w_um, h_um, gt, edge_margin_px=edge_m)
            trials += 1
            if result is None:
                continue

            cx, cy, rot, flip_x, flip_y, img = result

            for px_f, py_f, ox, oy, tx, ty, c_id, dist in _detect_spots(img, vp, gt, edge_margin_px=edge_m):
                log_data.append({
                    'scale_fraction': round(sf, 4),
                    'render_mpp':     round(render_mpp, 4),
                    'fov_um':         round(fov_um, 2),
                    'vp_center_x':    round(cx, 2),
                    'vp_center_y':    round(cy, 2),
                    'vp_rot_deg':     round(rot, 2),
                    'vp_flip_x':      flip_x,
                    'vp_flip_y':      flip_y,
                    'color_id':       c_id,
                    'screen_px_x':    round(px_f, 2),
                    'screen_px_y':    round(py_f, 2),
                    'obs_um_x':       round(ox, 4),
                    'obs_um_y':       round(oy, 4),
                    'true_um_x':      round(tx, 4),
                    'true_um_y':      round(ty, 4),
                    'error_um':       dist,
                    'error_px':       dist / render_mpp,
                    'spot_diam_px':   (spot_radius_um * 2.0) / render_mpp,
                })
                obs_count += 1

        elapsed = time.time() - start_time
        print(f"trials={trials}  N={obs_count}  ({elapsed:.1f}s)")

    df = pd.DataFrame(log_data)
    df.to_csv(output_csv, index=False)

    elapsed = time.time() - start_time
    print(f"\n[*] Done in {elapsed:.2f}s — {len(df)} observations → {output_csv}")

    if len(df) > 0:
        summary_rows = []
        for sf_bin, grp in df.groupby(pd.cut(df['scale_fraction'], bins=breakpoints)):
            if len(grp) == 0:
                continue
            summary_rows.append({
                'scale_range':  str(sf_bin),
                'mean_sf':      round(float(grp['scale_fraction'].mean()), 4),
                'N':            len(grp),
                'mean_um':      grp['error_um'].mean(),
                'med_um':       grp['error_um'].median(),
                'p95_um':       grp['error_um'].quantile(0.95),
                'max_um':       grp['error_um'].max(),
                'mean_px':      grp['error_px'].mean(),
                'spot_diam_px': grp['spot_diam_px'].median(),
            })
        summary = pd.DataFrame(summary_rows)
        summary_csv = output_csv.replace('.csv', '_summary.csv')
        summary.to_csv(summary_csv, index=False, float_format='%.6f')
        print(f"  Summary → {summary_csv}")

    return df


# =============================================================================
# VARIANT 2: HIGH-RES 1:1 TEST  →  spatial heatmap of error across canvas
# =============================================================================

def run_highres_test(
    mpp            : float = 0.33,
    w_px           : int   = 8192,
    h_px           : int   = 8192,
    spot_radius_um : float = 0.66,   # ~2px at native res for stable centroid
    min_sep_um     : float = 100.0,
    vp_width_px    : int   = 512,
    min_obs        : int   = 100,
    max_trials     : int   = 5000,
    output_csv     : str   = "viewport_highres.csv",
) -> pd.DataFrame:
    """
    1:1 resolution test. Each trial generates a fresh canvas with 7 randomly
    placed spots so observations accumulate across ALL canvas regions over time.
    This builds a spatial heatmap that reveals any position-dependent errors
    (edge effects, axis proximity, corner artefacts, etc.).

    Run with min_obs=1000 overnight for a publication-quality heatmap.
    """
    print("==============================================")
    print(" SPACKLE: High-Res 1:1 Spatial Heatmap Test")
    print("==============================================")

    w_um   = w_px * mpp
    h_um   = h_px * mpp
    fov_um = vp_width_px * mpp          # 1:1: render_mpp == source_mpp

    print(f"[*] Canvas: {w_px}x{h_px} px  ({w_um:.0f}x{h_um:.0f} µm)  mpp={mpp}")
    print(f"[*] Spots:  7 per trial, r={spot_radius_um}µm, min_sep={min_sep_um}µm")
    print(f"[*] FOV at 1:1: {fov_um:.1f}µm  Target: N≥{min_obs}\n")

    vp = ViewPort.create(target_width_px=vp_width_px,
                         center_um=(w_um / 2, h_um / 2), fov_um=(fov_um, fov_um))

    # At 1:1, render_mpp == mpp — compute margin once
    edge_m     = float(np.ceil(spot_radius_um / mpp)) + 1.0

    log_data   = []
    start_time = time.time()
    trials = obs_count = blank_trials = 0

    while obs_count < min_obs and trials < max_trials:
        # Fresh canvas and spots every trial
        canvas = np.zeros((h_px, w_px, 3), dtype=np.uint16)
        gt     = _draw_7_spots(canvas, mpp, w_um, h_um, spot_radius_um, min_sep_um)
        chan   = _make_channel(canvas, mpp)

        if len(vp.layers) == 0:
            vp.add_layer(chan).adjust_layer(blend_mode='normal')
        else:
            vp.add_layer(chan, overwrite=True)

        result = _cast_viewport(vp, fov_um, w_um, h_um, gt, edge_margin_px=edge_m)
        trials += 1
        if result is None:
            blank_trials += 1
            continue

        cx, cy, rot, flip_x, flip_y, img = result

        for px_f, py_f, ox, oy, tx, ty, c_id, dist in _detect_spots(img, vp, gt, edge_margin_px=edge_m):
            log_data.append({
                'render_mpp':   mpp,
                'vp_center_x':  round(cx, 2),
                'vp_center_y':  round(cy, 2),
                'vp_rot_deg':   round(rot, 2),
                'vp_flip_x':    flip_x,
                'vp_flip_y':    flip_y,
                'color_id':     c_id,
                'screen_px_x':  round(px_f, 2),
                'screen_px_y':  round(py_f, 2),
                'obs_um_x':     round(ox, 4),
                'obs_um_y':     round(oy, 4),
                'true_um_x':    round(tx, 4),    # heatmap x axis
                'true_um_y':    round(ty, 4),    # heatmap y axis
                'error_um':     dist,
                'error_px':     dist / mpp,
            })
            obs_count += 1

        if trials % 20 == 0:
            elapsed = time.time() - start_time
            print(f"   ... {trials} trials  N={obs_count}  "
                  f"blank={blank_trials}  ({elapsed:.1f}s)")

    df = pd.DataFrame(log_data)
    df.to_csv(output_csv, index=False)

    elapsed = time.time() - start_time
    print(f"\n[*] Done in {elapsed:.2f}s — {len(df)} observations → {output_csv}")
    print(f"    Total trials: {trials}  Blank (no spot in FOV): {blank_trials}")

    if len(df) > 0:
        print(f"\n  Mean  : {df['error_um'].mean():.6f} µm")
        print(f"  Median: {df['error_um'].median():.6f} µm")
        print(f"  95th% : {df['error_um'].quantile(0.95):.6f} µm")
        print(f"  Max   : {df['error_um'].max():.6f} µm")

        summary_csv = output_csv.replace('.csv', '_summary.csv')
        pd.DataFrame([{
            'test':        '1:1_highres',
            'mpp':         mpp,
            'spot_r_um':   spot_radius_um,
            'min_sep_um':  min_sep_um,
            'trials':      trials,
            'blank':       blank_trials,
            'N':           len(df),
            'mean_um':     df['error_um'].mean(),
            'med_um':      df['error_um'].median(),
            'p95_um':      df['error_um'].quantile(0.95),
            'max_um':      df['error_um'].max(),
            'mean_px':     df['error_px'].mean(),
        }]).to_csv(summary_csv, index=False, float_format='%.6f')
        print(f"  Summary → {summary_csv}")

    return df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_scale_sweep(
        min_trials=10,
        min_obs=100,
        output_csv="viewport_scale_sweep.csv"
    )

    run_highres_test(
        spot_radius_um=0.66,
        min_obs=100,
        output_csv="viewport_highres.csv"
    )
