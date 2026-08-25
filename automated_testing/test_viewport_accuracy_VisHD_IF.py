import numpy as np
import pandas as pd
import cv2
import time
from scipy.spatial import KDTree
from typing import Dict, List, Tuple

# platpy imports
from channels.image_channel import ImageChannel
from transform_utils import SpatialMetadata, create_transform_matrix
from viewport import ViewPort


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def _color_id(r: bool, g: bool, b: bool) -> int:
    """Map RGB boolean flags to 1-7 (0 = background, skip)."""
    return (1 if r else 0) + (2 if g else 0) + (4 if b else 0)


def _build_test_canvas(
    mpp: float, w_px: int, h_px: int,
    spot_radius_um: float, spacing_um: float
) -> Tuple[ImageChannel, Dict, Dict, int]:
    """
    Generate a synthetic 16-bit RGB canvas with a 7-colour spot grid.
    Returns the ImageChannel, ground-truth dict, KDTree dict, and total spot count.
    """
    MAX_VAL   = 65535
    color_map: Dict[int, Tuple[int, int, int]] = {
        1: (MAX_VAL,       0,       0),
        2: (      0, MAX_VAL,       0),
        3: (MAX_VAL, MAX_VAL,       0),
        4: (      0,       0, MAX_VAL),
        5: (MAX_VAL,       0, MAX_VAL),
        6: (      0, MAX_VAL, MAX_VAL),
        7: (MAX_VAL, MAX_VAL, MAX_VAL),
    }
    spot_r_px  = max(1, int(spot_radius_um / mpp))
    spacing_px = max(spot_r_px * 4, int(spacing_um / mpp))
    canvas     = np.zeros((h_px, w_px, 3), dtype=np.uint16)
    ground_truth: Dict[int, List[Tuple[float, float]]] = {cid: [] for cid in color_map}

    for y in range(spacing_px, h_px - spacing_px, spacing_px):
        for x in range(spacing_px, w_px - spacing_px, spacing_px):
            c_id = 1 + ((x // spacing_px + y // spacing_px) % 7)
            cv2.circle(canvas, (x, y), spot_r_px, color_map[c_id], -1)
            ground_truth[c_id].append((x * mpp, y * mpp))

    trees     = {cid: KDTree(pts) for cid, pts in ground_truth.items() if pts}
    n_spots   = sum(len(v) for v in ground_truth.values())
    chan      = ImageChannel(
        name="synthetic_grid",
        image=canvas,
        spatial_meta=SpatialMetadata(
            transform_matrix=create_transform_matrix(mpp, mpp),
            local_bounds=(0.0, float(w_px), 0.0, float(h_px)),
            unit="microns"
        )
    )
    return chan, ground_truth, trees, n_spots


def _detect_spots(
    img: np.ndarray, vp: ViewPort,
    trees: Dict, ground_truth: Dict
) -> List[Tuple]:
    """
    Find lit blobs in a rendered RGBA image, identify by colour,
    map centroids back to microns, match to nearest ground-truth spot.
    Returns list of (px_f, py_f, obs_x, obs_y, true_x, true_y, c_id, error_um).
    """
    bright = (np.max(img[..., :3], axis=2) > 50).astype(np.uint8) * 255
    n_lab, _, _, centroids = cv2.connectedComponentsWithStats(bright)
    results = []
    for j in range(1, n_lab):
        px_f, py_f = centroids[j]
        px_i = int(np.clip(round(px_f), 0, img.shape[1] - 1))
        py_i = int(np.clip(round(py_f), 0, img.shape[0] - 1))
        rgb  = img[py_i, px_i, :3]
        c_id = _color_id(rgb[0] > 100, rgb[1] > 100, rgb[2] > 100)
        if c_id == 0 or c_id not in trees:
            continue
        obs_um       = vp.pixels_to_microns(np.array([[px_f, py_f]]))[0]
        obs_x, obs_y = float(obs_um[0]), float(obs_um[1])
        dist, idx    = trees[c_id].query((obs_x, obs_y))
        tx, ty       = ground_truth[c_id][idx]
        results.append((px_f, py_f, obs_x, obs_y, tx, ty, c_id, dist))
    return results


# =============================================================================
# VARIANT 1: SCALE SWEEP  (scatterplot data: error vs scale fraction)
# =============================================================================

def run_scale_sweep(
    mpp            : float = 0.33,  # 20k x 20k @ 0.33 mpp is ~ size of VisHD IF image
    w_px           : int   = 20000,
    h_px           : int   = 20000,
    spot_radius_um : float = 2.0,
    spacing_um     : float = 150.0,
    vp_width_px    : int   = 512,
    min_trials     : int   = 250,    # minimum renders before a range can finish  ACTUAL VALUE SET AT THE BOTTOM
    min_obs        : int   = 1000,   # minimum spot observations before a range finishes
    max_trials     : int   = 5000,   # safety cap per range
    output_csv     : str   = "viewport_scale_sweep.csv",
) -> pd.DataFrame:
    """
    Sweep scale_fraction from 1.0 (1:1) down to ~0.05 (1:20) in 0.05-wide bands.
    Each band iterates until BOTH min_trials and min_obs are satisfied.
    scale_fraction = source_mpp / render_mpp  (1.0 = native, 0.05 = 20x downscale)
    """
    print("==============================================")
    print(" platpy: Scale Sweep Accuracy Test")
    print("==============================================")

    w_um = w_px * mpp
    h_um = h_px * mpp

    print(f"[*] Building {w_px}x{h_px} canvas  spot_r={spot_radius_um}µm  spacing={spacing_um}µm")
    chan, gt, trees, n_spots = _build_test_canvas(mpp, w_px, h_px, spot_radius_um, spacing_um)
    print(f"[*] {n_spots} ground-truth spots placed.")

    vp = ViewPort.create(target_width_px=vp_width_px,
                         center_um=(w_um / 2, h_um / 2), fov_um=(500.0, 500.0))
    vp.add_layer(chan).adjust_layer(blend_mode='normal')

    # Build scale ranges (0.95-1.0), (0.90-0.95), ..., stopping where FOV > canvas
    min_valid_sf = (mpp * vp_width_px) / (min(w_um, h_um) * 0.90)
    breakpoints  = np.round(np.arange(0.05, 1.01, 0.05), 2)
    raw_ranges   = list(zip(breakpoints[:-1], breakpoints[1:]))[::-1]  # high→low
    scale_ranges = [(max(float(lo), round(min_valid_sf, 3)), float(hi))
                    for lo, hi in raw_ranges if float(hi) > min_valid_sf]

    if not scale_ranges:
        print("[!] Canvas too small for any valid scale range.")
        return pd.DataFrame()

    print(f"[*] {len(scale_ranges)} scale bands  "
          f"(stop per band: trials≥{min_trials} AND N≥{min_obs})\n")

    log_data   = []
    start_time = time.time()

    for lo, hi in scale_ranges:
        trials = obs_count = 0
        label  = f"[{lo:.2f}, {hi:.2f}]"
        print(f"  {label}  ", end='', flush=True)

        while not (trials >= min_trials and obs_count >= min_obs):
            if trials >= max_trials:
                print(f"  [!] max_trials={max_trials} reached.")
                break

            sf         = float(np.random.uniform(lo, hi))
            render_mpp = mpp / sf
            fov_um     = render_mpp * vp_width_px
            margin     = fov_um / 2.0 + spacing_um

            if margin * 2.0 >= min(w_um, h_um):
                trials += 1
                continue

            cx     = float(np.random.uniform(margin, w_um - margin))
            cy     = float(np.random.uniform(margin, h_um - margin))
            rot    = float(np.random.uniform(-180, 180))
            flip_x = bool(np.random.choice([True, False]))
            flip_y = bool(np.random.choice([True, False]))

            vp.update_view(center_um=(cx, cy), fov_um=(fov_um, fov_um),
                           rotation_deg=rot, flip_x=flip_x, flip_y=flip_y)
            img = np.asarray(vp.render().image)

            for px_f, py_f, ox, oy, tx, ty, c_id, dist in _detect_spots(img, vp, trees, gt):
                log_data.append({
                    'scale_fraction': round(sf, 4),
                    'render_mpp':     round(render_mpp, 4),
                    'fov_um':         round(fov_um, 2),
                    'vp_rot_deg':     round(rot, 2),
                    'vp_flip_x':      flip_x,
                    'vp_flip_y':      flip_y,
                    'color_id':       c_id,
                    'screen_px_x':    px_f,
                    'screen_px_y':    py_f,
                    'obs_um_x':       ox,
                    'obs_um_y':       oy,
                    'true_um_x':      tx,
                    'true_um_y':      ty,
                    'error_um':       dist,
                    'error_px':       dist / render_mpp,
                    'spot_diam_px':   (spot_radius_um * 2.0) / render_mpp,
                })
                obs_count += 1

            trials += 1

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
# VARIANT 2: HIGH-RES 1:1 TEST  (spatial heatmap data)
# =============================================================================

def run_highres_test(
    mpp            : float = 0.33,    # 20k x 20k @ 0.33 mpp is ~ size of VisHD IF image
    w_px           : int   = 20000,
    h_px           : int   = 20000,
    spot_radius_um : float = 0.66,  # ~2 pixels — stable centroid at native res
    spacing_um     : float = 50.0,  # tighter grid: more obs per view at 1:1
    vp_width_px    : int   = 512,
    min_obs        : int   = 1000,    ### ACTUAL VALUE SET AT THE BOTTOM
    max_trials     : int   = 2000000,
    output_csv     : str   = "viewport_highres.csv",
) -> pd.DataFrame:
    """
    1:1 resolution test — fixed scale, random position/rotation/flip.
    Runs until min_obs observations collected.
    Logs full spatial position (obs_um_x/y) for heatmap analysis: does any
    region of the canvas have systematically higher error?
    """
    print("==============================================")
    print(" platpy: High-Res 1:1 Accuracy Test")
    print("==============================================")

    w_um   = w_px * mpp
    h_um   = h_px * mpp
    fov_um = vp_width_px * mpp          # 1:1: render_mpp == source_mpp
    margin = fov_um / 2.0 + spacing_um

    print(f"[*] Building canvas at 1:1  spot_r={spot_radius_um}µm  spacing={spacing_um}µm")
    chan, gt, trees, n_spots = _build_test_canvas(mpp, w_px, h_px, spot_radius_um, spacing_um)
    print(f"[*] {n_spots} ground-truth spots.  Target: N≥{min_obs}\n")

    vp = ViewPort.create(target_width_px=vp_width_px,
                         center_um=(w_um / 2, h_um / 2), fov_um=(fov_um, fov_um))
    vp.add_layer(chan).adjust_layer(blend_mode='normal')

    log_data   = []
    start_time = time.time()
    trials = obs_count = 0

    while obs_count < min_obs and trials < max_trials:
        cx     = float(np.random.uniform(margin, w_um - margin))
        cy     = float(np.random.uniform(margin, h_um - margin))
        rot    = float(np.random.uniform(-180, 180))
        flip_x = bool(np.random.choice([True, False]))
        flip_y = bool(np.random.choice([True, False]))

        vp.update_view(center_um=(cx, cy), fov_um=(fov_um, fov_um),
                       rotation_deg=rot, flip_x=flip_x, flip_y=flip_y)
        img = np.asarray(vp.render().image)

        for px_f, py_f, ox, oy, tx, ty, c_id, dist in _detect_spots(img, vp, trees, gt):
            log_data.append({
                'scale_fraction': 1.0,
                'render_mpp':     mpp,
                'fov_um':         fov_um,
                'vp_center_x':    cx,
                'vp_center_y':    cy,
                'vp_rot_deg':     round(rot, 2),
                'vp_flip_x':      flip_x,
                'vp_flip_y':      flip_y,
                'color_id':       c_id,
                'screen_px_x':    px_f,
                'screen_px_y':    py_f,
                'obs_um_x':       ox,
                'obs_um_y':       oy,
                'true_um_x':      tx,
                'true_um_y':      ty,
                'error_um':       dist,
                'error_px':       dist / mpp,
            })
            obs_count += 1

        trials += 1
        if trials % 20 == 0:
            elapsed = time.time() - start_time
            print(f"   ... {trials} renders  N={obs_count}  ({elapsed:.1f}s)")

    df = pd.DataFrame(log_data)
    df.to_csv(output_csv, index=False)

    elapsed = time.time() - start_time
    print(f"\n[*] Done in {elapsed:.2f}s — {len(df)} observations → {output_csv}")

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
            'spacing_um':  spacing_um,
            'trials':      trials,
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
    # *** ACTUAL RUN VALUES ARE SET HERE — check before launching an overnight run ***

    run_scale_sweep(
        min_obs=50,
        min_trials=50,
        max_trials=1000000,
        output_csv="viewport_scale_sweep_VisHD_IF.csv"
    )

    run_highres_test(
        spot_radius_um=0.66,    # ~2px at native res — stable centroid
        spacing_um=50.0,
        min_obs=100,
        max_trials=2000000,
        output_csv="viewport_highres_VisHD_IF.csv"
    )
