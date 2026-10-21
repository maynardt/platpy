"""
benchmarks/lazy_vs_raw.py -- Speed and peak-memory comparison between
platpy's lazy-loading bundle pipeline and a "dumb" straight Python/pandas
script reading equivalent raw data, for the same task: open a dataset and
pull out one small spatial region's transcripts.

Each task runs in its OWN subprocess (not just a fresh function call in
this process) -- Windows/Python don't reliably release freed memory back
to the OS, so measuring two tasks back-to-back in one process would let
the first task's lingering allocations pollute the second's peak reading.
Peak memory is psutil's peak_wset (Windows peak working set) sampled
after the task completes, not an estimate.

Usage:
    python benchmarks/lazy_vs_raw.py                    # run both tasks, print comparison
    python benchmarks/lazy_vs_raw.py --task platpy       # run just the platpy arm (used internally)
    python benchmarks/lazy_vs_raw.py --task raw          # run just the raw arm (used internally)

Edit RAW_DIR / BUNDLE_DIR / ROI_BOX below to point at real data before running.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

# --- Point these at real data before running ---
RAW_DIR = Path(r"L:\lamantia\Xenium Files\LD_WT Adult 7.7.26\output-XETG00181__0074571__LD_FRONTAL__20260702__151815")
BUNDLE_DIR = Path(r"C:\temp\adult_rebuild\platpy_data\XeniumSamples\ADULT_571_LD_FRONTAL_01")
# Both arms MUST query the same absolute physical location -- RAW_DIR covers
# the whole multi-section raw slide, while BUNDLE_DIR is already cropped to
# one section, so each arm deriving its own "center" independently (e.g.
# from its own data's min/max) queries two different places entirely and
# silently produces a meaningless (often zero-row) comparison. These
# absolute microns were read once from the bundle's own transcripts extent;
# re-derive and update them if BUNDLE_DIR changes.
ROI_CENTER_X_UM = 1196.8
ROI_CENTER_Y_UM = 1370.9
ROI_HALF_WIDTH_UM = 200.0
# BUNDLE_DIR's own recorded crop (sample.rois['extraction_bounds'], added by
# today's work) -- the raw_cells task needs this to scope to the SAME
# section, not the whole multi-section raw slide RAW_DIR actually points to.
SECTION_MIN_X_UM, SECTION_MAX_X_UM = 408.685, 3479.566
SECTION_MIN_Y_UM, SECTION_MAX_Y_UM = 40.633, 4478.018


def _report(label: str, t0: float, n_rows: int):
    dt = time.perf_counter() - t0
    peak_mb = psutil.Process().memory_info().peak_wset / 1e6
    print("RESULT " + json.dumps({"label": label, "time_s": dt, "peak_mb": peak_mb, "n_rows": n_rows}))


def run_platpy():
    """Open the bundle lazily, resolve a small ROI via DataPort (block-index
    narrowed), pull out that region's transcripts -- the actual intended
    usage pattern this session's work was built for."""
    t0 = time.perf_counter()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from platpy.io_utils import load_sample
    from platpy.dataport import DataPort

    sample = load_sample(str(BUNDLE_DIR), verbose=False)
    dp = DataPort.from_roi(ROI_CENTER_X_UM - ROI_HALF_WIDTH_UM, ROI_CENTER_X_UM + ROI_HALF_WIDTH_UM,
                            ROI_CENTER_Y_UM - ROI_HALF_WIDTH_UM, ROI_CENTER_Y_UM + ROI_HALF_WIDTH_UM)
    result = dp.extract_omic(sample.omics['transcripts'])
    _report("platpy (lazy bundle + DataPort)", t0, result.n_obs)


def run_platpy_cells():
    """build_cells_xenium() against the save-time cache -- the actual
    intended usage pattern (the cache exists specifically so this doesn't
    recompute on every load)."""
    t0 = time.perf_counter()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from platpy.io_utils import load_sample
    from platpy.cell_utils import build_cells_xenium

    sample = load_sample(str(BUNDLE_DIR), verbose=False)
    adata = build_cells_xenium(sample, verbose=False)
    _report("platpy build_cells_xenium (cached)", t0, adata.n_obs)


def run_raw_cells():
    """The 'dumb' equivalent: read transcripts.parquet fresh and pivot into
    a cells x genes matrix by hand -- what build_cells_xenium() itself did
    before today's save-time caching existed, and what anyone would
    reasonably write without it."""
    t0 = time.perf_counter()
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    df = pd.read_parquet(RAW_DIR / "transcripts.parquet",
                          columns=["cell_id", "feature_name", "x_location", "y_location"])
    df = df.rename(columns={"feature_name": "feature", "x_location": "x", "y_location": "y"})
    # RAW_DIR is the whole multi-section raw slide; scope to the same single
    # section BUNDLE_DIR represents (see SECTION_*_UM's comment above), or
    # this builds cells for a completely different, much larger physical area.
    df = df[
        (df["x"] >= SECTION_MIN_X_UM) & (df["x"] <= SECTION_MAX_X_UM) &
        (df["y"] >= SECTION_MIN_Y_UM) & (df["y"] <= SECTION_MAX_Y_UM)
    ]
    df = df[~df["cell_id"].astype(str).isin(["0", "-1", "UNASSIGNED"])]
    cells = sorted(df["cell_id"].astype(str).unique())
    genes = sorted(df["feature"].unique())
    cell_idx = {c: i for i, c in enumerate(cells)}
    gene_idx = {g: i for i, g in enumerate(genes)}
    rows = df["cell_id"].astype(str).map(cell_idx).values
    cols = df["feature"].map(gene_idx).values
    X = sp.csr_matrix((np.ones(len(df), dtype=np.float32), (rows, cols)),
                       shape=(len(cells), len(genes)))
    _report("raw pandas groupby/pivot (no cache)", t0, X.shape[0])


def run_raw():
    """The 'dumb' equivalent: read the whole raw transcripts.parquet into
    memory with plain pandas (the obvious first thing anyone would write
    without any of this session's infrastructure), then filter by bbox."""
    t0 = time.perf_counter()
    import pandas as pd

    df = pd.read_parquet(RAW_DIR / "transcripts.parquet")
    df = df.rename(columns={"x_location": "x", "y_location": "y"})
    # Same absolute ROI box as the platpy arm -- see ROI_CENTER_*_UM's
    # comment above for why this can't be independently re-derived per arm.
    mask = (
        (df["x"] >= ROI_CENTER_X_UM - ROI_HALF_WIDTH_UM) & (df["x"] <= ROI_CENTER_X_UM + ROI_HALF_WIDTH_UM) &
        (df["y"] >= ROI_CENTER_Y_UM - ROI_HALF_WIDTH_UM) & (df["y"] <= ROI_CENTER_Y_UM + ROI_HALF_WIDTH_UM)
    )
    result = df[mask]
    _report("raw pandas.read_parquet + filter", t0, len(result))


TASKS = {"platpy": run_platpy, "raw": run_raw,
         "platpy_cells": run_platpy_cells, "raw_cells": run_raw_cells}

# Each pair is run and compared together: (platpy arm, raw arm, description).
COMPARISONS = [
    ("platpy", "raw", "Open bundle + query one small ROI's transcripts"),
    ("platpy_cells", "raw_cells", "Build a cells x genes matrix"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASKS.keys()), default=None)
    args = parser.parse_args()

    if args.task is not None:
        TASKS[args.task]()
        return

    # No --task: driver mode -- run every task as its own subprocess (so
    # peak-memory readings never carry over between them), then compare
    # each platpy/raw pair.
    rows = {}
    for task in TASKS:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--task", task],
            capture_output=True, text=True,
        )
        print(proc.stdout.strip())
        if proc.returncode != 0:
            print(f"  [{task} FAILED]\n{proc.stderr}")
            continue
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                rows[task] = json.loads(line[len("RESULT "):])

    print()
    for platpy_task, raw_task, desc in COMPARISONS:
        if platpy_task not in rows or raw_task not in rows:
            continue
        t_p, t_r = rows[platpy_task]["time_s"], rows[raw_task]["time_s"]
        m_p, m_r = rows[platpy_task]["peak_mb"], rows[raw_task]["peak_mb"]
        print(f"--- {desc} ---")
        print(f"  Time:   platpy {t_p:.3f}s  vs  raw {t_r:.3f}s   -> {t_r / t_p:.1f}x faster")
        print(f"  Memory: platpy {m_p:.1f}MB vs  raw {m_r:.1f}MB  -> {m_r / m_p:.1f}x less peak memory")


if __name__ == "__main__":
    main()
