#!/usr/bin/env python3
"""
summarize_robot_runs.py — robot runs -> protocol metrics (report §5.1.2).

    python3 robot-integration/tools/summarize_robot_runs.py [--run-root ~/nao_robot_runs]
        [--label PPO_robot] [--out robot_results.csv] [--include-dry-run]

Reads every <run>/run.json and writes one row per repetition in the format of
experiments/test_evaluation_script/external_results_template.csv, so robot
results can be merged with the simulation results by summarize_results.py.
F_S and the fall rate use the SAME functions as the simulation
(experiments/test_evaluation_script/metrics.py).

Distance per run: the operator's measurement (tools/record_measurement.py)
if present; otherwise D for a success and the odometry estimate for a fall or
timeout (flagged in the 'distance_source' column, and counted in the summary).
Dry runs and aborted runs are not repetitions and are skipped (aborted runs
are listed so they can be repeated).
"""
import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location(
    "metrics", REPO_ROOT / "experiments/test_evaluation_script/metrics.py")
metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metrics)

COLUMNS = ["label", "delta_deg", "repetition", "target_distance_m", "outcome", "success", "fell",
           "walk_time_s", "distance_m", "lateral_left_m", "heading_drift_deg"]


def run_row(meta):
    m = meta.get("measured") or {}
    outcome = m.get("outcome_override") or meta.get("outcome")
    D = float(meta["target_distance_m"])
    if m.get("distance_m") is not None:
        dist, src = float(m["distance_m"]), "measured"
    elif outcome == "success":
        dist, src = D, "course"
    else:
        dist, src = float(meta.get("odometry_forward_m", 0.0)), "odometry"
    lat = m.get("lateral_left_m")
    lat = float(lat) if lat is not None else float(meta.get("odometry_lateral_m", 0.0))
    return {
        "label": meta["label"], "delta_deg": meta["delta_deg"], "repetition": meta["repetition"],
        "target_distance_m": D, "outcome": outcome, "success": outcome == "success",
        "fell": outcome == "fall", "walk_time_s": float(meta.get("walk_time_s", 0.0)),
        "distance_m": dist, "lateral_left_m": lat,
        "heading_drift_deg": metrics.heading_drift_deg(dist, lat),
        "distance_source": src, "run_dir": meta["_dir"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", type=Path, default=Path("~/nao_robot_runs"))
    ap.add_argument("--label")
    ap.add_argument("--out", type=Path, default=Path("robot_results.csv"))
    ap.add_argument("--include-dry-run", action="store_true")
    a = ap.parse_args()

    rows, aborted = [], []
    for p in sorted(a.run_root.expanduser().glob("*/run.json")):
        meta = json.loads(p.read_text())
        meta["_dir"] = str(p.parent)
        if a.label and meta.get("label") != a.label:
            continue
        if meta.get("dry_run") and not a.include_dry_run:
            continue
        r = run_row(meta)
        (aborted if r["outcome"] in ("aborted", None) else rows).append(r)

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS + ["distance_source", "run_dir"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} repetitions -> {a.out}")

    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], float(r["delta_deg"]))].append(r)
    if groups:
        print(f"\n{'label':20s} {'δ[°]':>5s} {'N':>3s} {'done':>4s} {'TC[%]':>6s} "
              f"{'F_S [m/s]':>16s} {'time [s]':>14s} {'lateral [m]':>15s}")
    for (label, delta), rs in sorted(groups.items()):
        fs = [metrics.fs_index(r["distance_m"], r["walk_time_s"], r["success"], r["target_distance_m"])
              for r in rs]
        fs_m, fs_s = metrics.mean_std(fs)
        t_m, t_s = metrics.mean_std([r["walk_time_s"] for r in rs])
        l_m, l_s = metrics.mean_std([r["lateral_left_m"] for r in rs])
        reps = sorted(int(r["repetition"]) for r in rs)
        dup = len(reps) != len(set(reps))
        fmt = lambda m, s: f"{m:.3f}" + ("" if math.isnan(s) else f" ± {s:.3f}")   # noqa: E731
        print(f"{label:20s} {delta:5g} {len(rs):3d} {sum(r['success'] for r in rs):4d} "
              f"{metrics.fall_rate_percent(r['fell'] for r in rs):6.0f} {fmt(fs_m, fs_s):>16s} "
              f"{fmt(t_m, t_s):>14s} {fmt(l_m, l_s):>15s}"
              f"{'  (duplicate repetitions!)' if dup else ''}")
    est = [r for r in rows if r["distance_source"] == "odometry"]
    if est:
        print(f"\n{len(est)} run(s) use the ODOMETRY estimate as distance (no tape measurement):")
        for r in est:
            print(f"   {r['run_dir']}")
    if aborted:
        print(f"\n{len(aborted)} aborted run(s), not counted (repeat them):")
        for r in aborted:
            print(f"   {r['run_dir']}")


if __name__ == "__main__":
    main()
