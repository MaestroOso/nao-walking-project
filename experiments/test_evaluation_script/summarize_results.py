#!/usr/bin/env python3
"""
summarize_results.py — Aggregate evaluation runs into Table 5-3 and Figure 5-3
==============================================================================

Reads every ``episodes_<label>.csv`` written by evaluate_algorithms.py (plus
any external CSVs given with ``--extra``, e.g. the manufacturer's walking
routine recorded outside Isaac Sim), and produces:

  * ``summary.csv`` — one row per (configuration, D, delta): N, F_S mean/std,
    fall rate TC, success count, and the auxiliary metrics (time, distance,
    lateral deviation, heading drift), all as mean/std.
  * ``robustness.png`` — Figure 5-3: fall rate and F_S versus delta, for every
    configuration evaluated at two or more delta values.

F_S is RECOMPUTED here from distance/time/success with metrics.fs_index(), so
simulator runs and externally recorded runs are scored identically.

Does not need Isaac Sim. Usage:
    python summarize_results.py
    python summarize_results.py --extra manufacturer_results.csv
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from metrics import fs_index, heading_drift_deg, fall_rate_percent, mean_std  # noqa: E402

REQUIRED = ["label", "delta_deg", "repetition", "target_distance_m",
            "success", "fell", "walk_time_s", "distance_m"]

# Fixed colour per configuration (colour follows the entity, never its rank).
# Validated categorical palette, slots 1-5; markers/line styles add a second,
# non-colour encoding so series stay distinguishable in grayscale print.
SERIES_STYLE = {
    "PPO":  ("#2a78d6", "o", "-"),
    "SAC":  ("#eb6834", "s", "--"),
    "PGRL": ("#1baf7a", "^", "-."),
}
FALLBACK_STYLES = [("#eda100", "D", ":"), ("#e87ba4", "v", "-")]


def to_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
            if missing:
                sys.exit(f"[summary] {path} is missing columns: {missing}")
            for raw in reader:
                r = {
                    "label": raw["label"].strip(),
                    "delta_deg": to_float(raw["delta_deg"]),
                    "target_distance_m": to_float(raw["target_distance_m"]),
                    "success": to_bool(raw["success"]),
                    "fell": to_bool(raw["fell"]),
                    "walk_time_s": to_float(raw["walk_time_s"]),
                    "distance_m": to_float(raw["distance_m"]),
                    "lateral_left_m": to_float(raw.get("lateral_left_m")),
                    "outcome": (raw.get("outcome") or "").strip(),
                    "source": str(path),
                }
                hd = to_float(raw.get("heading_drift_deg"))
                if math.isnan(hd) and not math.isnan(r["lateral_left_m"]):
                    hd = heading_drift_deg(r["distance_m"], r["lateral_left_m"])
                r["heading_drift_deg"] = hd
                r["fs"] = fs_index(r["distance_m"], r["walk_time_s"], r["success"],
                                   r["target_distance_m"])
                old_fs = to_float(raw.get("fs"))
                if not math.isnan(old_fs) and abs(old_fs - r["fs"]) > 1e-6:
                    print(f"[summary] note: {path} {r['label']} rep {raw['repetition']}: "
                          f"stored F_S {old_fs:.5f} != recomputed {r['fs']:.5f} (using recomputed)")
                rows.append(r)
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["target_distance_m"], r["delta_deg"])].append(r)

    summary = []
    for (label, dist, delta), sub in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        out = {"label": label, "target_distance_m": dist, "delta_deg": delta, "n": len(sub),
               "successes": sum(r["success"] for r in sub),
               "falls": sum(r["fell"] for r in sub),
               "fall_rate_pct": fall_rate_percent(r["fell"] for r in sub)}
        for key in ("fs", "walk_time_s", "distance_m", "lateral_left_m", "heading_drift_deg"):
            out[f"{key}_mean"], out[f"{key}_std"] = mean_std(r[key] for r in sub)
        summary.append(out)
    return summary


def print_table(summary):
    def f(x, spec):
        return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)

    print(f"\n{'Configuration':<16} {'D[m]':>5} {'δ[°]':>5} {'N':>3} {'F_S [m/s]':>17} "
          f"{'TC[%]':>6} {'t [s]':>13} {'d [m]':>13} {'lat [m]':>14} {'head [°]':>13}")
    print("-" * 118)
    for s in summary:
        print(f"{s['label']:<16} {s['target_distance_m']:>5.2f} {s['delta_deg']:>5.1f} {s['n']:>3} "
              f"{f(s['fs_mean'], '>8.4f')} ± {f(s['fs_std'], '<6.4f')} "
              f"{s['fall_rate_pct']:>6.1f} "
              f"{f(s['walk_time_s_mean'], '>6.2f')} ± {f(s['walk_time_s_std'], '<4.2f')} "
              f"{f(s['distance_m_mean'], '>6.3f')} ± {f(s['distance_m_std'], '<4.2f')} "
              f"{f(s['lateral_left_m_mean'], '>7.3f')} ± {f(s['lateral_left_m_std'], '<4.2f')} "
              f"{f(s['heading_drift_deg_mean'], '>6.1f')} ± {f(s['heading_drift_deg_std'], '<4.1f')}")


def plot_robustness(summary, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_label = defaultdict(list)
    for s in summary:
        by_label[s["label"]].append(s)
    # Only configurations evaluated at >= 2 deltas belong on a robustness curve
    # (the manufacturer's routine is only available at delta = 0).
    series = {k: sorted(v, key=lambda s: s["delta_deg"]) for k, v in by_label.items()
              if len({s["delta_deg"] for s in v}) >= 2}
    if not series:
        print("[summary] no configuration has >= 2 delta values — skipping plot")
        return

    ink, muted, grid = "#2b2b2b", "#6b6b6b", "#e6e6e6"
    fig, (ax_tc, ax_fs) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    fallback = iter(FALLBACK_STYLES)
    for label, pts in series.items():
        color, marker, ls = SERIES_STYLE.get(label.upper()) or next(fallback, ("#6b6b6b", "x", "-"))
        x = [p["delta_deg"] for p in pts]
        ax_tc.plot(x, [p["fall_rate_pct"] for p in pts], color=color, marker=marker,
                   linestyle=ls, linewidth=2, markersize=7, label=label)
        fs_m = [p["fs_mean"] for p in pts]
        fs_s = [0.0 if math.isnan(p["fs_std"]) else p["fs_std"] for p in pts]
        ax_fs.errorbar(x, fs_m, yerr=fs_s, color=color, marker=marker, linestyle=ls,
                       linewidth=2, markersize=7, capsize=3, elinewidth=1, label=label)

    deltas = sorted({p["delta_deg"] for pts in series.values() for p in pts})
    for ax, title, ylabel in ((ax_tc, "Fall rate", "TC [%]"),
                              (ax_fs, "Speed index (mean ± std)", "$F_S$ [m/s]")):
        ax.set_title(title, color=ink, fontsize=11, loc="left")
        ax.set_xlabel("Initial-pose perturbation δ [°]", color=muted)
        ax.set_ylabel(ylabel, color=muted)
        ax.set_xticks(deltas)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted)
    ax_tc.set_ylim(-5, 105)
    ax_tc.legend(frameon=False, labelcolor=ink)

    fig.savefig(out_path, dpi=200)
    print(f"[summary] figure: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Aggregate NAO walking evaluation results")
    p.add_argument("--results-dir", default=str(SCRIPT_DIR / "results"))
    p.add_argument("--extra", nargs="*", default=[],
                   help="Additional CSVs (same columns, see external_results_template.csv)")
    p.add_argument("--out", default=None, help="Summary CSV (default: <results-dir>/summary.csv)")
    p.add_argument("--plot", default=None, help="Figure (default: <results-dir>/robustness.png)")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    results_dir = Path(args.results_dir).resolve()
    paths = sorted(results_dir.glob("episodes_*.csv")) + [Path(e).resolve() for e in args.extra]
    if not paths:
        sys.exit(f"[summary] no episodes_*.csv in {results_dir} and no --extra files")
    print("[summary] inputs:\n  " + "\n  ".join(str(p_) for p_ in paths))

    rows = load_rows(paths)
    distances = sorted({r["target_distance_m"] for r in rows})
    if len(distances) > 1:
        print(f"[summary] WARNING: results mix several course lengths {distances} — "
              f"F_S is only comparable at the same D")

    summary = summarize(rows)
    print_table(summary)

    out_csv = Path(args.out) if args.out else results_dir / "summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n[summary] table: {out_csv}")

    if not args.no_plot:
        plot_robustness(summary, Path(args.plot) if args.plot else results_dir / "robustness.png")


if __name__ == "__main__":
    main()
