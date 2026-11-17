"""
metrics.py — Evaluation metrics for the NAO walking comparison (report §5.1.2)
==============================================================================

Pure-Python / NumPy only: no Isaac Sim or Stable-Baselines3 imports, so this
module can be used both by `evaluate_algorithms.py` (inside the Isaac Sim
process) and by `summarize_results.py` (plain Python, after the fact).

Keeping the metric definitions in ONE place guarantees that results recorded
in the simulator and results recorded elsewhere (e.g. the manufacturer's
walking routine, merged in through an external CSV) are scored with exactly
the same formulas.
"""

import math

import numpy as np


def fs_index(distance_m, walk_time_s, success, target_distance_m):
    """Speed-based fitness index F_S (report Eq. 4-5, from [14]).

        F_S = D / t                      if the course was completed
        F_S = (d / t) * 0.5 * (|d| / D)  otherwise

    d = distance walked along the forward direction (may be NEGATIVE if the
        robot walked backwards — kept negative on purpose, see below),
    t = walking time (seconds since the end of the warm-up),
    D = target distance of the course.

    Sign handling (project decision): the report's formula multiplies d/t by
    d/D. For a negative d that product would be POSITIVE, i.e. walking
    backwards would be rewarded. Using |d| in the fraction-completed factor
    keeps the sign of the speed term, so a backwards walk scores negative.
    For d >= 0 this is identical to the report's formula.

    A run that never got past the warm-up (t == 0) scores 0.
    """
    if walk_time_s <= 0.0:
        return 0.0
    if success:
        return target_distance_m / walk_time_s
    return (distance_m / walk_time_s) * 0.5 * (abs(distance_m) / target_distance_m)


def heading_drift_deg(forward_m, lateral_left_m, min_displacement_m=0.02):
    """Direction of travel relative to the intended forward direction [deg].

    Computed from the robot's DISPLACEMENT (start -> end of the run), not from
    its orientation quaternion, so it does not depend on the quaternion
    convention used in the environment.

    Positive = drifted towards the robot's left. Returns NaN when the robot
    barely moved (direction undefined).
    """
    if math.hypot(forward_m, lateral_left_m) < min_displacement_m:
        return float("nan")
    return math.degrees(math.atan2(lateral_left_m, forward_m))


def fall_rate_percent(fell_flags):
    """Fall rate TC = 100 * n_falls / N (report Eq. 5-1)."""
    fell_flags = list(fell_flags)
    if not fell_flags:
        return float("nan")
    return 100.0 * sum(bool(f) for f in fell_flags) / len(fell_flags)


def mean_std(values):
    """Mean and SAMPLE standard deviation (ddof=1), ignoring NaNs.

    Sample std is the right estimator for N=10 repetitions; returns std=NaN
    when fewer than two finite values are available.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
    return float(np.mean(arr)), std
