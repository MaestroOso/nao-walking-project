#!/usr/bin/env python3
"""
check_controller.py — sanity checks BEFORE training (baseline_test28).

  1. Initial parameters (gains 0, f = a = 1, stiffness 1) must reproduce the
     open-loop run of the looped gait (experiments/test_27_gait_loop: fell at
     ~2 s gait time, ~0.41 m).
  2. Tilt sign: prints g_x, g_y while the robot falls forward. Expected with
     the (w, x, y, z) convention: g ≈ (0, 0) upright, g_x grows POSITIVE when
     the torso pitches forward.
  3. Compensator effect: one episode with each forced ankle-pitch gain
     (default ±0.5) to see that the compensator changes the outcome and in
     which direction.

Usage (from this folder, Isaac Sim venv active):
  python check_controller.py              # headless
  python check_controller.py --gui        # watch it
"""
import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgrl_params as P  # noqa: E402


def traced_episode(env, phys, label, distance=1.0):
    """Like pgrl_episode.evaluate_theta, but also prints the tilt trace."""
    from pgrl_episode import evaluate_theta
    trace = []
    orig = env.compute_leg_targets

    def wrapped():
        q = orig()
        if env._step_count >= env.gait_warmup_steps and \
                (env._step_count - env.gait_warmup_steps) % 8 == 0:
            pos = env._get_base_position()
            trace.append(((env._step_count - env.gait_warmup_steps) * env.control_dt,
                          *env.last_tilt, pos[2], np.abs(env.last_comp).max()))
        return q
    env.compute_leg_targets = wrapped
    try:
        res = evaluate_theta(env, phys=phys, distance_m=distance)
    finally:
        env.compute_leg_targets = orig
    print(f"\n=== {label} ===")
    print(f"{'t_gait':>6} {'g_x':>7} {'g_y':>7} {'h[m]':>6} {'|comp|max[deg]':>14}")
    for t, gx, gy, h, c in trace:
        print(f"{t:6.2f} {gx:+7.3f} {gy:+7.3f} {h:6.3f} {np.degrees(c):14.2f}")
    print(f"-> {res['outcome']}, d={res['distance_m']:+.3f} m, t={res['walk_time_s']:.2f} s, "
          f"lat={res['lateral_left_m']:+.3f} m, F_S={res['fs']:+.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--gain", type=float, default=0.5)
    args = ap.parse_args()

    from pgrl_episode import make_env
    env = make_env("human" if args.gui else "headless")
    cwd = Path.cwd()

    base = P.INIT_PHYS.copy()
    traced_episode(env, base, "1-2. initial parameters (should fall at ~0.41 m)")
    os.chdir(cwd)
    for g in (+args.gain, -args.gain):
        phys = base.copy()
        phys[0] = g                                   # K_ank_pitch
        traced_episode(env, phys, f"3. K_ank_pitch = {g:+.2f}")
        os.chdir(cwd)
    env.close()


if __name__ == "__main__":
    main()
