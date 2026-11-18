#!/usr/bin/env python3
"""
evaluate_policy.py — watch / evaluate a trained PGRL policy (test_28).

Loads θ from runs/<run-name>/state.json (or the initial parameters with
--initial) and runs episodes with the same measurement as training and as
the common evaluation script (1 m course, F_S, fall detection).

Usage (from this folder, Isaac Sim venv active):
  python evaluate_policy.py                          # GUI, real time, pgrl_v1
  python evaluate_policy.py --episodes 3 --hold-open
  python evaluate_policy.py --initial                # base parameters (= test_27)
  python evaluate_policy.py --headless --episodes 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(line_buffering=True)   # SimulationApp.close() hard-exits

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pgrl_params as P  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="pgrl_v1")
    p.add_argument("--initial", action="store_true",
                   help="use the initial parameters instead of the trained θ")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--distance", type=float, default=1.0, help="course length D [m]")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--fast", action="store_true",
                   help="GUI as fast as possible (default: real time)")
    p.add_argument("--hold-open", action="store_true",
                   help="keep the GUI open after the last episode until it is closed")
    return p.parse_args()


def main():
    args = parse_args()
    if args.initial:
        specs = P.PARAM_SPECS
        theta, label = P.initial_theta(specs), "initial parameters"
    else:
        state_path = HERE / "runs" / args.run_name / "state.json"
        if not state_path.is_file():
            sys.exit(f"No trained policy at {state_path}")
        state = json.loads(state_path.read_text())
        theta = np.array(state["theta"])
        specs = P.load_run_specs(state_path.parent)   # ranges used by that run
        label = f"{args.run_name} (iteration {state['iteration']}, {state['evals']} evals)"

    print("=" * 70)
    print(f"PGRL policy: {label}")
    for n, v in P.as_dict(theta, specs).items():
        print(f"  {n:16s} {v:+.3f}")
    print("=" * 70)

    from pgrl_episode import make_env, evaluate_theta
    gui = not args.headless
    env = make_env("human" if gui else "headless")

    if gui and not args.fast:                      # pace the GUI at real time
        step = env.step

        def paced_step(action=None):
            t0 = time.time()
            out = step(action)
            time.sleep(max(0.0, env.control_dt - (time.time() - t0)))
            return out
        env.step = paced_step

    results = []
    for ep in range(args.episodes):
        res = evaluate_theta(env, phys=P.to_physical(theta, specs), distance_m=args.distance)
        results.append(res)
        print(f"ep {ep + 1}: {res['outcome']:<12} d={res['distance_m']:+.3f} m  "
              f"t={res['walk_time_s']:.2f} s  lat={res['lateral_left_m']:+.3f} m  "
              f"v={res['mean_speed_mps']:.3f} m/s  F_S={res['fs']:+.4f}")

    if len(results) > 1:
        fs = [r["fs"] for r in results]
        print(f"F_S mean {np.mean(fs):+.4f}, falls {sum(r['fell'] for r in results)}/{len(results)}")

    if args.hold_open and gui:
        print("GUI held open — close the window (or Ctrl+C) to exit.")
        try:
            while env._sim_app.is_running():
                env._sim_app.update()
        except KeyboardInterrupt:
            pass
    env.close()


if __name__ == "__main__":
    main()
