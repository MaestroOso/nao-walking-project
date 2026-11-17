#!/usr/bin/env python3
"""
run_open_loop.py — test_27: play the looped Webots gait OPEN LOOP (zero
policy actions) in the PPO baseline environment and report how far the robot
gets and when / how it falls.

No file outside experiments/test_27_gait_loop/ is modified:
  * NaoWalkingEnv and WebotsReferenceGait are IMPORTED from
    walking-solutions/ppo/baseline_test24/ (read-only; bytecode writing is
    disabled so no .pyc lands in that folder).
  * After the env is built, its gait object is replaced by a
    WebotsReferenceGait pointed at this folder's data/looped_*.npy files.
    Everything else (physics, PD gains, warm-up 60 steps, grace 50 steps,
    termination, 1500-step limit) is exactly the baseline env.
  * With action = 0, the commanded leg targets are exactly q_ref(t).

Usage (from this folder, with the Isaac Sim venv active):
    python build_looped_gait.py           # once
    python run_open_loop.py               # GUI, 1 episode
    python run_open_loop.py --headless --episodes 3
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True   # never write __pycache__ into baseline_test24
# SimulationApp.close() hard-exits the process without flushing Python's
# stdout buffer, so force line buffering or all prints are lost when piped.
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

HERE = Path(__file__).resolve().parent
ENV_DIR = HERE.parents[1] / "walking-solutions" / "ppo" / "baseline_test24"
DATA = HERE / "data"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--slow", action="store_true",
                   help="sleep 1/30 s per step so GUI runs ~real time")
    p.add_argument("--hold-open", action="store_true",
                   help="keep the GUI open after the last episode until Ctrl+C")
    return p.parse_args()


def main():
    args = parse_args()
    for f in ("looped_keyframes.npy", "looped_timestamps.npy", "looped_joint_names.txt"):
        if not (DATA / f).is_file():
            sys.exit(f"Missing {DATA / f} — run build_looped_gait.py first.")

    # The baseline env resolves the URDF path relative to the CWD
    # ("../../shared/robot/nao.urdf"), so run from its folder (read-only).
    os.chdir(ENV_DIR)
    sys.path.insert(0, str(ENV_DIR))
    from nao_walking_env import NaoWalkingEnv
    from webots_reference_gait import WebotsReferenceGait

    env = NaoWalkingEnv(render_mode="headless" if args.headless else "human")

    looped = WebotsReferenceGait(
        keyframes_path=str(DATA / "looped_keyframes.npy"),
        timestamps_path=str(DATA / "looped_timestamps.npy"),
        joint_names_path=str(DATA / "looped_joint_names.txt"),
    )
    diff = np.degrees(np.abs(looped._first_frame - env._leg_home)).max()
    print(f"[test_27] looped gait t=0 vs env home pose: max diff {diff:.3f} deg")
    env._gait = looped

    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    dt = env.control_dt
    warm = env.gait_warmup_steps

    print(f"[test_27] episodes={args.episodes}, control dt={dt:.4f}s, "
          f"warm-up={warm} steps, max={env.max_episode_steps} steps")
    results = []
    for ep in range(args.episodes):
        env.reset()
        start = env._get_base_position().copy()
        warm_pos = None
        step, done, term, trunc = 0, False, False, False
        print(f"\n--- Episode {ep + 1} ---")
        print(f"{'t_gait[s]':>9} {'fwd[m]':>8} {'lat[m]':>8} {'h[m]':>6}")
        while not done:
            _, _, term, trunc, _ = env.step(zero)
            step += 1
            done = term or trunc
            pos = env._get_base_position()
            if step == warm:
                warm_pos = pos.copy()
            if step >= warm and (step - warm) % 30 == 0:   # once per gait second
                ref = warm_pos if warm_pos is not None else start
                print(f"{(step - warm) * dt:9.1f} {-(pos[0] - ref[0]):8.3f} "
                      f"{pos[1] - ref[1]:8.3f} {pos[2]:6.3f}")
            if args.slow and not args.headless:
                time.sleep(dt)

        pos, quat = env._nao.get_world_pose()
        ref = warm_pos if warm_pos is not None else start
        fwd = -(pos[0] - ref[0])
        lat = pos[1] - ref[1]
        t_gait = max(step - warm, 0) * dt
        if trunc and not term:
            cause = "time limit (no fall)"
        elif pos is None or np.any(np.isnan(pos)):
            cause = "NaN"
        elif pos[2] < env.term_height:
            cause = f"fell (h={pos[2]:.3f} m)"
        else:
            cause = f"tilt (quat={np.round(quat, 3).tolist()})"
        results.append((step, t_gait, fwd, lat, cause))
        print(f"[test_27] ep {ep + 1}: steps={step}, gait time={t_gait:.2f}s "
              f"(= {t_gait / 2.64:.1f} gait cycles), forward={fwd:.3f} m, "
              f"lateral={lat:+.3f} m, avg fwd vel={fwd / max(t_gait, 1e-9):.3f} m/s, "
              f"end: {cause}")

    print("\n" + "=" * 70)
    print("SUMMARY test_27 (looped Webots gait, open loop, zero actions)")
    for i, (s, tg, f, l, c) in enumerate(results, 1):
        print(f"  ep {i}: {s:5d} steps  {tg:6.2f}s  fwd {f:6.3f} m  lat {l:+.3f} m  {c}")

    # Persist results BEFORE env.close() (which hard-exits the process).
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(out / f"open_loop_{stamp}.csv", "w") as fh:
        fh.write("episode,steps,gait_time_s,forward_m,lateral_m,end\n")
        for i, (s, tg, f, l, c) in enumerate(results, 1):
            fh.write(f"{i},{s},{tg:.3f},{f:.4f},{l:.4f},\"{c}\"\n")
    print(f"[test_27] results saved to {out / f'open_loop_{stamp}.csv'}")

    if args.hold_open and not args.headless:
        print("\n[test_27] GUI held open — Ctrl+C to exit.")
        try:
            while env._sim_app.is_running():
                env._sim_app.update()
        except KeyboardInterrupt:
            pass
    env.close()


if __name__ == "__main__":
    main()
