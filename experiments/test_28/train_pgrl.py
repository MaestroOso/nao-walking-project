#!/usr/bin/env python3
"""
train_pgrl.py — PGRL training (Kohl & Stone [20] as used by Kulk & Welsh [14];
report §4.5.3, Eq. 4-5 and 4-6).

One iteration:
  1. Generate P perturbed policies θ + Δ, Δ_i ∈ {−ε, 0, +ε} uniformly at random
     (normalised parameter space [0, 1]^12).
  2. Run one episode with each (1 m course, δ = 0°, same warm-up / grace /
     fall criteria / 50 s limit as PPO and SAC) → F_S per policy.
  3. For each parameter i, average F_S over the policies where it went up
     (F⁺), stayed (F⁰), went down (F⁻):
         A_i = 0                     if F⁰ > F⁺ and F⁰ > F⁻
         A_i = tanh(F⁺ − F⁻)         otherwise
     (A_i = 0 too if one of the three groups is empty this iteration.)
  4. θ ← clip(θ + η·A/‖A‖, 0, 1)  — step normalised to length η, as in
     Kohl & Stone's original algorithm (option b). Save state + logs.

Resumable / chunked: each process runs at most --iterations iterations, then
saves and exits (Isaac Sim memory growth, report §4.1). Re-running the same
command continues from runs/<run-name>/state.json until --max-evals.

Outputs in runs/<run-name>/:
  state.json        θ, iteration, evaluation count, RNG state (for resume)
  evaluations.csv   one row per episode (perturbation, params, outcome, F_S)
  iterations.csv    one row per iteration (mean F_S, falls, cost, θ) —
                    source for Figure 5-2 and Table 5-2
  config.json       hyper-parameters of the run

Usage (from this folder, Isaac Sim venv active):
  python train_pgrl.py --run-name pgrl_v1 --iterations 10
"""
import argparse
import csv
import json
import os
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
    p.add_argument("--iterations", type=int, default=10,
                   help="iterations to run in THIS process (chunk size)")
    p.add_argument("--max-evals", type=int, default=2000,
                   help="total evaluation budget (cost) for the whole run")
    p.add_argument("--policies", type=int, default=12, help="policies per iteration")
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--distance", type=float, default=1.0, help="course length D [m]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gui", action="store_true")
    return p.parse_args()


EVAL_FIELDS = (["iteration", "k", "eval_index"] + [f"d_{n}" for n in P.NAMES]
               + P.NAMES + ["outcome", "fell", "success", "walk_time_s",
                            "distance_m", "lateral_left_m", "fs", "wall_time_s"])
ITER_FIELDS = (["iteration", "evals_cum", "fs_mean", "fs_max", "fs_min", "n_falls",
                "n_success", "wall_time_s"] + P.NAMES)


def append_csv(path, fields, row):
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def pgrl_update(theta, deltas, scores, eta):
    """Eq. 4-6 for the direction, Kohl & Stone [20] normalisation for the step.

    deltas: (P, n) array of {-ε, 0, +ε}; scores: (P,) F_S.

    A_i is computed per Eq. 4-6 (0 if leaving the parameter unchanged scored
    best, else tanh(F⁺ − F⁻)). The adjustment vector is then rescaled to
    length η, as in Kohl & Stone's original algorithm:  θ ← θ + η·A/‖A‖.
    Without this, F_S differences of ~0.005 m/s give steps of ~1e-4 (tanh is
    linear there), i.e. no learning within the budget (decision 2026-11-17,
    option b). Returns (new θ, applied step).
    """
    A = np.zeros_like(theta)
    for i in range(theta.size):
        up, zero, down = deltas[:, i] > 0, deltas[:, i] == 0, deltas[:, i] < 0
        if not (up.any() and zero.any() and down.any()):
            continue
        f_up, f_zero, f_down = scores[up].mean(), scores[zero].mean(), scores[down].mean()
        if f_zero > f_up and f_zero > f_down:
            continue
        A[i] = np.tanh(f_up - f_down)
    norm = np.linalg.norm(A)
    step = eta * A / norm if norm > 0 else A
    return np.clip(theta + step, 0.0, 1.0), step


def main():
    args = parse_args()
    run_dir = HERE / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"

    if state_path.exists():
        state = json.loads(state_path.read_text())
        theta = np.array(state["theta"])
        specs = P.load_run_specs(run_dir)        # ranges this run was trained with
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]
        print(f"[pgrl] RESUME {args.run_name}: iteration {state['iteration']}, "
              f"{state['evals']} evals")
    else:
        specs = P.PARAM_SPECS
        theta = P.initial_theta(specs)
        rng = np.random.default_rng(args.seed)
        state = {"iteration": 0, "evals": 0}
        (run_dir / "config.json").write_text(json.dumps({
            **vars(args), "param_specs": specs,
            "theta_init": theta.tolist(), "delta_deg": 0.0,
            "compensator": "torso tilt (projected gravity g_x, g_y)",
            "base_gait": "experiments/test_27_gait_loop/data (looped Webots Forwards.motion)",
        }, indent=2))
        print(f"[pgrl] NEW run {args.run_name}")

    if state["evals"] >= args.max_evals:
        print(f"[pgrl] budget reached ({state['evals']}/{args.max_evals}). Nothing to do.")
        return

    from pgrl_episode import make_env, evaluate_theta
    env = make_env("human" if args.gui else "headless")
    cwd = Path.cwd()

    for _ in range(args.iterations):
        if state["evals"] + args.policies > args.max_evals:
            print("[pgrl] budget would be exceeded — stopping.")
            break
        it = state["iteration"] + 1
        t_it = time.time()
        deltas = rng.choice([-args.epsilon, 0.0, args.epsilon],
                            size=(args.policies, P.N_PARAMS))
        scores, falls, succ = np.zeros(args.policies), 0, 0
        for k in range(args.policies):
            cand = np.clip(theta + deltas[k], 0.0, 1.0)
            res = evaluate_theta(env, phys=P.to_physical(cand, specs), distance_m=args.distance)
            os.chdir(cwd)
            scores[k] = res["fs"]
            falls += int(res["fell"])
            succ += int(res["success"])
            phys = P.to_physical(cand, specs)
            append_csv(run_dir / "evaluations.csv", EVAL_FIELDS, {
                "iteration": it, "k": k, "eval_index": state["evals"] + k + 1,
                **{f"d_{n}": float(d) for n, d in zip(P.NAMES, deltas[k])},
                **{n: float(v) for n, v in zip(P.NAMES, phys)},
                **{c: res[c] for c in ("outcome", "fell", "success", "walk_time_s",
                                       "distance_m", "lateral_left_m", "fs",
                                       "wall_time_s")},
            })
            print(f"  it {it:3d} k {k:2d}: {res['outcome']:<12} d={res['distance_m']:+.3f} m "
                  f"t={res['walk_time_s']:5.2f} s  F_S={res['fs']:+.4f}")

        theta, A = pgrl_update(theta, deltas, scores, args.eta)
        state["iteration"] = it
        state["evals"] += args.policies
        append_csv(run_dir / "iterations.csv", ITER_FIELDS, {
            "iteration": it, "evals_cum": state["evals"],
            "fs_mean": scores.mean(), "fs_max": scores.max(), "fs_min": scores.min(),
            "n_falls": falls, "n_success": succ, "wall_time_s": time.time() - t_it,
            **P.as_dict(theta, specs),
        })
        state["theta"] = theta.tolist()
        state["rng_state"] = rng.bit_generator.state
        state_path.write_text(json.dumps(state, indent=2))
        print(f"[pgrl] it {it}: F_S mean {scores.mean():+.4f} max {scores.max():+.4f} "
              f"falls {falls}/{args.policies}  evals {state['evals']}  "
              f"|A|max {np.abs(A).max():.4f}  ({time.time() - t_it:.0f}s)")
        print("       θ = " + ", ".join(f"{n}={v:.3f}" for n, v in P.as_dict(theta, specs).items()))

    env.close()   # hard-exits the process; everything is already on disk


if __name__ == "__main__":
    main()
