#!/usr/bin/env python3
"""
evaluate_algorithms.py — Common evaluation protocol for every walking algorithm
================================================================================

Implements the evaluation protocol of the report (§4.6.2 and §5.1) once, so
PPO, SAC and PGRL are all measured in exactly the same way:

  * Straight-line course of D metres (``--distance``) along the robot's forward
    direction (world -x, the convention used by every env in this project).
  * Deterministic policy (mean action, no exploration noise).
  * N repetitions (``--repetitions``, default 10) for each perturbation
    amplitude delta (``--deltas``, default 0 1 2 4 degrees). Before repetition
    k, each of the 12 leg joints gets an offset  delta * u_k[j],  with
    u_k ~ U[-1, 1]^12 drawn from seed (base_seed + k). The SAME u_k is used for
    every delta and every algorithm, so repetition k always starts from the
    same perturbation *direction*; only its amplitude changes.
  * The offset is applied to the pose the PD controller holds during the
    warm-up (teleport, settle, and the warm-up targets themselves).
  * Distance and time are measured from the end of the warm-up (the instant the
    reference gait starts). The run stops at the first of:
        - forward distance >= D                  -> "success"
        - env termination (height / tilt / NaN)  -> "fall_height" / "fall_tilt"
                                                    / "invalid"
        - env truncation (1500 steps = 50 s from
          the START of the episode)              -> "timeout" (not a fall)

The reward is NOT used: each algorithm's env may compute a different reward
(e.g. test_26_sac's body-frame revision), so all metrics are computed here from
the robot's physical state only (see metrics.py).

Run ONE algorithm per process: Isaac Sim's SimulationApp can only be created
once per process, and every algorithm folder ships its own
``nao_walking_env.py``. ``run_all_evaluations.sh`` loops over algorithms.

Usage (inside the Isaac Sim Python environment):
    python evaluate_algorithms.py --algo ppo --distance 1.0
    python evaluate_algorithms.py --algo sac --distance 1.5 --deltas 0 1 2 4
    python evaluate_algorithms.py --algo sac --model path/to/ckpt.zip --label SAC_ckpt4M
    python evaluate_algorithms.py --algo pgrl                       # test_28 pgrl_v1
    python evaluate_algorithms.py --algo pgrl --pgrl-initial --label PGRL_base
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from metrics import fs_index, heading_drift_deg, fall_rate_percent, mean_std  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Default model / environment per algorithm ("last version" of each).
# Override with --model / --env-dir.
# ─────────────────────────────────────────────────────────────────────────────
ALGO_DEFAULTS = {
    "ppo": {
        # PPO test_24 run, 27.2M steps (progress.txt), last trained 17 Aug.
        # Lives in the working folder in $HOME, outside this repo. A second
        # copy of the run (22.35M steps, 8 Sep) is in the gitignored
        # <repo>/nao_walking_project/ folder — see README "PPO model choice".
        # Note: model.num_timesteps reads ~53k for both, because train_ppo.py
        # resets it on every chunk resume; progress.txt has the true count.
        "model": Path.home() / "nao_walking_project/experiments/test_24/runs/test24/final_model.zip",
        "env_dir": REPO_ROOT / "walking-solutions/ppo/baseline_test24",
    },
    "sac": {
        # Latest SAC run (body-frame reward revision, still training).
        "model": REPO_ROOT / "experiments/test_26_sac/runs/sac_pilot_v2/final_model.zip",
        "env_dir": REPO_ROOT / "experiments/test_26_sac",
    },
    "pgrl": {
        # PGRL run pgrl_v1 (166 iterations, 1992 evaluations, gains in [-1, 1]).
        # Run folders are gitignored; re-create with test_28/run_chunked_training.sh.
        "model": REPO_ROOT / "experiments/test_28/runs/pgrl_v1",
        "env_dir": REPO_ROOT / "experiments/test_28",
    },
}

CSV_COLUMNS = [
    "label", "algo", "delta_deg", "repetition", "seed", "target_distance_m",
    "outcome", "success", "fell", "fell_during_warmup",
    "steps", "walk_time_s", "distance_m", "lateral_left_m",
    "heading_drift_deg", "mean_speed_mps", "fs",
    "max_abs_offset_deg", "wall_time_s",
]


# ═════════════════════════════════════════════════════════════════════════════
# Policy adapters
# ═════════════════════════════════════════════════════════════════════════════
class PolicyAdapter:
    """Contract between the evaluation loop and one algorithm.

    An adapter must provide:
      make_env(render_mode) -> env   an *evaluation env* (see contract below)
      act(obs) -> action             deterministic action for observation obs
      describe() -> dict             provenance info stored in the run's JSON

    Evaluation-env contract (what run_episode() relies on):
      env.set_leg_offset(offset_rad)   12 leg-joint offsets, applied on reset()
      env.reset() -> (obs, info)
      env.step(action) -> (obs, reward, terminated, truncated, info)
      env._nao.get_world_pose() -> (pos[3], quat[4])   (quat[3] = scalar part,
                                                        project convention)
      env._step_count, env.control_dt, env.gait_warmup_steps,
      env.term_height, env.term_orient_qw, env.n_legs, env.close()
    NaoWalkingEnv wrapped by make_evaluation_env_class() satisfies all of this.
    """

    def make_env(self, render_mode):
        raise NotImplementedError

    def act(self, obs):
        raise NotImplementedError

    def describe(self):
        return {}


class SB3Adapter(PolicyAdapter):
    """PPO / SAC policies trained with Stable-Baselines3 on NaoWalkingEnv."""

    def __init__(self, algo, model_path, env_dir):
        self.algo = algo
        self.model_path = Path(model_path).resolve()
        self.env_dir = Path(env_dir).resolve()
        if not self.model_path.is_file():
            sys.exit(f"[eval] Model not found: {self.model_path}")
        if not (self.env_dir / "nao_walking_env.py").is_file():
            sys.exit(f"[eval] No nao_walking_env.py in env dir: {self.env_dir}")

        # SB3/torch are imported before Isaac Sim's SimulationApp is created,
        # exactly as in the existing evaluate_policy.py scripts.
        from stable_baselines3 import PPO, SAC
        algo_cls = {"ppo": PPO, "sac": SAC}[algo]
        self.model = algo_cls.load(str(self.model_path), device="cpu")
        print(f"[eval] Loaded {algo.upper()} model: {self.model_path} "
              f"(num_timesteps={getattr(self.model, 'num_timesteps', '?')})")

    def make_env(self, render_mode):
        # The env files resolve their URDF / reference-gait paths RELATIVE TO
        # THE CURRENT WORKING DIRECTORY (e.g. "../../shared/robot/nao.urdf"),
        # so we must run from the algorithm's own folder.
        os.chdir(self.env_dir)
        sys.path.insert(0, str(self.env_dir))
        from nao_walking_env import NaoWalkingEnv
        env = make_evaluation_env_class(NaoWalkingEnv)(render_mode=render_mode)

        if tuple(self.model.observation_space.shape) != tuple(env.observation_space.shape):
            sys.exit(f"[eval] Observation space mismatch: model "
                     f"{self.model.observation_space.shape} vs env {env.observation_space.shape}")
        if tuple(self.model.action_space.shape) != tuple(env.action_space.shape):
            sys.exit(f"[eval] Action space mismatch: model "
                     f"{self.model.action_space.shape} vs env {env.action_space.shape}")
        return env

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def describe(self):
        env_file = self.env_dir / "nao_walking_env.py"
        return {
            "model_path": str(self.model_path),
            "model_sha256": sha256_of(self.model_path),
            "model_num_timesteps": int(getattr(self.model, "num_timesteps", -1)),
            "env_dir": str(self.env_dir),
            "env_file_sha256": sha256_of(env_file),
        }


class PGRLAdapter(PolicyAdapter):
    """PGRL controller from experiments/test_28 (report §4.5.3).

    The policy is the 12-parameter vector θ found by PGRL (4 torso-tilt
    compensator gains, gait frequency / amplitude factors, 6 leg stiffness
    factors) applied to the looped Webots gait of test_27. The controller
    lives INSIDE the env (`PGRLWalkingEnv.step()` computes the joint targets
    from the parameters and ignores its `action` argument), so act() returns
    nothing.

    * `model_path`: a run folder or its `state.json` (e.g.
      experiments/test_28/runs/pgrl_v1). θ is mapped to physical values with
      the parameter ranges THAT run was trained with (its config.json),
      because the ranges changed after pgrl_v1 (gains [-1,1] -> [-3,3]).
    * `initial=True`: ignore the model and use the initial parameters
      (gains 0, f = a = 1, stiffness 1) — the "PGRL base gait without
      compensator" row of Table 5-3 (use --label PGRL_base).

    The env subclasses the PPO baseline NaoWalkingEnv, so warm-up, grace
    period, fall criteria and the 1500-step limit are the same as PPO/SAC;
    make_evaluation_env_class() adds the initial-pose perturbation exactly as
    for the other algorithms.
    """

    def __init__(self, model_path, env_dir, initial=False):
        self.env_dir = Path(env_dir).resolve()
        if not (self.env_dir / "pgrl_env.py").is_file():
            sys.exit(f"[eval] No pgrl_env.py in env dir: {self.env_dir}")
        sys.path.insert(0, str(self.env_dir))
        import pgrl_params as P
        self.P = P
        self.initial = initial

        if initial:
            self.state_path, self.state = None, {}
            self.specs = P.PARAM_SPECS
            self.phys = P.to_physical(P.initial_theta(self.specs), self.specs)
            src = "initial parameters"
        else:
            path = Path(model_path).resolve()
            self.state_path = path / "state.json" if path.is_dir() else path
            if not self.state_path.is_file():
                sys.exit(f"[eval] PGRL state not found: {self.state_path}")
            self.state = json.loads(self.state_path.read_text())
            self.specs = P.load_run_specs(self.state_path.parent)
            self.phys = P.to_physical(np.array(self.state["theta"]), self.specs)
            src = (f"{self.state_path} (iteration {self.state.get('iteration')}, "
                   f"{self.state.get('evals')} evaluations)")
        print(f"[eval] PGRL policy: {src}")
        for n, v in zip(P.NAMES, self.phys):
            print(f"[eval]   {n:16s} {v:+.3f}")

    def make_env(self, render_mode):
        # pgrl_episode.make_env() wraps PGRLWalkingEnv with this script's
        # make_evaluation_env_class() and changes into the baseline env folder
        # (asset paths are relative to the CWD, as for PPO/SAC).
        from pgrl_episode import make_env
        env = make_env(render_mode)
        env.set_params(phys=self.phys)      # applied on every reset()
        return env

    def act(self, obs):
        return None

    def describe(self):
        return {
            "policy": "initial parameters" if self.initial else str(self.state_path),
            "state_sha256": None if self.initial else sha256_of(self.state_path),
            "iteration": self.state.get("iteration"),
            "evaluations": self.state.get("evals"),
            "param_specs": self.specs,
            "params_physical": {n: float(v) for n, v in zip(self.P.NAMES, self.phys)},
            "env_dir": str(self.env_dir),
            "env_file_sha256": sha256_of(self.env_dir / "pgrl_env.py"),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation env: NaoWalkingEnv + initial-pose perturbation
# ═════════════════════════════════════════════════════════════════════════════
def make_evaluation_env_class(base_cls):
    """Subclass an algorithm's NaoWalkingEnv to support pose perturbation.

    The training env is not modified. On reset(), the nominal leg home pose is
    replaced by  clip(nominal + offset, joint limits). NaoWalkingEnv uses that
    pose for the teleport, the 80+5-step settle, and as the leg target during
    the warm-up — i.e. "the pose the PD controller holds during warm-up".
    When the warm-up ends, the reference gait takes over from its own first
    frame (the nominal pose), exactly as in training.
    """

    class EvaluationEnv(base_cls):
        def set_leg_offset(self, offset_rad):
            self._eval_leg_offset = np.asarray(offset_rad, dtype=np.float32)

        def reset(self, seed=None, options=None):
            if not self._sim_initialized:
                self._init_sim()        # builds _full_home_pose / _leg_dof_indices
            if not hasattr(self, "_eval_nominal_leg_home"):
                self._eval_nominal_leg_home = self._leg_home.copy()
                self._eval_nominal_full_home = self._full_home_pose.copy()

            offset = getattr(self, "_eval_leg_offset",
                             np.zeros(self.n_legs, dtype=np.float32))
            leg_home = np.clip(self._eval_nominal_leg_home + offset,
                               self._leg_lower, self._leg_upper).astype(np.float32)
            full_home = self._eval_nominal_full_home.copy()
            full_home[self._leg_dof_indices] = leg_home

            self._leg_home = leg_home
            self._full_home_pose = full_home
            self.applied_leg_offset_rad = leg_home - self._eval_nominal_leg_home
            return super().reset(seed=seed, options=options)

    EvaluationEnv.__name__ = f"Evaluation{base_cls.__name__}"
    return EvaluationEnv


# ═════════════════════════════════════════════════════════════════════════════
# One repetition
# ═════════════════════════════════════════════════════════════════════════════
def _get_pose(env):
    pos, quat = env._nao.get_world_pose()
    pos_ok = pos is not None and np.all(np.isfinite(pos))
    return (np.asarray(pos, dtype=float) if pos_ok else None), quat


def _classify_termination(env):
    """Mirror NaoWalkingEnv._check_termination() to name the cause.

    Uses the project's existing quaternion convention (quat[3] treated as the
    scalar part, |quat[3]| < term_orient_qw => tilt), so falls are counted
    exactly as they were during training.
    """
    pos, quat = _get_pose(env)
    if pos is None:
        return "invalid"
    if pos[2] < env.term_height:
        return "fall_height"
    if quat is not None and np.all(np.isfinite(quat)) and abs(float(quat[3])) < env.term_orient_qw:
        return "fall_tilt"
    return "fall_other"


def run_episode(env, adapter, target_distance_m, offset_rad):
    """Run one repetition and return its raw measurements (no aggregation)."""
    env.set_leg_offset(offset_rad)
    obs, _ = env.reset()

    dt = env.control_dt
    # The reference gait starts at step == gait_warmup_steps (gait time t=0),
    # so the state *after* step (gait_warmup_steps - 1) is the "end of warm-up".
    start_step = max(int(env.gait_warmup_steps) - 1, 0)

    start_xy = None
    if start_step == 0:
        pos, _ = _get_pose(env)
        start_xy = pos[:2].copy() if pos is not None else None

    forward, lateral_left, walk_time = 0.0, 0.0, 0.0
    prev_forward, prev_time = 0.0, 0.0
    outcome = None
    t0 = time.time()

    while outcome is None:
        action = adapter.act(obs)
        obs, _reward, terminated, truncated, _info = env.step(action)
        step = int(env._step_count)
        pos, _ = _get_pose(env)

        if start_xy is None and step >= start_step and pos is not None:
            start_xy = pos[:2].copy()

        if start_xy is not None and pos is not None:
            prev_forward, prev_time = forward, walk_time
            # Forward = world -x; robot's left = world -y (facing -x, z up).
            forward = -(pos[0] - start_xy[0])
            lateral_left = -(pos[1] - start_xy[1])
            walk_time = (step - start_step) * dt

        if terminated:
            outcome = _classify_termination(env)
        elif start_xy is not None and forward >= target_distance_m:
            outcome = "success"
            # Interpolate the crossing time inside the last control step so
            # F_S = D / t is not biased by the 33 ms step granularity.
            if forward > prev_forward:
                frac = (target_distance_m - prev_forward) / (forward - prev_forward)
                walk_time = prev_time + frac * (walk_time - prev_time)
        elif truncated:
            outcome = "timeout"

    success = outcome == "success"
    fell = outcome.startswith("fall") or outcome == "invalid"
    return {
        "outcome": outcome,
        "success": success,
        "fell": fell,
        "fell_during_warmup": fell and start_xy is None,
        "steps": int(env._step_count),
        "walk_time_s": walk_time,
        "distance_m": forward,
        "lateral_left_m": lateral_left,
        "heading_drift_deg": heading_drift_deg(forward, lateral_left),
        "mean_speed_mps": forward / walk_time if walk_time > 0 else 0.0,
        "fs": fs_index(forward, walk_time, success, target_distance_m),
        "max_abs_offset_deg": float(np.degrees(np.max(np.abs(env.applied_leg_offset_rad)))),
        "wall_time_s": time.time() - t0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
def sha256_of(path):
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info():
    def run(*cmd):
        try:
            return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def perturbation_direction(base_seed, repetition, n_joints):
    """u_k ~ U[-1,1]^n, identical for every delta and every algorithm."""
    return np.random.default_rng(base_seed + repetition).uniform(-1.0, 1.0, size=n_joints)


def fmt(x, spec):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", required=True, choices=sorted(ALGO_DEFAULTS))
    p.add_argument("--label", default=None,
                   help="Name of this configuration in the results (default: ALGO in caps)")
    p.add_argument("--model", default=None, help="Model file (default: latest run of --algo)")
    p.add_argument("--env-dir", default=None,
                   help="Folder with the algorithm's nao_walking_env.py (default per --algo)")
    p.add_argument("--pgrl-initial", action="store_true",
                   help="PGRL only: use the initial parameters (base gait without "
                        "compensator, Table 5-3 row PGRL_base) instead of --model")
    p.add_argument("--distance", type=float, default=1.0, help="Course length D [m]")
    p.add_argument("--deltas", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0],
                   help="Perturbation amplitudes delta [deg]")
    p.add_argument("--repetitions", type=int, default=10, help="Repetitions N per delta")
    p.add_argument("--base-seed", type=int, default=0,
                   help="Repetition k uses seed base_seed + k")
    p.add_argument("--out-dir", default=str(SCRIPT_DIR / "results"))
    p.add_argument("--gui", action="store_true", help="Render the Isaac Sim window")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing results for this label")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    label = args.label or args.algo.upper()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"episodes_{label}.csv"
    meta_path = out_dir / f"run_{label}.json"
    if csv_path.exists() and not args.overwrite:
        sys.exit(f"[eval] {csv_path} already exists — use --overwrite or another --label")
    if args.distance <= 0 or args.repetitions <= 0:
        sys.exit("[eval] --distance and --repetitions must be positive")

    defaults = ALGO_DEFAULTS[args.algo]
    if args.algo == "pgrl":
        adapter = PGRLAdapter(args.model or defaults["model"],
                              args.env_dir or defaults["env_dir"],
                              initial=args.pgrl_initial)
    else:
        adapter = SB3Adapter(args.algo,
                             args.model or defaults["model"],
                             args.env_dir or defaults["env_dir"])

    print("=" * 78)
    print(f"NAO walking evaluation — {label}")
    print(f"  D = {args.distance} m | deltas = {args.deltas} deg | "
          f"N = {args.repetitions} | base seed = {args.base_seed}")
    print("=" * 78)

    env = adapter.make_env(render_mode="human" if args.gui else "headless")

    started = datetime.datetime.now().isoformat(timespec="seconds")
    meta = {
        "label": label,
        "algo": args.algo,
        "started": started,
        "protocol": {
            "target_distance_m": args.distance,
            "deltas_deg": args.deltas,
            "repetitions": args.repetitions,
            "base_seed": args.base_seed,
            "deterministic_policy": True,
            "forward_axis": "world -x",
            "distance_and_time_measured_from": "end of warm-up (gait start)",
        },
        "env": {
            "class": type(env).__mro__[1].__module__,
            "control_dt_s": env.control_dt,
            "gait_warmup_steps": env.gait_warmup_steps,
            "grace_period_steps": getattr(env, "grace_period", None),
            "max_episode_steps": env.max_episode_steps,
            "time_limit_s_from_episode_start": env.max_episode_steps * env.control_dt,
            "term_height_m": env.term_height,
            "term_orient_qw": env.term_orient_qw,
        },
        "adapter": adapter.describe(),
        "git": git_info(),
        "python": platform.python_version(),
        "command": " ".join(sys.argv),
    }

    rows = []
    header = (f"{'delta':>5} {'rep':>3} {'outcome':<12} {'t[s]':>6} {'d[m]':>7} "
              f"{'lat[m]':>7} {'head':>6} {'F_S':>7}")
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for delta in args.deltas:
                print(f"\n── delta = {delta}° ──\n{header}")
                for k in range(args.repetitions):
                    u = perturbation_direction(args.base_seed, k, env.n_legs)
                    res = run_episode(env, adapter, args.distance, np.radians(delta) * u)
                    row = {"label": label, "algo": args.algo, "delta_deg": delta,
                           "repetition": k, "seed": args.base_seed + k,
                           "target_distance_m": args.distance, **res}
                    writer.writerow(row)
                    f.flush()            # keep partial results if Isaac Sim crashes
                    rows.append(row)
                    print(f"{delta:>5.1f} {k:>3} {res['outcome']:<12} "
                          f"{res['walk_time_s']:>6.2f} {res['distance_m']:>7.3f} "
                          f"{res['lateral_left_m']:>7.3f} {fmt(res['heading_drift_deg'], '>6.1f')} "
                          f"{res['fs']:>7.4f}")

        # ── Per-delta summary (the full table comes from summarize_results.py) ──
        print("\n" + "=" * 78)
        print(f"SUMMARY — {label} (D = {args.distance} m, N = {args.repetitions})")
        print(f"{'delta':>5} {'F_S mean':>9} {'F_S std':>8} {'TC[%]':>6} {'success':>8}")
        for delta in args.deltas:
            sub = [r for r in rows if r["delta_deg"] == delta]
            fs_m, fs_s = mean_std(r["fs"] for r in sub)
            print(f"{delta:>5.1f} {fs_m:>9.4f} {fmt(fs_s, '>8.4f')} "
                  f"{fall_rate_percent(r['fell'] for r in sub):>6.1f} "
                  f"{sum(r['success'] for r in sub):>4}/{len(sub)}")
        print(f"\nEpisodes: {csv_path}\nRun info: {meta_path}")
    finally:
        meta["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        meta["episodes_recorded"] = len(rows)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        env.close()


if __name__ == "__main__":
    main()
