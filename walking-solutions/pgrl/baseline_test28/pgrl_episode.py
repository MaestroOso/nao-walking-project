"""
pgrl_episode.py — run ONE walking episode of a PGRL policy and score it.

The measurement follows the project's common evaluation protocol (report
§4.6.2 / §5.1): start point at the end of warm-up, forward = world −x,
success at the target distance with crossing-time interpolation, fall
classification, F_S (Eq. 4-5). The F_S that PGRL optimises during training
is therefore the same score as Table 5-3.

`make_evaluation_env_class()` and `run_episode()` are copies of the
functions of the same name in `experiments/test_evaluation_script/
evaluate_algorithms.py`, and `pgrl_metrics.py` is a copy of that folder's
`metrics.py`. They are kept here so this folder is self-contained; if the
evaluation script changes, these copies do NOT follow.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pgrl_metrics import fs_index, heading_drift_deg  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation env: PGRLWalkingEnv + initial-pose perturbation
# (copy of evaluate_algorithms.make_evaluation_env_class)
# ═════════════════════════════════════════════════════════════════════════════
def make_evaluation_env_class(base_cls):
    """Subclass an env to support an initial-pose perturbation.

    On reset(), the nominal leg home pose is replaced by
    clip(nominal + offset, joint limits). NaoWalkingEnv uses that pose for the
    teleport, the 80+5-step settle, and as the leg target during the warm-up.
    When the warm-up ends, the gait takes over from its own first frame (the
    nominal pose), exactly as in training. Offset 0 (training) = no change.
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
# One episode (copy of evaluate_algorithms.run_episode)
# ═════════════════════════════════════════════════════════════════════════════
def _get_pose(env):
    pos, quat = env._nao.get_world_pose()
    pos_ok = pos is not None and np.all(np.isfinite(pos))
    return (np.asarray(pos, dtype=float) if pos_ok else None), quat


def _classify_termination(env):
    """Mirror NaoWalkingEnv._check_termination() to name the cause."""
    pos, quat = _get_pose(env)
    if pos is None:
        return "invalid"
    if pos[2] < env.term_height:
        return "fall_height"
    if quat is not None and np.all(np.isfinite(quat)) and abs(float(quat[3])) < env.term_orient_qw:
        return "fall_tilt"
    return "fall_other"


def run_episode(env, adapter, target_distance_m, offset_rad):
    """Run one episode and return its raw measurements."""
    env.set_leg_offset(offset_rad)
    obs, _ = env.reset()

    dt = env.control_dt
    # The gait starts at step == gait_warmup_steps (gait time t=0), so the
    # state *after* step (gait_warmup_steps - 1) is the "end of warm-up".
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
            # Interpolate the crossing time inside the last control step.
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
# PGRL helpers
# ═════════════════════════════════════════════════════════════════════════════
class _NoAction:
    """The PGRL controller lives inside the env; the 'action' is unused."""

    def act(self, obs):
        return None


def make_env(render_mode="headless"):
    from pgrl_env import PGRLWalkingEnv
    # nao_walking_env / webots_reference_gait resolve their default asset
    # paths (../../shared/...) relative to the CWD: run from this folder.
    os.chdir(HERE)
    return make_evaluation_env_class(PGRLWalkingEnv)(render_mode=render_mode)


def evaluate_theta(env, theta=None, phys=None, distance_m=1.0, offset_rad=None):
    """Run one episode with the given parameters; return run_episode's dict."""
    env.set_params(theta=theta, phys=phys)
    if offset_rad is None:
        offset_rad = np.zeros(env.n_legs, dtype=np.float32)   # δ = 0 (training)
    return run_episode(env, _NoAction(), distance_m, offset_rad)
