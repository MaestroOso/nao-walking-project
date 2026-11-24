"""
controller.py — per-step joint targets, reproducing the training envs.

`ResidualController` (PPO, SAC) mirrors `NaoWalkingEnv.step()`:

    obs   = observation(state, phase signal of the PREVIOUS step)
    a     = clip(policy(obs), -1, 1)                 # also during warm-up
    a     = 0  during warm-up if control.warmup_actions is false (SAC)
    n    += 1
    q_ref = warm-up pose            if n < warmup_steps
            gait(t), t = (n - warmup_steps)·dt       otherwise
    q*    = clip(q_ref + scale·a, limits)

`PGRLController` mirrors `PGRLWalkingEnv.compute_leg_targets()`:

    n    += 1
    q*    = warm-up pose                                       if n < warmup_steps
            clip(q_h + a·(gait(f·t) - q_h) + Δq(g_x, g_y), limits)  otherwise

with q_h the warm-up pose (in the common evaluation script the perturbed home
pose is also the centre of the amplitude scaling, so this is reproduced) and
Δq the torso-tilt compensator on ankle/hip pitch and roll of both legs.

The warm-up pose is the nominal home pose plus the protocol's perturbation
offset, clipped to the joint limits (as `make_evaluation_env_class`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .deployment_spec import DeploymentSpec
from .estimators import RobotState
from .gait import ReferenceGait
from .observation import ObservationBuilder
from .policies import MlpPolicy, PGRLParams
from .rotations import GRAVITY_DIR, rotate_inverse_wxyz

# PGRL stiffness factor order (pgrl_params: s_hip_yawpitch ... s_ankle_roll)
PGRL_STIFFNESS_JOINT_TYPES = ("HipYawPitch", "HipRoll", "HipPitch",
                              "KneePitch", "AnklePitch", "AnkleRoll")


@dataclass
class StepOutput:
    step: int                     # step counter AFTER this step (1-based, as the env)
    in_warmup: bool               # this step used the warm-up pose
    phase: float
    leg_targets: np.ndarray       # (12,) [rad], clipped to the joint limits
    obs: Optional[np.ndarray] = None
    action: Optional[np.ndarray] = None
    compensation: Optional[np.ndarray] = None


class _Controller:
    def __init__(self, spec: DeploymentSpec, leg_offset_rad=None):
        self.spec = spec
        self.gait = ReferenceGait(spec)
        self.lo, self.hi = spec.limits_rad[:, 0], spec.limits_rad[:, 1]
        off = np.zeros(12) if leg_offset_rad is None else np.asarray(leg_offset_rad, float)
        if off.shape != (12,):
            raise ValueError("leg offset must have 12 values")
        self.warmup_pose = np.clip(spec.home_pose_rad + off, self.lo, self.hi)
        self.applied_offset = self.warmup_pose - spec.home_pose_rad
        self.warmup_steps = spec.warmup_steps
        self.dt = spec.control_dt
        self.reset()

    def reset(self):
        self.step_count = 0
        self.phase = 0.0

    def _advance(self):
        """Increment the step counter; return gait time t or None during warm-up."""
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            self.phase = 0.0
            return None
        return (self.step_count - self.warmup_steps) * self.dt

    def leg_stiffness(self, base: float, max_value: float) -> np.ndarray:
        """NAOqi stiffness per leg joint (leg order)."""
        return np.full(12, min(base, max_value))


class ResidualController(_Controller):
    def __init__(self, spec: DeploymentSpec, policy: MlpPolicy, leg_offset_rad=None):
        if spec.action is None or spec.observation is None:
            raise ValueError(f"{spec.name}: residual controller needs action + observation")
        super().__init__(spec, leg_offset_rad)
        self.policy = policy
        self.obs_builder = ObservationBuilder(spec)
        if policy.obs_dim != self.obs_builder.dim or policy.act_dim != 12:
            raise ValueError(f"policy dims ({policy.obs_dim}->{policy.act_dim}) do not match "
                             f"deployment ({self.obs_builder.dim}->12)")
        self.scale = spec.action.scale_rad
        self._warmup_signal = np.asarray(
            spec.observation.component("gait_phase").options["warmup_value"], float) \
            if spec.observation.component("gait_phase") else None

    def phase_signal(self) -> np.ndarray:
        if self.step_count < self.warmup_steps and self._warmup_signal is not None:
            return self._warmup_signal
        return self.gait.phase_signal(self.phase)

    def step(self, state: RobotState) -> StepOutput:
        obs = self.obs_builder.build(state, self.phase_signal())
        action = np.clip(self.policy.act(obs), -1.0, 1.0)
        t = self._advance()
        if t is None:
            ref = self.warmup_pose
            if not self.spec.warmup_actions:     # training env ignored it here
                action = np.zeros_like(action)
        else:
            self.phase = self.gait.phase(t)
            ref = self.gait.targets(t)
        targets = np.clip(ref + action * self.scale, self.lo, self.hi)
        return StepOutput(self.step_count, t is None, self.phase, targets, obs, action)


class PGRLController(_Controller):
    def __init__(self, spec: DeploymentSpec, params: PGRLParams, leg_offset_rad=None):
        if spec.pgrl is None:
            raise ValueError(f"{spec.name}: not a PGRL deployment")
        super().__init__(spec, leg_offset_rad)
        self.params = params
        idx = {n: i for i, n in enumerate(spec.leg_order)}
        self._ank_pitch = [idx["LAnklePitch"], idx["RAnklePitch"]]
        self._hip_pitch = [idx["LHipPitch"], idx["RHipPitch"]]
        self._ank_roll = [idx["LAnkleRoll"], idx["RAnkleRoll"]]
        self._hip_roll = [idx["LHipRoll"], idx["RHipRoll"]]
        self._stiff_idx = [PGRL_STIFFNESS_JOINT_TYPES.index(n[1:]) for n in spec.leg_order]

    def step(self, state: RobotState) -> StepOutput:
        t = self._advance()
        comp = np.zeros(12)
        if t is None:
            q = self.warmup_pose.copy()
        else:
            p = self.params
            q_ref = self.gait.targets(p.f * t)
            q = self.warmup_pose + p.a * (q_ref - self.warmup_pose)
            self.phase = self.gait.phase(p.f * t)
            g = rotate_inverse_wxyz(state.quat_world, GRAVITY_DIR)    # yaw-invariant tilt
            k_ap, k_hp, k_ar, k_hr = p.gains
            comp[self._ank_pitch] = k_ap * g[0]
            comp[self._hip_pitch] = k_hp * g[0]
            comp[self._ank_roll] = k_ar * g[1]
            comp[self._hip_roll] = k_hr * g[1]
        targets = np.clip(q + comp, self.lo, self.hi)
        return StepOutput(self.step_count, t is None, self.phase, targets, compensation=comp)

    def leg_stiffness(self, base: float, max_value: float) -> np.ndarray:
        """Sim stiffness factor s_j scales kp/kd; on the robot it scales NAOqi
        stiffness, clamped to max_value (NAOqi accepts at most 1.0)."""
        s = self.params.stiffness[self._stiff_idx]
        return np.clip(base * s, 0.0, max_value)
