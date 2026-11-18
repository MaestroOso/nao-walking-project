"""
observation.py — build the policy input exactly as the training env did.

The vector layout comes from deployment.yaml (`observation.components`).
Each component is computed from a `RobotState` in the training conventions:

projected_gravity   rotate_inverse(q_world, (0, 0, -1))
angular_velocity    rotate_inverse(q_world, R(q_world)·ω_body) · scale
                    (Isaac returned a WORLD-frame ω and the env rotated it
                    back with the same function; for `wxyz` this is just
                    ω_body·scale, for `legacy_xyzw` it is the distorted value
                    the PPO baseline was trained on)
linear_velocity     world_xy: Rz(yaw_world)·(v_fwd, v_lat)   (sim world frame)
                    heading:  (v_fwd, v_lat)
leg_joint_positions 2(q - mid)/(max - min), limits from deployment.yaml
base_height         estimated torso height [m]
gait_phase          (sin, cos)(2π·phase); `warmup_value` during warm-up

`rotate_inverse` is the correct or the legacy function according to
`observation.quaternion_convention`.
"""
from __future__ import annotations

import math

import numpy as np

from .deployment_spec import DeploymentSpec
from .estimators import RobotState
from .rotations import GRAVITY_DIR, ROTATE_INVERSE, rotation_matrix


class ObservationBuilder:
    def __init__(self, spec: DeploymentSpec):
        if spec.observation is None:
            raise ValueError(f"{spec.name}: no observation section (not an sb3 policy)")
        self.spec = spec
        self.obs_spec = spec.observation
        self.rot_inv = ROTATE_INVERSE[self.obs_spec.quaternion_convention]
        lo, hi = spec.limits_rad[:, 0], spec.limits_rad[:, 1]
        self._mid, self._range = (hi + lo) / 2.0, hi - lo
        self.dim = self.obs_spec.dim

    def components(self, st: RobotState, phase_signal: np.ndarray) -> dict:
        """{component name: values} (useful for logging and debugging)."""
        out = {}
        q = st.quat_world
        for c in self.obs_spec.components:
            if c.name == "projected_gravity":
                v = self.rot_inv(q, GRAVITY_DIR)
            elif c.name == "angular_velocity":
                w_world = rotation_matrix(q) @ st.gyro_body
                v = self.rot_inv(q, w_world) * float(c.options["scale"])
            elif c.name == "linear_velocity":
                if c.options["frame"] == "heading":
                    v = st.v_heading
                else:
                    cy, sy = math.cos(st.yaw_world), math.sin(st.yaw_world)
                    f, l = st.v_heading
                    v = np.array([cy * f - sy * l, sy * f + cy * l])
            elif c.name == "leg_joint_positions":
                v = 2.0 * (st.leg_q - self._mid) / self._range
            elif c.name == "base_height":
                v = np.array([st.base_height])
            elif c.name == "gait_phase":
                v = phase_signal
            else:  # unreachable: names are validated by deployment_spec
                raise ValueError(c.name)
            out[c.name] = np.asarray(v, dtype=np.float64).reshape(c.dim)
        return out

    def build(self, st: RobotState, phase_signal: np.ndarray) -> np.ndarray:
        parts = self.components(st, phase_signal)
        obs = np.concatenate([parts[c.name] for c in self.obs_spec.components]).astype(np.float32)
        if self.obs_spec.nonfinite_to_zero:
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs
