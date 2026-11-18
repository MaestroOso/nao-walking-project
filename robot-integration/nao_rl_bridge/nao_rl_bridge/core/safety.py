"""
safety.py — checks that stand between the policy and the motors.

* Rate limit: each joint command may move at most `max_joint_step_deg`
  from the previous command per control step (at 30 Hz, 12° ≈ 6.3 rad/s,
  about the NAO's motor speed). Clipped steps are counted and logged,
  because they make the robot behave differently from simulation.
* Fall detection: torso tilt above `fall_tilt_deg` or estimated torso
  height below `min_base_height_m`. Much earlier than the simulator's
  criteria (height < 0.10 m, rotation > ~145°) so the operator can catch
  the robot; a run stopped this way counts as a fall in the protocol.
* Abort: sensor data older than `sensor_timeout_s`, or non-finite values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .estimators import RobotState
from .rotations import tilt_angle


@dataclass
class SafetyConfig:
    max_joint_step_deg: float = 12.0
    fall_tilt_deg: float = 40.0
    min_base_height_m: float = 0.20
    sensor_timeout_s: float = 0.1


class SafetySupervisor:
    def __init__(self, cfg: SafetyConfig):
        if cfg.max_joint_step_deg <= 0 or cfg.fall_tilt_deg <= 0 or cfg.sensor_timeout_s <= 0:
            raise ValueError("safety limits must be positive")
        self.cfg = cfg
        self.max_step = math.radians(cfg.max_joint_step_deg)

    def check(self, state: RobotState, sensor_age_s: float) -> Optional[Tuple[str, str]]:
        """None if OK, else (kind, reason) with kind 'fall' or 'abort'."""
        if sensor_age_s > self.cfg.sensor_timeout_s:
            return "abort", f"stale sensors ({sensor_age_s * 1000:.0f} ms)"
        vals = np.concatenate([state.leg_q, state.gyro_body, state.v_heading,
                               [state.roll, state.pitch, state.base_height]])
        if not np.all(np.isfinite(vals)):
            return "abort", "non-finite sensor values"
        tilt = math.degrees(tilt_angle(state.roll, state.pitch))
        if tilt > self.cfg.fall_tilt_deg:
            return "fall", f"tilt {tilt:.1f} deg > {self.cfg.fall_tilt_deg}"
        if state.base_height < self.cfg.min_base_height_m:
            return "fall", f"torso height {state.base_height:.3f} m < {self.cfg.min_base_height_m}"
        return None

    def limit_rate(self, targets: np.ndarray, previous: np.ndarray) -> Tuple[np.ndarray, int]:
        delta = np.clip(targets - previous, -self.max_step, self.max_step)
        limited = previous + delta
        return limited, int(np.count_nonzero(np.abs(limited - targets) > 1e-9))
