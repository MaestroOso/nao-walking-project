"""
estimators.py — turn raw robot sensors into the state the policies observe.

The simulator gave the policies quantities the real NAO cannot measure
directly. This module rebuilds them from the encoders and the torso IMU:

* Orientation in the TRAINING world frame. The IMU gives roll and pitch
  relative to gravity (drift-free) and a yaw angle that is an integrated gyro
  (drifts slowly). The world yaw used by the observation is
      yaw_world = spawn_yaw + (yaw_imu - yaw_imu_at_episode_start)
  so the robot "faces world -x" at the start, as at spawn in Isaac.
* Base height: the lowest sole (stance foot) is on the ground, so
      h = max over feet of  -(R_rp · p_foot)_z  + height_offset_m
  with p_foot from leg forward kinematics in the torso frame and R_rp the
  roll/pitch rotation. `height_offset_m` absorbs the difference between the
  URDF sole frame and the contact surface used in simulation.
* Base velocity in the heading frame (yaw-only frame): the stance foot is
  static, so the torso moves with minus the foot's velocity,
      v = -(dp/dt + ω_z × p),   p = R_rp · p_foot (stance foot),
  followed by an exponential low-pass filter.
* Odometry: v integrated (rotated by the yaw change) gives forward / lateral
  displacement since the end of the warm-up, for logging and the optional
  automatic finish line.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .kinematics import LegKinematics
from .rotations import quat_from_rpy, rpy_matrix, wrap_angle


@dataclass
class SensorSample:
    """One synchronous snapshot of the robot's sensors."""
    joints: Dict[str, float]      # joint name -> angle [rad] (from /joint_states)
    roll: float                   # IMU AngleX [rad]
    pitch: float                  # IMU AngleY [rad]
    yaw: float                    # IMU AngleZ [rad] (integrated, drifts)
    gyro: np.ndarray              # body-frame angular velocity [rad/s]
    stamp: float                  # monotonic time of the sample [s]


@dataclass
class RobotState:
    """What the controller consumes (everything in training conventions)."""
    leg_q: np.ndarray             # (12,) in deployment leg order [rad]
    roll: float
    pitch: float
    yaw_world: float              # yaw in the training world frame [rad]
    quat_world: np.ndarray        # (w, x, y, z) torso orientation, training world
    gyro_body: np.ndarray         # (3,) [rad/s]
    v_heading: np.ndarray         # (2,) forward, lateral [m/s] (estimated)
    base_height: float            # [m] (estimated)
    stance_foot: str
    odometry: np.ndarray          # (2,) forward, lateral displacement since reset_odometry [m]


class StateEstimator:
    def __init__(self, kinematics: LegKinematics, leg_order, spawn_yaw_rad: float,
                 velocity_filter_alpha: float = 0.5, height_offset_m: float = 0.0):
        if not 0.0 < velocity_filter_alpha <= 1.0:
            raise ValueError("velocity_filter_alpha must be in (0, 1]")
        self.kin = kinematics
        self.leg_order = tuple(leg_order)
        self.spawn_yaw = float(spawn_yaw_rad)
        self.alpha = float(velocity_filter_alpha)
        self.height_offset = float(height_offset_m)
        self.reset()

    def reset(self):
        self._yaw0: Optional[float] = None
        self._prev_feet: Optional[Dict[str, np.ndarray]] = None
        self._prev_stamp: Optional[float] = None
        self._v = np.zeros(2)
        self.reset_odometry()

    def reset_odometry(self):
        self._odom = np.zeros(2)
        self._odom_yaw0: Optional[float] = None

    def update(self, s: SensorSample) -> RobotState:
        if self._yaw0 is None:
            self._yaw0 = s.yaw
        dyaw = wrap_angle(s.yaw - self._yaw0)
        yaw_world = wrap_angle(self.spawn_yaw + dyaw)
        quat = quat_from_rpy(s.roll, s.pitch, yaw_world)

        # Feet in the level (roll/pitch-compensated, yaw-free) frame.
        R_rp = rpy_matrix(s.roll, s.pitch, 0.0)
        feet_body = self.kin.foot_positions(s.joints)
        feet = {f: R_rp @ p for f, p in feet_body.items()}
        stance = min(feet, key=lambda f: feet[f][2])      # lowest sole
        height = -feet[stance][2] + self.height_offset

        gyro = np.asarray(s.gyro, dtype=np.float64)
        if self._prev_feet is not None and s.stamp > self._prev_stamp:
            dt = s.stamp - self._prev_stamp
            p, p_prev = feet[stance], self._prev_feet[stance]
            wz = float((R_rp @ gyro)[2])
            dp = (p - p_prev) / dt
            v_raw = -(dp[:2] + wz * np.array([-p[1], p[0]]))
            self._v = self.alpha * v_raw + (1 - self.alpha) * self._v
            if self._odom_yaw0 is None:
                self._odom_yaw0 = dyaw
            a = dyaw - self._odom_yaw0                    # heading change since odometry start
            c, sn = math.cos(a), math.sin(a)
            self._odom += dt * np.array([c * self._v[0] - sn * self._v[1],
                                         sn * self._v[0] + c * self._v[1]])
        self._prev_feet, self._prev_stamp = feet, s.stamp

        leg_q = np.array([s.joints[n] for n in self.leg_order], dtype=np.float64)
        return RobotState(leg_q=leg_q, roll=s.roll, pitch=s.pitch, yaw_world=yaw_world,
                          quat_world=quat, gyro_body=gyro, v_heading=self._v.copy(),
                          base_height=float(height), stance_foot=stance,
                          odometry=self._odom.copy())
