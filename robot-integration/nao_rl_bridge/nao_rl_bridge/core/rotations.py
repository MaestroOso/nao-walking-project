"""
rotations.py — orientation helpers shared by the observation builder and the
estimators.

Conventions
-----------
* Quaternions are SCALAR-FIRST (w, x, y, z), the convention Isaac Sim uses.
* Euler angles are roll (x), pitch (y), yaw (z) with R = Rz(yaw)·Ry(pitch)·Rx(roll),
  the same convention as NAOqi's InertialSensor AngleX/AngleY/AngleZ and ROS
  `tf2` setRPY. Body frame: x forward, y left, z up (NAO torso = URDF base_link).

Two "rotate a world vector into the body frame" functions exist on purpose:

* `rotate_inverse_wxyz` — the correct one (R^T v).
* `rotate_inverse_legacy` — a verbatim copy of `_quat_rotate_inverse` from
  `walking-solutions/ppo/baseline_test24/nao_walking_env.py`, which unpacks an
  Isaac (w, x, y, z) quaternion as (x, y, z, w). Policies trained with that
  code learned from its output, so the robot must feed them the same
  distortion (deployment.yaml: `quaternion_convention: legacy_xyzw`).
"""
from __future__ import annotations

import math

import numpy as np

GRAVITY_DIR = np.array([0.0, 0.0, -1.0])


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """(w, x, y, z) of R = Rz(yaw)·Ry(pitch)·Rx(roll)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def rpy_from_quat(q) -> tuple:
    """(roll, pitch, yaw) of a (w, x, y, z) quaternion."""
    w, x, y, z = (float(e) for e in q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def rotation_matrix(q) -> np.ndarray:
    """3x3 matrix of a (w, x, y, z) quaternion (body -> world)."""
    w, x, y, z = (float(e) for e in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rpy_matrix(roll: float, pitch: float, yaw: float = 0.0) -> np.ndarray:
    return rotation_matrix(quat_from_rpy(roll, pitch, yaw))


def rotate_inverse_wxyz(q, v) -> np.ndarray:
    """World vector -> body frame (correct (w, x, y, z) handling)."""
    w, x, y, z = (float(e) for e in q)
    qv = np.array([-x, -y, -z])
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def rotate_inverse_legacy(q, v) -> np.ndarray:
    """Verbatim copy of baseline_test24 `_quat_rotate_inverse` (reads q as x, y, z, w).

    `q` is the true (w, x, y, z) orientation, exactly what Isaac passed in.
    """
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    qv = np.array([-x, -y, -z])
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


ROTATE_INVERSE = {"wxyz": rotate_inverse_wxyz, "legacy_xyzw": rotate_inverse_legacy}


def tilt_angle(roll: float, pitch: float) -> float:
    """Angle between the torso z axis and the vertical [rad]."""
    return math.acos(max(-1.0, min(1.0, math.cos(roll) * math.cos(pitch))))


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
