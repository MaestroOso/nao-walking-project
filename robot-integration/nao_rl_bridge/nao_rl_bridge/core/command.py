"""
command.py — from 12 leg targets to a full NAOqi joint command.

The NAO has ONE motor for both HipYawPitch joints (RHipYawPitch is slaved to
LHipYawPitch), while the simulator modelled them independently, so a policy
may ask for two different values. `hip_yaw_pitch_mode` decides what is sent:
  * "mean": both get the mean of the two targets (default);
  * "left": both get the left target (what NAOqi does internally).
The disagreement is returned so it can be logged.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

HIP_YAW_PITCH_MODES = ("mean", "left")


def leg_command(leg_order, leg_targets: np.ndarray, mode: str) -> Tuple[np.ndarray, float]:
    """Apply the HipYawPitch rule. Returns (targets, |L - R| disagreement [rad])."""
    if mode not in HIP_YAW_PITCH_MODES:
        raise ValueError(f"hip_yaw_pitch_mode must be one of {HIP_YAW_PITCH_MODES}")
    out = np.array(leg_targets, dtype=np.float64)
    il, ir = leg_order.index("LHipYawPitch"), leg_order.index("RHipYawPitch")
    gap = abs(out[il] - out[ir])
    v = 0.5 * (out[il] + out[ir]) if mode == "mean" else out[il]
    out[il] = out[ir] = v
    return out, float(gap)


def full_command(leg_order, leg_targets: np.ndarray,
                 fixed_pose: Dict[str, float]) -> Tuple[List[str], List[float]]:
    """Joint names and angles for every controlled joint (legs + fixed pose)."""
    names = list(leg_order) + list(fixed_pose)
    angles = [float(a) for a in leg_targets] + [float(fixed_pose[n]) for n in fixed_pose]
    return names, angles
