"""
kinematics.py — leg forward kinematics from the NAO URDF.

Used to estimate what the real robot cannot measure but the policies observe
(torso height, torso velocity): with the joint encoders we know where each
sole is relative to the torso, and the foot on the ground is a fixed point.

Only the standard library XML parser and NumPy are used. The chain from
`base_link` to `l_sole` / `r_sole` is read from the URDF (joint origins,
axes and fixed joints), so it stays correct if the URDF changes.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .rotations import rpy_matrix


def _origin_tf(joint: ET.Element) -> np.ndarray:
    o = joint.find("origin")
    xyz = [0.0, 0.0, 0.0] if o is None else [float(v) for v in o.get("xyz", "0 0 0").split()]
    rpy = [0.0, 0.0, 0.0] if o is None else [float(v) for v in o.get("rpy", "0 0 0").split()]
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(*rpy)
    T[:3, 3] = xyz
    return T


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
    T = np.eye(4)
    T[:3, :3] = R
    return T


class LegKinematics:
    """Sole positions in the base_link (torso) frame."""

    def __init__(self, urdf_path, base_link: str = "base_link",
                 feet: Tuple[str, str] = ("l_sole", "r_sole")):
        root = ET.parse(Path(urdf_path)).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        self.feet = feet
        self._chains: Dict[str, List[Tuple[str, str, np.ndarray, np.ndarray]]] = {}
        for foot in feet:
            chain = []
            link = foot
            while link != base_link:
                if link not in by_child:
                    raise ValueError(f"{urdf_path}: no path from {base_link} to {foot}")
                j = by_child[link]
                axis = j.find("axis")
                ax = (np.array([float(v) for v in axis.get("xyz").split()])
                      if axis is not None else np.zeros(3))
                chain.append((j.get("name"), j.get("type"), _origin_tf(j), ax))
                link = j.find("parent").get("link")
            self._chains[foot] = list(reversed(chain))
        self.joint_names = sorted({n for c in self._chains.values()
                                   for n, t, _, _ in c if t in ("revolute", "continuous")})

    def foot_positions(self, q: Dict[str, float]) -> Dict[str, np.ndarray]:
        """{foot: (3,) position in base_link frame} for joint angles q [rad]."""
        out = {}
        for foot, chain in self._chains.items():
            T = np.eye(4)
            for name, jtype, T0, axis in chain:
                T = T @ T0
                if jtype in ("revolute", "continuous"):
                    T = T @ _axis_angle(axis, float(q.get(name, 0.0)))
            out[foot] = T[:3, 3].copy()
        return out
