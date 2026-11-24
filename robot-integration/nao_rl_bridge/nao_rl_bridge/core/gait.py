"""
gait.py — reference gait playback, reproducing the two training variants.

* `hold_last` — `WebotsReferenceGait.get_targets` (PPO/SAC): linear
  interpolation of the literal Webots clip; after the last timestamp the
  last frame is held.
* `wrap` — `LoopedGait.targets` (PGRL, walking-solutions/pgrl/baseline_test28): the clip is periodic with
  duration = last timestamp + one frame interval; inside that final interval
  the last frame is blended back into the first.

The phase signal is (t mod period) / period, with period = phase_period_s.
"""
from __future__ import annotations

import math

import numpy as np

from .deployment_spec import DeploymentSpec


class ReferenceGait:
    def __init__(self, spec: DeploymentSpec):
        self.q, self.t = spec.load_gait_arrays()        # (N, 12) in leg_order, (N,)
        self.after_end = spec.gait.after_end
        self.period = spec.gait.phase_period_s
        dt = float(np.median(np.diff(self.t)))
        self.duration = float(self.t[-1]) + dt          # loop length for "wrap"

    @property
    def first_frame(self) -> np.ndarray:
        return self.q[0].copy()

    def targets(self, t: float) -> np.ndarray:
        """12 leg targets [rad] at gait time t >= 0 (seconds since warm-up end)."""
        t = float(t)
        if self.after_end == "hold_last":
            if t >= self.t[-1]:
                return self.q[-1].copy()
        else:
            t = t % self.duration
            if t > self.t[-1]:
                frac = (t - self.t[-1]) / (self.duration - self.t[-1])
                return (1 - frac) * self.q[-1] + frac * self.q[0]
        return np.array([np.interp(t, self.t, self.q[:, c]) for c in range(self.q.shape[1])])

    def phase(self, t: float) -> float:
        return (float(t) % self.period) / self.period

    @staticmethod
    def phase_signal(phase: float) -> np.ndarray:
        a = 2.0 * math.pi * phase
        return np.array([math.sin(a), math.cos(a)])
