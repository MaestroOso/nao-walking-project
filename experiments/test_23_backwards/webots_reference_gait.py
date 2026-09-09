"""
WebotsReferenceGait — literal replay of the Webots Forwards.motion trajectory.

This is a deliberate strip-down. All prior versions applied transformations
(windowing, rolling to double-support, home-anchoring, symmetrizing, ramps)
that progressively moved us away from the raw Webots data while still
failing to produce a stable gait.

This version does ONE thing: interpolate the raw Webots trajectory at any
time t, and return the absolute joint angles Webots commanded at that
moment. No rolling, no windowing, no offsets, no symmetrization, no ramp.

Usage contract:
  - The env's `_leg_home` pose MUST match the first row of the file (t=0),
    so there is no discontinuity when warmup ends.
  - Phase is (t mod period) / period for observation-signal purposes;
    targets come from get_targets(t) directly, not from phase.
"""
import math
import numpy as np
from pathlib import Path


class WebotsReferenceGait:
    EXPECTED_JOINT_ORDER = [
        "LHipYawPitch", "LHipRoll", "LHipPitch",
        "LKneePitch",   "LAnklePitch", "LAnkleRoll",
        "RHipYawPitch", "RHipRoll", "RHipPitch",
        "RKneePitch",   "RAnklePitch", "RAnkleRoll",
    ]

    def __init__(self,
                 keyframes_path="nao_gait_data/forwards_keyframes.npy",
                 timestamps_path="nao_gait_data/forwards_timestamps.npy",
                 joint_names_path="nao_gait_data/forwards_joint_names.txt",
                 leg_home=None,        # ignored (first-frame is authoritative)
                 ramp_duration_s=0.0,  # ignored
                 amplitude_scale=1.0,  # ignored
                 gait_hz=None,
                 lateral_only=False,
                 **_ignored_kwargs):

        q = np.load(keyframes_path)
        ts = np.load(timestamps_path)
        names = Path(joint_names_path).read_text().strip().split("\n")

        col_idx = [names.index(n) for n in self.EXPECTED_JOINT_ORDER]
        q = q[:, col_idx].astype(np.float32)

        self._t = ts.astype(np.float32)
        self._q = q
        self._period = float(ts[-1])
        self._first_frame = q[0].copy()
        self.lateral_only = lateral_only

        print(f"[WebotsReferenceGait] LITERAL REPLAY -- "
              f"{len(self._t)} keyframes, duration={self._period:.3f}s")
        fd = np.degrees(self._first_frame)
        print(f"[WebotsReferenceGait] t=0 pose (deg): "
              f"L[hipR={fd[1]:+.1f} hipP={fd[2]:+.1f} knee={fd[3]:+.1f} "
              f"ankP={fd[4]:+.1f} ankR={fd[5]:+.1f}]  "
              f"R[hipR={fd[7]:+.1f} hipP={fd[8]:+.1f} knee={fd[9]:+.1f} "
              f"ankP={fd[10]:+.1f} ankR={fd[11]:+.1f}]")
        print(f"[WebotsReferenceGait] (no windowing, no rolling, "
              f"no symmetrization, no ramp)")

    def get_phase(self, t):
        return (t % self._period) / self._period

    def get_phase_signal(self, phase):
        a = 2.0 * math.pi * phase
        return math.sin(a), math.cos(a)

    def get_targets(self, phase, t_since_warmup=None):
        """Return 12 leg joint targets at time `t_since_warmup` (seconds).
        Clamp to final frame after file end."""
        if t_since_warmup is None:
            t_query = float(phase) * self._period
        else:
            t_query = float(t_since_warmup)
        if t_query >= self._t[-1]:
            return self._q[-1].copy()
        return np.array(
            [np.interp(t_query, self._t, self._q[:, c]) for c in range(12)],
            dtype=np.float32,
        )
