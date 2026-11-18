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

LOOPING (loop=True, the default since sac_pilot_v3):
  The original behaviour played the 2.6 s clip ONCE and then held its last
  frame for the rest of the 50 s episode, while the phase signal kept
  cycling. With loop=True the targets come from the test_27 looped clip
  (experiments/test_27_gait_loop/data/looped_*.npy: back-to-back copies of
  the same clip, one copy every 2.64 s = 2.60 s + one 40 ms frame step),
  wrapped in time exactly like test_28's PGRL `LoopedGait`, so SAC and PGRL
  follow the identical base gait. The phase period is the 2.64 s copy
  period, so the phase signal stays in sync with the targets. Known seam:
  hip roll / ankle roll targets jump ~3.3 deg at each copy boundary (not
  smoothed, to stay identical to PGRL).
"""
import math
import numpy as np
from pathlib import Path

_LOOPED_DIR = (Path(__file__).resolve().parents[1]
               / "test_27_gait_loop" / "data")


class WebotsReferenceGait:
    EXPECTED_JOINT_ORDER = [
        "LHipYawPitch", "LHipRoll", "LHipPitch",
        "LKneePitch",   "LAnklePitch", "LAnkleRoll",
        "RHipYawPitch", "RHipRoll", "RHipPitch",
        "RKneePitch",   "RAnklePitch", "RAnkleRoll",
    ]

    def __init__(self,
                 keyframes_path="../../walking-solutions/shared/reference_gaits/forwards_keyframes.npy",
                 timestamps_path="../../walking-solutions/shared/reference_gaits/forwards_timestamps.npy",
                 joint_names_path="../../walking-solutions/shared/reference_gaits/forwards_joint_names.txt",
                 leg_home=None,        # ignored (first-frame is authoritative)
                 ramp_duration_s=0.0,  # ignored
                 amplitude_scale=1.0,  # ignored
                 gait_hz=None,
                 lateral_only=False,
                 loop=True,
                 looped_dir=_LOOPED_DIR,
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
        self.loop = loop

        if loop:
            lq = np.load(Path(looped_dir) / "looped_keyframes.npy")
            lt = np.load(Path(looped_dir) / "looped_timestamps.npy")
            lnames = (Path(looped_dir) / "looped_joint_names.txt").read_text().split()
            lq = lq[:, [lnames.index(n) for n in self.EXPECTED_JOINT_ORDER]]
            # Sanity: the looped file must start with exactly this clip.
            if not np.allclose(lq[:len(q)], q, atol=1e-6):
                raise ValueError(f"{looped_dir} does not start with the shared "
                                 "Forwards clip — rebuild it (test_27)")
            dt = float(np.median(np.diff(ts)))
            self._period = float(ts[-1]) + dt          # 2.64 s per copy
            self._loop_q = lq.astype(np.float64)
            self._loop_t = lt.astype(np.float64)
            # Same as test_28 LoopedGait.duration: wrapping at (last t + dt)
            # continues the periodic file seamlessly at any length.
            self._loop_duration = float(self._loop_t[-1]) + dt

        print(f"[WebotsReferenceGait] LITERAL REPLAY -- "
              f"{len(self._t)} keyframes, clip={float(ts[-1]):.3f}s, "
              + (f"LOOPED every {self._period:.3f}s (test_27 file, "
                 f"{self._loop_duration:.2f}s before wrap)" if loop
                 else "held at last frame after clip end"))
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
        loop=True: cycle the clip (identical to test_28 LoopedGait.targets).
        loop=False: clamp to the final frame after the clip ends."""
        if t_since_warmup is None:
            t_query = float(phase) * self._period
        else:
            t_query = float(t_since_warmup)
        if self.loop:
            tq = t_query % self._loop_duration
            if tq > self._loop_t[-1]:      # inside the last 40 ms: blend last -> first
                frac = (tq - self._loop_t[-1]) / (self._loop_duration - self._loop_t[-1])
                out = (1 - frac) * self._loop_q[-1] + frac * self._loop_q[0]
            else:
                out = np.array([np.interp(tq, self._loop_t, self._loop_q[:, c])
                                for c in range(12)])
            return out.astype(np.float32)
        if t_query >= self._t[-1]:
            return self._q[-1].copy()
        return np.array(
            [np.interp(t_query, self._t, self._q[:, c]) for c in range(12)],
            dtype=np.float32,
        )
