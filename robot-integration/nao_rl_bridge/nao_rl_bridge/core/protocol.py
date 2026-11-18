"""
protocol.py — the report's evaluation protocol (§5.1.1) on the real robot.

* Perturbation: repetition k with amplitude δ adds radians(δ)·u_k to the 12
  leg joints of the warm-up pose, u_k ~ U[-1, 1]^12 from seed base_seed + k,
  i.e. exactly `perturbation_direction()` of
  experiments/test_evaluation_script/evaluate_algorithms.py, so repetition k
  starts from the same pose in simulation and on the robot.
* Episode: warm-up (`warmup_steps`), then walking. Time and distance count
  from the end of the warm-up. The run ends at the first of:
    - finish line: the operator marks it (or, optionally, the odometry
      estimate reaches D)                                  -> "success"
    - fall detected by the safety supervisor               -> "fall"
    - `max_episode_steps` reached (50 s from the start)    -> "timeout"
    - operator stop / emergency stop / sensor or timing fault -> "aborted"
  Aborted runs are not protocol repetitions and should be repeated.

`EpisodeRunner` is pure logic (no ROS, no NAOqi): the node feeds it one
sensor sample per control tick and sends whatever command it returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .command import leg_command
from .controller import StepOutput
from .estimators import RobotState, SensorSample, StateEstimator
from .safety import SafetySupervisor

OUTCOMES = ("success", "fall", "timeout", "aborted")


def perturbation_direction(base_seed: int, repetition: int, n_joints: int = 12) -> np.ndarray:
    return np.random.default_rng(base_seed + repetition).uniform(-1.0, 1.0, size=n_joints)


def perturbation_offset(delta_deg: float, base_seed: int, repetition: int) -> np.ndarray:
    return np.radians(delta_deg) * perturbation_direction(base_seed, repetition)


@dataclass
class TickResult:
    state: RobotState
    command: Optional[np.ndarray] = None       # (12,) leg targets to send, None when done
    step: Optional[StepOutput] = None
    rate_limited: int = 0
    hip_yaw_gap: float = 0.0
    done: bool = False
    outcome: Optional[str] = None
    reason: str = ""


@dataclass
class EpisodeSummary:
    outcome: Optional[str] = None
    reason: str = ""
    steps: int = 0
    walk_started: bool = False
    walk_time_s: float = 0.0
    odometry_forward_m: float = 0.0
    odometry_lateral_m: float = 0.0
    fell_during_warmup: bool = False
    rate_limited_steps: int = 0
    max_hip_yaw_gap_deg: float = 0.0
    extra: dict = field(default_factory=dict)


class EpisodeRunner:
    def __init__(self, controller, estimator: StateEstimator, safety: SafetySupervisor,
                 target_distance_m: float, auto_finish_on_odometry: bool,
                 hip_yaw_pitch_mode: str = "mean"):
        self.ctl = controller
        self.est = estimator
        self.safety = safety
        self.D = float(target_distance_m)
        self.auto_finish = bool(auto_finish_on_odometry)
        self.hyp_mode = hip_yaw_pitch_mode
        self.max_steps = controller.spec.max_episode_steps
        self.summary = EpisodeSummary()
        self._prev_cmd: Optional[np.ndarray] = None
        self._walk_t0: Optional[float] = None
        self._finish_requested = False
        self._stop_reason: Optional[str] = None
        self.done = False

    # ── operator events ──
    def request_finish(self):
        """Operator: the robot crossed the finish line."""
        self._finish_requested = True

    def request_stop(self, reason: str):
        """Operator / node: abort the run."""
        self._stop_reason = reason

    # ── control tick ──
    def tick(self, sample: SensorSample, now: float) -> TickResult:
        if self.done:
            raise RuntimeError("episode already finished")
        warmup = self.ctl.warmup_steps
        # The state sampled now is the state after step (warmup - 1): the end
        # of the warm-up, from which time and distance are measured.
        if self._walk_t0 is None and self.ctl.step_count >= max(warmup - 1, 0):
            self._walk_t0 = now
            self.est.reset_odometry()
            self.summary.walk_started = True

        state = self.est.update(sample)
        if self._walk_t0 is not None:
            self.summary.walk_time_s = now - self._walk_t0
        self.summary.odometry_forward_m, self.summary.odometry_lateral_m = state.odometry

        if self._stop_reason is not None:
            return self._end(state, "aborted", self._stop_reason)
        verdict = self.safety.check(state, now - sample.stamp)
        if verdict is not None:
            kind, reason = verdict
            if kind == "fall":
                self.summary.fell_during_warmup = self._walk_t0 is None
            return self._end(state, kind if kind == "fall" else "aborted", reason)
        if self._finish_requested:
            return self._end(state, "success", "operator marked the finish line")
        if self.auto_finish and self._walk_t0 is not None and state.odometry[0] >= self.D:
            return self._end(state, "success", f"odometry reached {self.D} m")
        if self.ctl.step_count >= self.max_steps:
            return self._end(state, "timeout", f"{self.max_steps} steps")

        out = self.ctl.step(state)
        targets, gap = leg_command(self.ctl.spec.leg_order, out.leg_targets, self.hyp_mode)
        prev = self._prev_cmd if self._prev_cmd is not None else state.leg_q
        cmd, n_lim = self.safety.limit_rate(targets, prev)
        self._prev_cmd = cmd
        self.summary.steps = out.step
        self.summary.rate_limited_steps += int(n_lim > 0)
        self.summary.max_hip_yaw_gap_deg = max(self.summary.max_hip_yaw_gap_deg, float(np.degrees(gap)))
        return TickResult(state=state, command=cmd, step=out, rate_limited=n_lim, hip_yaw_gap=gap)

    def _end(self, state, outcome, reason) -> TickResult:
        self.done = True
        self.summary.outcome, self.summary.reason = outcome, reason
        return TickResult(state=state, done=True, outcome=outcome, reason=reason)
