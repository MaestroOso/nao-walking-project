"""
session.py — one protocol repetition on the robot, independent of ROS.

    prepare   wait for sensors -> NAOqi setup -> wake up (StandInit) ->
              stiffness -> interpolate to the warm-up pose -> settle -> pose check
    run       30 Hz loop: sensors -> EpisodeRunner.tick -> joint command
    finish    home / freeze / rest, depending on the outcome (robot.yaml `finish`)

The ROS node supplies three callables: `get_sample()` (latest SensorSample
or None), `send_command(names, angles, speed)` and optionally
`publish_debug(tick_result)`. Everything is logged with RunLogger.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable, Optional

import numpy as np

from .core.command import full_command, leg_command
from .core.estimators import StateEstimator
from .core.exported import ExportedPolicy
from .core.kinematics import LegKinematics
from .core.protocol import EpisodeRunner, EpisodeSummary, perturbation_offset
from .core.run_log import RunLogger
from .core.safety import SafetyConfig, SafetySupervisor

SENSOR_WAIT_S = 2.0


class WalkSession:
    def __init__(self, exported: ExportedPolicy, cfg: dict, backend,
                 get_sample: Callable, send_command: Callable,
                 publish_debug: Optional[Callable] = None,
                 label: str = "", delta_deg: float = 0.0, repetition: int = 0,
                 logger=None, clock=time.monotonic, sleep=time.sleep):
        self.ex, self.cfg, self.backend = exported, cfg, backend
        self.get_sample, self.send_command = get_sample, send_command
        self.publish_debug = publish_debug
        self.label = label or exported.name
        self.delta_deg, self.repetition = float(delta_deg), int(repetition)
        self.log = logger or logging.getLogger("nao_rl_bridge.session")
        self.clock, self.sleep = clock, sleep
        self.spec = exported.spec
        self._stop_reason: Optional[str] = None
        self._finish = threading.Event()
        self.runner: Optional[EpisodeRunner] = None
        self.run_dir = None

        seed = int(cfg["protocol"]["base_seed"]) + self.repetition
        self.offset = perturbation_offset(self.delta_deg, int(cfg["protocol"]["base_seed"]), self.repetition)
        self.seed = seed
        self.ctl = exported.make_controller(self.offset)
        spawn_yaw = math.radians(self.spec.observation.spawn_yaw_deg) if self.spec.observation else math.pi
        e = cfg["estimator"]
        self.est = StateEstimator(LegKinematics(exported.urdf_path), self.spec.leg_order, spawn_yaw,
                                  e["velocity_filter_alpha"], e["height_offset_m"])
        s = cfg["safety"]
        self.safety = SafetySupervisor(SafetyConfig(
            max_joint_step_deg=s["max_joint_step_deg"], fall_tilt_deg=s["fall_tilt_deg"],
            min_base_height_m=s["min_base_height_m"], sensor_timeout_s=s["sensor_timeout_s"]))

    # ── operator events (thread-safe) ──
    def request_finish(self):
        self._finish.set()
        if self.runner is not None:
            self.runner.request_finish()

    def request_stop(self, reason: str):
        self._stop_reason = reason
        if self.runner is not None:
            self.runner.request_stop(reason)

    # ── helpers ──
    def _command_for(self, leg_targets):
        legs, _ = leg_command(self.spec.leg_order, leg_targets, self.cfg["command"]["hip_yaw_pitch_mode"])
        return full_command(self.spec.leg_order, legs, self.spec.fixed_pose_rad)

    def _wait_for_sensors(self):
        t_end = self.clock() + SENSOR_WAIT_S
        while self.clock() < t_end:
            s = self.get_sample()
            if s is not None and self.clock() - s.stamp < self.cfg["safety"]["sensor_timeout_s"]:
                return s
            self.sleep(0.05)
        return None

    def _stiffness(self):
        st = self.cfg["stiffness"]
        legs = self.ctl.leg_stiffness(float(st["legs"]), float(st["max"]))
        out = dict(zip(self.spec.leg_order, (float(v) for v in legs)))
        out.update({n: min(float(st["arms_head"]), float(st["max"])) for n in self.spec.fixed_pose_rad})
        return out

    # ── the run ──
    def run(self) -> EpisodeSummary:
        cfg = self.cfg
        rl = RunLogger(cfg["logging"]["run_root"], self.label, self.delta_deg, self.repetition)
        self.run_dir = rl.dir
        rl.write_meta(
            label=self.label, policy=self.ex.name, algorithm=self.spec.algorithm,
            export_dir=self.ex.path, export_created=self.ex.manifest.get("created"),
            export_source=self.ex.manifest.get("source"), dry_run=bool(self.backend.dry_run),
            delta_deg=self.delta_deg, repetition=self.repetition, seed=self.seed,
            applied_offset_deg=np.degrees(self.ctl.applied_offset),
            target_distance_m=float(cfg["protocol"]["distance_m"]), robot_config=cfg,
            stiffness=self._stiffness())
        summary = EpisodeSummary()
        try:
            summary = self._run(rl)
        except Exception as e:                       # never leave the robot driven blindly
            self.log.error(f"run failed: {e!r}")
            summary = self.runner.summary if self.runner else summary
            summary.outcome, summary.reason = "aborted", f"exception: {e!r}"
            self._finish_action("aborted")
        finally:
            rl.write_meta(**{k: v for k, v in summary.__dict__.items() if k != "extra"},
                          **summary.extra)
            rl.close()
        self.log.info(f"run finished: {summary.outcome} ({summary.reason}) -> {rl.dir}")
        return summary

    def _abort_before_start(self, reason) -> EpisodeSummary:
        self.log.error(f"run not started: {reason}")
        s = EpisodeSummary(outcome="aborted", reason=reason)
        self._finish_action("aborted")
        return s

    def _run(self, rl: RunLogger) -> EpisodeSummary:
        cfg, p = self.cfg, self.cfg["prepare"]
        sample = self._wait_for_sensors()
        if sample is None:
            return EpisodeSummary(outcome="aborted", reason="no fresh sensor data")
        needed = set(self.spec.leg_order)
        missing = sorted(needed - set(sample.joints))
        if missing:
            return EpisodeSummary(outcome="aborted", reason=f"joint states lack {missing}")

        # ── prepare ──
        self.backend.prepare(p["disable_autonomous_life"], p["fall_manager_enabled"],
                             p["smart_stiffness_enabled"])
        self.backend.wake_up()
        self.backend.set_stiffness(self._stiffness(), float(cfg["stiffness"]["ramp_s"]))
        names, angles = self._command_for(self.ctl.warmup_pose)
        self.backend.move_to(names, angles, float(p["move_duration_s"]))
        self.sleep(float(p["settle_s"]))
        if self._stop_reason:
            return self._abort_before_start(self._stop_reason)

        sample = self.get_sample()
        expected = dict(zip(names, angles))
        err = max(abs(sample.joints[n] - expected[n]) for n in self.spec.leg_order)
        rl.write_meta(pose_error_deg=math.degrees(err))
        if math.degrees(err) > float(p["max_pose_error_deg"]):
            if self.backend.dry_run:
                self.log.warning(f"[dry-run] pose error {math.degrees(err):.1f} deg ignored")
            else:
                return self._abort_before_start(f"pose error {math.degrees(err):.1f} deg after prepare")

        # ── run ──
        self.ctl.reset()
        self.est.reset()
        self.runner = EpisodeRunner(self.ctl, self.est, self.safety,
                                    float(cfg["protocol"]["distance_m"]),
                                    bool(cfg["protocol"]["auto_finish_on_odometry"]),
                                    cfg["command"]["hip_yaw_pitch_mode"])
        if self._finish.is_set():
            self.runner.request_finish()
        if self._stop_reason:
            self.runner.request_stop(self._stop_reason)
        dt = self.spec.control_dt
        speed = float(cfg["command"]["speed_fraction"])
        max_over = int(cfg["safety"]["max_consecutive_overruns"])
        over, n_over_total, max_late = 0, 0, 0.0
        t_start = next_t = self.clock()
        while True:
            t_tick = self.clock()
            sample = self.get_sample()
            if sample is None:
                self.runner.request_stop("sensor data lost")
                sample = self._last_sample
            self._last_sample = sample
            res = self.runner.tick(sample, t_tick)
            if res.done:
                break
            names, angles = full_command(self.spec.leg_order, res.command, self.spec.fixed_pose_rad)
            self.send_command(names, angles, speed)
            compute_ms = (self.clock() - t_tick) * 1000
            if self.publish_debug:
                self.publish_debug(res)
            st, out = res.state, res.step
            rl.log_tick({
                "t": t_tick - t_start, "step": out.step, "in_warmup": int(out.in_warmup),
                "phase": out.phase, "roll": st.roll, "pitch": st.pitch, "yaw_world": st.yaw_world,
                "gyro": st.gyro_body, "leg_q": st.leg_q, "v_heading": st.v_heading,
                "base_height": st.base_height, "stance_left": int(st.stance_foot == "l_sole"),
                "odometry": st.odometry,
                "obs": out.obs if out.obs is not None else [],
                "action": out.action if out.action is not None else [],
                "compensation": out.compensation if out.compensation is not None else [],
                "targets": out.leg_targets, "command": res.command,
                "rate_limited": res.rate_limited, "hip_yaw_gap": res.hip_yaw_gap,
                "compute_ms": compute_ms,
            })
            next_t += dt
            delay = next_t - self.clock()
            if delay < 0:
                over += 1
                n_over_total += 1
                max_late = max(max_late, -delay)
                if over >= max_over:
                    self.runner.request_stop(f"control loop late {over} ticks in a row")
                next_t = self.clock()
            else:
                over = 0
                self.sleep(delay)

        summary = self.runner.summary
        summary.extra.update(loop_overruns=n_over_total, max_late_ms=max_late * 1000)
        self._finish_action(summary.outcome)
        return summary

    _last_sample = None

    def _finish_action(self, outcome: Optional[str]):
        f = self.cfg["finish"]
        action = f.get(f"on_{outcome}", "freeze") if outcome else "freeze"
        all_names = list(self.spec.leg_order) + list(self.spec.fixed_pose_rad)
        try:
            if action == "home":
                names, angles = self._command_for(self.spec.home_pose_rad)
                self.backend.move_to(names, angles, float(f["home_duration_s"]))
            elif action == "rest":
                self.backend.rest()
            else:
                self.backend.freeze(all_names)
        except Exception as e:
            self.log.error(f"finish action '{action}' failed: {e!r}")
