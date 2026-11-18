"""WalkSession end to end, without ROS or a robot.

A fake robot ("plant") applies every command instantly and reports an
upright IMU; a fake clock makes a whole repetition run in milliseconds.
"""
import csv
import json
import math

import numpy as np
import pytest

from conftest import PPO_YAML, URDF
from nao_rl_bridge.core.deployment_spec import load_deployment
from nao_rl_bridge.core.estimators import SensorSample
from nao_rl_bridge.core.exported import ExportedPolicy
from nao_rl_bridge.core.policies import MlpPolicy
from nao_rl_bridge.core.robot_config import load_robot_config
from nao_rl_bridge.robot.naoqi_backend import DryRunBackend
from nao_rl_bridge.session import WalkSession

SPEC = load_deployment(PPO_YAML)


def _exported():
    rng = np.random.default_rng(0)
    W = [rng.normal(0, 0.1, (16, 23)), rng.normal(0, 0.1, (12, 16))]
    b = [np.zeros(16), np.zeros(12)]
    mlp = MlpPolicy(W, b, "tanh", "clip", -np.ones(12), np.ones(12))
    return ExportedPolicy(path=PPO_YAML.parent, manifest={"created": "test"}, spec=SPEC,
                          urdf_path=URDF, mlp=mlp)


class Plant:
    """Fake robot + clock."""

    def __init__(self, tick_dt=1 / 30, pitch_after=None, finish_after=None, slow_after=None):
        self.t = 0.0
        self.joints = {n: 0.0 for n in SPEC.leg_order}
        self.joints.update({n: 0.0 for n in SPEC.fixed_pose_rad})
        self.pitch = 0.0
        self.commands = 0
        self.pitch_after, self.finish_after, self.slow_after = pitch_after, finish_after, slow_after
        self.session = None

    def clock(self):
        return self.t

    def sleep(self, d):
        self.t += d

    def sample(self):
        return SensorSample(joints=dict(self.joints), roll=0.0, pitch=self.pitch, yaw=0.0,
                            gyro=np.zeros(3), stamp=self.t)

    def send(self, names, angles, speed):
        self.joints.update(dict(zip(names, angles)))
        self.commands += 1
        if self.pitch_after and self.commands >= self.pitch_after:
            self.pitch = math.radians(60)
        if self.finish_after and self.commands == self.finish_after:
            self.session.request_finish()
        if self.slow_after and self.commands >= self.slow_after:
            self.t += 0.1                     # computation takes longer than a tick


class PlantBackend(DryRunBackend):
    def __init__(self, plant, dry_run=True):
        super().__init__()
        self.plant, self.dry_run = plant, dry_run

    def move_to(self, names, angles, duration_s):
        super().move_to(names, angles, duration_s)
        self.plant.joints.update(dict(zip(names, angles)))
        self.plant.t += duration_s


def _session(tmp_path, plant, dry_run=True, **cfg_over):
    cfg = load_robot_config()
    cfg["logging"]["run_root"] = str(tmp_path)
    for sec, kv in cfg_over.items():
        cfg[sec].update(kv)
    backend = PlantBackend(plant, dry_run)
    s = WalkSession(_exported(), cfg, backend, plant.sample, plant.send, label="t",
                    delta_deg=2.0, repetition=1, clock=plant.clock, sleep=plant.sleep)
    plant.session = s
    return s, backend


def _calls(backend):
    return [c[0] for c in backend.calls]


def test_full_run_finish_line(tmp_path):
    plant = Plant(finish_after=120)
    s, backend = _session(tmp_path, plant)
    summ = s.run()
    assert summ.outcome == "success" and summ.walk_started
    assert _calls(backend)[:4] == ["prepare", "wake_up", "set_stiffness", "move_to"]
    assert _calls(backend)[-1] == "move_to"                 # finish: back to home
    meta = json.loads((s.run_dir / "run.json").read_text())
    assert meta["outcome"] == "success" and meta["seed"] == 1 and meta["delta_deg"] == 2.0
    assert max(abs(x) for x in meta["applied_offset_deg"]) <= 2.0
    rows = list(csv.DictReader(open(s.run_dir / "ticks.csv")))
    assert len(rows) == 120 and "obs_22" in rows[0] and "command_11" in rows[0]
    assert meta["walk_time_s"] == pytest.approx((120 - 59) / 30, abs=1e-6)


def test_fall_stops_and_rests(tmp_path):
    plant = Plant(pitch_after=80)
    s, backend = _session(tmp_path, plant)
    summ = s.run()
    assert summ.outcome == "fall" and "tilt" in summ.reason
    assert _calls(backend)[-1] == "rest"


def test_timeout(tmp_path):
    plant = Plant()
    s, _ = _session(tmp_path, plant)
    assert s.run().outcome == "timeout"
    assert plant.commands == SPEC.max_episode_steps


def test_timing_overruns_abort(tmp_path):
    plant = Plant(slow_after=70)
    s, backend = _session(tmp_path, plant)
    summ = s.run()
    assert summ.outcome == "aborted" and ("late" in summ.reason or "stale" in summ.reason)
    assert _calls(backend)[-1] == "freeze"


def test_no_sensors_aborts_without_moving(tmp_path):
    plant = Plant()
    s, backend = _session(tmp_path, plant)
    s.get_sample = lambda: None
    assert s.run().outcome == "aborted"
    assert "move_to" not in _calls(backend) and "wake_up" not in _calls(backend)


def test_pose_check_live_vs_dry_run(tmp_path):
    class Stuck(PlantBackend):
        def move_to(self, names, angles, duration_s):   # robot does not reach the pose
            self.plant.t += duration_s
    for dry, expected in ((False, "aborted"), (True, "timeout")):
        plant = Plant()
        cfg = load_robot_config()
        cfg["logging"]["run_root"] = str(tmp_path / str(dry))
        backend = Stuck(plant, dry_run=dry)
        s = WalkSession(_exported(), cfg, backend, plant.sample, plant.send,
                        clock=plant.clock, sleep=plant.sleep)
        assert s.run().outcome == expected


def test_operator_stop_before_start(tmp_path):
    plant = Plant()
    s, backend = _session(tmp_path, plant)
    s.request_stop("operator stop")
    summ = s.run()
    assert summ.outcome == "aborted" and plant.commands == 0
