"""Unit tests for the robot-side core (estimators, safety, runner, policies, config)."""
import math

import numpy as np
import pytest

from conftest import PPO_YAML, URDF, ScriptedPolicy
from nao_rl_bridge.core.command import full_command, leg_command
from nao_rl_bridge.core.controller import ResidualController
from nao_rl_bridge.core.deployment_spec import load_deployment
from nao_rl_bridge.core.estimators import SensorSample, StateEstimator
from nao_rl_bridge.core.kinematics import LegKinematics
from nao_rl_bridge.core.policies import MlpPolicy
from nao_rl_bridge.core.protocol import EpisodeRunner
from nao_rl_bridge.core.robot_config import RobotConfigError, load_robot_config
from nao_rl_bridge.core.rotations import (quat_from_rpy, rotate_inverse_wxyz,
                                          rotation_matrix, rpy_from_quat)
from nao_rl_bridge.core.safety import SafetyConfig, SafetySupervisor

SPEC = load_deployment(PPO_YAML)
HOME = dict(zip(SPEC.leg_order, SPEC.home_pose_rad))


# ── rotations ────────────────────────────────────────────────────────────────
def test_rpy_roundtrip_and_matrix():
    rng = np.random.default_rng(0)
    for _ in range(200):
        r, p, y = rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2), rng.uniform(-3, 3)
        q = quat_from_rpy(r, p, y)
        np.testing.assert_allclose(rpy_from_quat(q), (r, p, y), atol=1e-9)
        v = rng.normal(size=3)
        np.testing.assert_allclose(rotate_inverse_wxyz(q, v), rotation_matrix(q).T @ v, atol=1e-12)


def test_forward_lean_gives_positive_gx():
    """Sign convention the sensor check verifies on the robot."""
    g = rotate_inverse_wxyz(quat_from_rpy(0.0, math.radians(10), 0.0), [0, 0, -1])
    assert g[0] > 0.17 and abs(g[1]) < 1e-12
    g = rotate_inverse_wxyz(quat_from_rpy(math.radians(10), 0.0, 0.0), [0, 0, -1])
    assert g[1] < -0.17      # leaning to the robot's right (positive roll) gives g_y < 0


# ── kinematics / estimator ───────────────────────────────────────────────────
def test_fk_matches_sim_heights():
    kin = LegKinematics(URDF)
    straight = kin.foot_positions({n: 0.0 for n in SPEC.leg_order})
    assert straight["l_sole"][2] == pytest.approx(-0.33301, abs=1e-5)   # sim spawn height 0.3344
    home = kin.foot_positions(HOME)
    assert home["l_sole"][2] == pytest.approx(home["r_sole"][2])
    assert 0.29 < -home["l_sole"][2] < 0.32


class _FixedFootKinematics:
    """Torso moves; the left sole stays at a fixed world point."""

    def __init__(self):
        self.torso = np.zeros(3)
        self.foot_world = np.array([0.0, 0.05, -0.30])

    def foot_positions(self, q):
        p = self.foot_world - self.torso
        return {"l_sole": p, "r_sole": p + np.array([0.0, -0.1, 0.03])}


def _sample(t, joints=None, roll=0.0, pitch=0.0, yaw=0.0, gyro=(0, 0, 0)):
    return SensorSample(joints=joints or dict(HOME), roll=roll, pitch=pitch, yaw=yaw,
                        gyro=np.array(gyro, float), stamp=t)


def test_velocity_height_and_odometry_from_stance_foot():
    kin = _FixedFootKinematics()
    est = StateEstimator(kin, SPEC.leg_order, math.pi, velocity_filter_alpha=1.0, height_offset_m=0.002)
    dt, v = 1 / 30, np.array([0.2, -0.05])
    for k in range(31):
        kin.torso[:2] = v * k * dt
        st = est.update(_sample(k * dt))
    np.testing.assert_allclose(st.v_heading, v, atol=1e-9)
    assert st.base_height == pytest.approx(0.302)
    assert st.stance_foot == "l_sole"
    np.testing.assert_allclose(st.odometry, v * 30 * dt, atol=1e-9)
    assert math.cos(st.yaw_world) == pytest.approx(-1.0)   # spawn yaw π, no IMU yaw change


def test_yaw_is_relative_to_episode_start():
    est = StateEstimator(_FixedFootKinematics(), SPEC.leg_order, math.pi)
    est.update(_sample(0.0, yaw=1.0))
    st = est.update(_sample(0.1, yaw=1.2))
    assert st.yaw_world == pytest.approx(-math.pi + 0.2)   # wrapped π + 0.2


# ── safety ───────────────────────────────────────────────────────────────────
def _state(**kw):
    est = StateEstimator(LegKinematics(URDF), SPEC.leg_order, math.pi)
    return est.update(_sample(0.0, **kw))


def test_safety_checks():
    s = SafetySupervisor(SafetyConfig(fall_tilt_deg=40, min_base_height_m=0.2, sensor_timeout_s=0.1))
    assert s.check(_state(), 0.01) is None
    assert s.check(_state(pitch=math.radians(45)), 0.01)[0] == "fall"
    assert s.check(_state(), 0.5)[0] == "abort"
    prev = np.zeros(12)
    lim, n = s.limit_rate(np.full(12, 1.0), prev)
    assert n == 12 and np.allclose(lim, math.radians(12))
    lim, n = s.limit_rate(np.full(12, 0.1), prev)
    assert n == 0 and np.allclose(lim, 0.1)


# ── command ──────────────────────────────────────────────────────────────────
def test_hip_yaw_pitch_rule():
    t = np.zeros(12)
    t[0], t[6] = 0.2, 0.1
    out, gap = leg_command(SPEC.leg_order, t, "mean")
    assert out[0] == out[6] == pytest.approx(0.15) and gap == pytest.approx(0.1)
    out, _ = leg_command(SPEC.leg_order, t, "left")
    assert out[0] == out[6] == pytest.approx(0.2)
    names, angles = full_command(SPEC.leg_order, out, SPEC.fixed_pose_rad)
    assert len(names) == len(angles) == 12 + 14 and names[0] == "LHipYawPitch"


# ── policies ─────────────────────────────────────────────────────────────────
def test_mlp_policy_matches_manual_forward_pass():
    rng = np.random.default_rng(0)
    W = [rng.normal(size=(8, 23)), rng.normal(size=(8, 8)), rng.normal(size=(12, 8))]
    b = [rng.normal(size=8), rng.normal(size=8), rng.normal(size=12)]
    x = rng.normal(size=23)
    h = np.tanh(W[1] @ np.tanh(W[0] @ x + b[0]) + b[1])
    raw = W[2] @ h + b[2]
    ppo = MlpPolicy(W, b, "tanh", "clip", -np.ones(12), np.ones(12))
    np.testing.assert_allclose(ppo.act(x), np.clip(raw, -1, 1), atol=1e-5)
    hr = np.maximum(W[1] @ np.maximum(W[0] @ x + b[0], 0) + b[1], 0)
    sac = MlpPolicy(W, b, "relu", "tanh", -np.ones(12), np.ones(12))
    np.testing.assert_allclose(sac.act(x), np.tanh(W[2] @ hr + b[2]), atol=1e-5)
    with pytest.raises(ValueError):
        MlpPolicy(W, b, "gelu", "clip", -np.ones(12), np.ones(12))


# ── episode runner ───────────────────────────────────────────────────────────
def _runner(max_steps_override=None, auto_finish=False, fall_tilt=40.0):
    ctl = ResidualController(SPEC, ScriptedPolicy(np.zeros((2000, 12))))
    if max_steps_override:
        ctl.spec = type(SPEC)(**{**SPEC.__dict__, "max_episode_steps": max_steps_override})
    kin = _FixedFootKinematics()
    est = StateEstimator(kin, SPEC.leg_order, math.pi, velocity_filter_alpha=1.0)
    r = EpisodeRunner(ctl, est, SafetySupervisor(SafetyConfig(fall_tilt_deg=fall_tilt,
                                                              min_base_height_m=0.1)),
                      target_distance_m=1.0, auto_finish_on_odometry=auto_finish)
    return r, kin


def _drive(r, kin, n, v=0.0, t0=0.0, **sample_kw):
    res = None
    for k in range(n):
        t = t0 + k / 30
        kin.torso[0] = v * t
        res = r.tick(_sample(t, **sample_kw), now=t)
        if res.done:
            break
    return res


def test_runner_warmup_then_walk_and_operator_finish():
    r, kin = _runner()
    res = _drive(r, kin, 59)
    assert not r.summary.walk_started and r.ctl.step_count == 59
    res = _drive(r, kin, 31, t0=59 / 30)         # tick at step_count 59 starts the walk
    assert r.summary.walk_started and res.command is not None
    assert r.summary.walk_time_s == pytest.approx(30 / 30)
    r.request_finish()
    res = _drive(r, kin, 1, t0=90 / 30)
    assert res.done and res.outcome == "success"
    with pytest.raises(RuntimeError):
        r.tick(_sample(4.0), 4.0)


def test_runner_auto_finish_on_odometry():
    r, kin = _runner(auto_finish=True)
    res = _drive(r, kin, 2000, v=0.5)
    assert res.outcome == "success"
    assert r.summary.odometry_forward_m == pytest.approx(1.0, abs=0.02)
    assert r.summary.walk_time_s == pytest.approx(2.0, abs=0.05)


def test_runner_timeout_fall_and_abort():
    r, kin = _runner(max_steps_override=100)
    assert _drive(r, kin, 200).outcome == "timeout" and r.summary.steps == 100
    r, kin = _runner()
    res = _drive(r, kin, 5, pitch=math.radians(50))
    assert res.outcome == "fall" and r.summary.fell_during_warmup
    r, kin = _runner()
    _drive(r, kin, 3)
    r.request_stop("operator stop")
    assert _drive(r, kin, 1, t0=0.1).outcome == "aborted"


# ── robot config ─────────────────────────────────────────────────────────────
def test_robot_config(tmp_path):
    cfg = load_robot_config()
    assert cfg["topics"]["joint_angles"] == "/joint_angles"
    p = tmp_path / "r.yaml"
    p.write_text("safety: {fall_tilt_deg: 30}\n")
    assert load_robot_config(p)["safety"]["fall_tilt_deg"] == 30
    p.write_text("safety: {fall_tilt: 30}\n")
    with pytest.raises(RobotConfigError, match="unknown key 'safety.fall_tilt'"):
        load_robot_config(p)
    p.write_text("finish: {on_fall: dance}\n")
    with pytest.raises(RobotConfigError):
        load_robot_config(p)


def test_packaged_robot_yaml_is_valid():
    from conftest import PKG_DIR
    load_robot_config(PKG_DIR / "config/robot.yaml")
