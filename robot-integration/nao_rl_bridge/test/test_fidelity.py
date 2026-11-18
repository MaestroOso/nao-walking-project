"""Robot-side code vs the training environments, step by step.

The real env classes are driven with a fake simulator (conftest.FakeNao):
at every step the fake physics jumps to a new random torso/joint state, the
env builds its observation and joint targets, and the robot-side controller
builds its own from the same state expressed as robot sensors
(roll/pitch/yaw, body-frame gyro, heading-frame velocity, height). Both must
agree. This covers the legacy quaternion handling of the PPO baseline, the
SAC v3 observation, the PGRL compensator, the warm-up/phase sequencing, the
gait playback and the initial-pose perturbation.
"""
import numpy as np
import pytest

from conftest import (PGRL_DIR, PPO_DIR, SAC_DIR, FakeNao, FakeWorld, ScriptedPolicy,
                      attach_fake_sim, load_module, pgrl_spec_dict, random_sim_state,
                      robot_state_from_sim, sac_spec_dict, spec_dict, write_spec)
from nao_rl_bridge.core.controller import PGRLController, ResidualController
from nao_rl_bridge.core.deployment_spec import load_deployment
from nao_rl_bridge.core.gait import ReferenceGait
from nao_rl_bridge.core.policies import PGRL_PARAM_NAMES, PGRLParams
from nao_rl_bridge.core.protocol import perturbation_offset

N_STEPS = 200   # warm-up (60) + whole Webots clip (78) + held last frame


def _run_residual(env, spec, rng, offset):
    nao = FakeNao()
    attach_fake_sim(env, nao, FakeWorld())
    lo, hi = env._leg_lower, env._leg_upper
    if offset is not None:                         # what make_evaluation_env_class does
        env._leg_home = np.clip(env._leg_home + offset, lo, hi).astype(np.float32)
    random_sim_state(rng, nao, lo, hi)
    actions = rng.uniform(-1.3, 1.3, (N_STEPS, 12))     # beyond ±1 to test clipping
    ctl = ResidualController(spec, ScriptedPolicy(actions), leg_offset_rad=offset)

    obs_env = env._get_obs()
    for k in range(N_STEPS):
        out = ctl.step(robot_state_from_sim(nao))
        np.testing.assert_allclose(out.obs, obs_env, atol=2e-5, err_msg=f"obs, step {k}")
        pending = FakeNao()
        random_sim_state(rng, pending, lo, hi)

        def physics(p=pending):
            nao.pos, nao.quat, nao.joints = p.pos, p.quat, p.joints
            nao.ang_vel, nao.lin_vel = p.ang_vel, p.lin_vel
        env._world.on_step = physics
        obs_env, *_ = env.step(out.action)
        np.testing.assert_allclose(out.leg_targets, nao.applied[-1][:12], atol=2e-5,
                                   err_msg=f"targets, step {k}")
        assert env._step_count == out.step
    return ctl


@pytest.mark.parametrize("delta", [0.0, 4.0])
def test_ppo_baseline_legacy_observation_and_targets(in_dir, delta):
    in_dir(PPO_DIR)
    env = load_module("ppo_env", PPO_DIR / "nao_walking_env.py").NaoWalkingEnv()
    spec = load_deployment(PPO_DIR / "deployment.yaml")
    offset = perturbation_offset(delta, 0, 3) if delta else None
    _run_residual(env, spec, np.random.default_rng(1), offset)


def test_sac_v3_observation_and_targets(in_dir, tmp_path):
    spec = load_deployment(write_spec(tmp_path, sac_spec_dict()))
    in_dir(SAC_DIR)
    env = load_module("sac_env", SAC_DIR / "nao_walking_env.py").NaoWalkingEnv()
    _run_residual(env, spec, np.random.default_rng(2), None)


def test_legacy_and_correct_conventions_really_differ(tmp_path):
    """Guard: if both profiles gave the same obs, the PPO test would prove nothing."""
    from nao_rl_bridge.core.observation import ObservationBuilder
    rng = np.random.default_rng(3)
    nao = FakeNao()
    random_sim_state(rng, nao, -np.ones(12), np.ones(12))
    st = robot_state_from_sim(nao)
    a = ObservationBuilder(load_deployment(write_spec(tmp_path, spec_dict(), "a.yaml"))).build(st, np.zeros(2))
    b = ObservationBuilder(load_deployment(write_spec(tmp_path, sac_spec_dict(), "b.yaml"))).build(st, np.zeros(2))
    assert np.abs(a[:3] - b[:3]).max() > 1e-2


@pytest.mark.parametrize("delta", [0.0, 2.0])
def test_pgrl_controller_matches_pgrl_env(in_dir, tmp_path, delta):
    spec = load_deployment(write_spec(tmp_path, pgrl_spec_dict(tmp_path / "run")))
    in_dir(PPO_DIR)    # PGRLWalkingEnv's base class resolves the gait from cwd
    env = load_module("pgrl_env_mod", PGRL_DIR / "pgrl_env.py").PGRLWalkingEnv()
    nao = FakeNao()
    attach_fake_sim(env, nao, FakeWorld())
    rng = np.random.default_rng(4)
    phys = np.array([0.9, 0.2, -1.0, -0.7, 1.27, 1.09, 0.78, 1.18, 1.05, 0.93, 1.14, 1.17])
    env.set_params(phys=phys)
    offset = perturbation_offset(delta, 0, 5) if delta else None
    if offset is not None:
        env._leg_home = np.clip(env._leg_home + offset, env._leg_lower, env._leg_upper)
    ctl = PGRLController(spec, PGRLParams(dict(zip(PGRL_PARAM_NAMES, phys))), offset)
    for k in range(N_STEPS):
        random_sim_state(rng, nao, env._leg_lower, env._leg_upper)
        env._step_count += 1                      # as PGRLWalkingEnv.step()
        expected = env.compute_leg_targets()
        out = ctl.step(robot_state_from_sim(nao))
        np.testing.assert_allclose(out.leg_targets, expected, atol=2e-5, err_msg=f"step {k}")
        assert out.in_warmup == (env._step_count < env.gait_warmup_steps)

    # Stiffness: the env scales kp/kd of leg joint i by phys[6 + i % 6].
    expected_s = np.array([phys[6 + i % 6] for i in range(12)])
    np.testing.assert_allclose(ctl.leg_stiffness(base=1.0, max_value=10.0), expected_s)
    assert ctl.leg_stiffness(base=1.0, max_value=1.0).max() <= 1.0


def test_hold_last_gait_matches_webots_reference_gait(in_dir):
    in_dir(PPO_DIR)
    ref = load_module("wrg", PPO_DIR / "webots_reference_gait.py").WebotsReferenceGait()
    gait = ReferenceGait(load_deployment(PPO_DIR / "deployment.yaml"))
    for t in np.linspace(0, 4.0, 401):
        np.testing.assert_allclose(gait.targets(t), ref.get_targets(0.0, t_since_warmup=t), atol=1e-6)
        assert abs(gait.phase(t) - ref.get_phase(t)) < 1e-9


def test_wrap_gait_matches_looped_gait(in_dir, tmp_path):
    spec = load_deployment(write_spec(tmp_path, pgrl_spec_dict()))
    in_dir(PPO_DIR)
    mod = load_module("pgrl_env_mod2", PGRL_DIR / "pgrl_env.py")
    ref = mod.LoopedGait(joint_order=list(spec.leg_order))
    gait = ReferenceGait(spec)
    assert abs(gait.duration - ref.duration) < 1e-9
    for t in np.linspace(0, 2.5 * ref.duration, 3001):
        np.testing.assert_allclose(gait.targets(t), ref.targets(t), atol=1e-9)


def test_perturbation_matches_evaluation_script():
    from conftest import REPO_ROOT
    ev = load_module("evaluate_algorithms",
                     REPO_ROOT / "experiments/test_evaluation_script/evaluate_algorithms.py")
    for k in range(10):
        np.testing.assert_array_equal(
            perturbation_offset(2.0, 0, k), np.radians(2.0) * ev.perturbation_direction(0, k, 12))
