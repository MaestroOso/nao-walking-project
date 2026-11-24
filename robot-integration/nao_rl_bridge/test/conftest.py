"""Shared test helpers.

The fidelity tests load the REAL training environments from the repo
(walking-solutions/ppo/baseline_test24, walking-solutions/sac/baseline_test26,
walking-solutions/pgrl/baseline_test28) and drive them with a fake simulator object, so the
robot-side code is compared against the code the policies were trained with.
No Isaac Sim is needed: the envs only create the simulator in reset().
"""
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
ROBOT_DIR = PKG_DIR.parent
REPO_ROOT = ROBOT_DIR.parent
sys.path.insert(0, str(PKG_DIR))

PPO_DIR = REPO_ROOT / "walking-solutions/ppo/baseline_test24"
SAC_DIR = REPO_ROOT / "walking-solutions/sac/baseline_test26"
PGRL_DIR = REPO_ROOT / "walking-solutions/pgrl/baseline_test28"
URDF = REPO_ROOT / "walking-solutions/shared/robot/nao.urdf"
PPO_YAML = PPO_DIR / "deployment.yaml"
SAC_YAML = SAC_DIR / "deployment.yaml"
sys.dont_write_bytecode = True     # keep the training folders free of __pycache__


def _install_gymnasium_stub():
    """Minimal gymnasium (Env, spaces.Box) if the real one is not installed."""
    try:
        import gymnasium  # noqa: F401
        return
    except ImportError:
        pass
    gym = types.ModuleType("gymnasium")
    spaces = types.ModuleType("gymnasium.spaces")

    class Box:
        def __init__(self, low, high, shape, dtype):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    class Env:
        def reset(self, seed=None, options=None):
            return None

    spaces.Box = Box
    gym.Env, gym.spaces = Env, spaces
    sys.modules["gymnasium"] = gym
    sys.modules["gymnasium.spaces"] = spaces


_TRAINING_MODULES = ("nao_walking_env", "webots_reference_gait", "pgrl_params", "pgrl_env")


def load_module(name, path):
    """Import a training-folder module with that folder first on sys.path.

    Each training folder ships its own copy of helper modules (e.g. the two
    webots_reference_gait.py differ in their default paths), so cached copies
    are dropped first and sys.path is restored afterwards.
    """
    _install_gymnasium_stub()
    for m in _TRAINING_MODULES:
        sys.modules.pop(m, None)
    old_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = old_path
    return mod


@pytest.fixture
def in_dir():
    """Context: chdir into a training folder (its env resolves paths from cwd)."""
    old = os.getcwd()

    def _cd(d):
        os.chdir(d)
    yield _cd
    os.chdir(old)


class FakeNao:
    """Stands in for the Isaac Articulation: returns a settable sim state."""

    def __init__(self):
        self.pos = np.array([0.0, 0.0, 0.30])
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])     # Isaac (w, x, y, z)
        self.joints = np.zeros(12)
        self.ang_vel = np.zeros(3)                     # world frame
        self.lin_vel = np.zeros(3)                     # world frame
        self.applied = []

    def get_world_pose(self):
        return self.pos.copy(), self.quat.copy()

    def get_joint_positions(self):
        return self.joints.copy()

    def get_angular_velocity(self):
        return self.ang_vel.copy()

    def get_linear_velocity(self):
        return self.lin_vel.copy()

    def apply_action(self, action):
        self.applied.append(np.array(action, dtype=float))


class FakeWorld:
    def __init__(self, on_step=None):
        self.on_step = on_step

    def step(self, render=False):
        if self.on_step:
            self.on_step()


def attach_fake_sim(env, nao, world):
    """Make a training env usable without Isaac: leg DOFs are indices 0..11."""
    env._nao, env._world = nao, world
    env._ArticulationAction = lambda joint_positions: joint_positions
    env._leg_dof_indices = list(range(12))
    env._full_home_pose = np.zeros(12, dtype=np.float32)
    env._has_ang_vel_api = True
    env._has_lin_vel_api = True
    env._sim_initialized = True
    env._step_count = 0
    env._phase = 0.0
    env._prev_quat = nao.quat.astype(np.float32)
    env._prev_action = np.zeros(12, dtype=np.float32)
    return env


def random_sim_state(rng, nao, leg_lo, leg_hi):
    """A plausible random torso state (tilt up to ~20°, any yaw)."""
    from nao_rl_bridge.core.rotations import quat_from_rpy
    roll, pitch = rng.uniform(-0.35, 0.35, 2)
    yaw = rng.uniform(-np.pi, np.pi)
    nao.quat = quat_from_rpy(roll, pitch, yaw)
    nao.pos = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.2, 0.33)])
    nao.joints = rng.uniform(leg_lo, leg_hi)
    nao.ang_vel = rng.normal(0, 1.5, 3)
    nao.lin_vel = rng.normal(0, 0.4, 3)


def robot_state_from_sim(nao, leg_order_names=None):
    """What the robot-side estimator would produce if its estimates were exact."""
    from nao_rl_bridge.core.estimators import RobotState
    from nao_rl_bridge.core.rotations import rotation_matrix, rpy_from_quat
    roll, pitch, yaw = rpy_from_quat(nao.quat)
    R = rotation_matrix(nao.quat)
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy = nao.lin_vel[:2]
    return RobotState(
        leg_q=nao.joints.copy(), roll=roll, pitch=pitch, yaw_world=yaw,
        quat_world=nao.quat.copy(), gyro_body=R.T @ nao.ang_vel,
        v_heading=np.array([c * vx + s * vy, -s * vx + c * vy]),
        base_height=float(nao.pos[2]), stance_foot="l_sole", odometry=np.zeros(2))


def spec_dict():
    """The PPO baseline deployment.yaml as a dict, with absolute gait paths."""
    import yaml
    d = yaml.safe_load(PPO_YAML.read_text())
    for k in ("keyframes", "timestamps", "joint_names"):
        d["reference_gait"][k] = str((PPO_DIR / d["reference_gait"][k]).resolve())
    return d


def sac_spec_dict():
    """The SAC baseline deployment.yaml as a dict, with absolute gait paths."""
    import yaml
    d = yaml.safe_load(SAC_YAML.read_text())
    for k in ("keyframes", "timestamps", "joint_names"):
        d["reference_gait"][k] = str((SAC_DIR / d["reference_gait"][k]).resolve())
    return d


def pgrl_spec_dict(run_dir="run"):
    looped = REPO_ROOT / "walking-solutions/shared/reference_gaits"
    d = spec_dict()
    d.update(name="pgrl_test", algorithm="pgrl")
    d["policy"] = {"format": "pgrl_run", "run_dir": str(run_dir)}
    d["reference_gait"] = {
        "keyframes": str(looped / "forwards_looped_keyframes.npy"),
        "timestamps": str(looped / "forwards_looped_timestamps.npy"),
        "joint_names": str(looped / "forwards_looped_joint_names.txt"),
        "after_end": "wrap", "phase_period_s": 2.64,
    }
    del d["action"], d["observation"]
    d["pgrl"] = {"controller": "tilt_compensator_v1"}
    return d


def write_spec(tmp_path, d, name="deployment.yaml"):
    import yaml
    p = Path(tmp_path) / name
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


class ScriptedPolicy:
    """Returns pre-drawn actions and records the observations it was given."""

    def __init__(self, actions, obs_dim=23):
        self.actions = list(actions)
        self.obs_dim, self.act_dim = obs_dim, 12
        self.seen = []

    def act(self, obs):
        self.seen.append(np.array(obs))
        return self.actions[len(self.seen) - 1]
