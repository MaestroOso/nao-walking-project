"""
robot_config.py — hardware-side settings (config/robot.yaml).

deployment.yaml says how a policy was TRAINED; robot.yaml says how THIS
robot is driven (topics, speeds, stiffness, safety limits, protocol). The
file is merged over DEFAULTS; unknown keys are rejected so a typo cannot
silently fall back to a default.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .command import HIP_YAW_PITCH_MODES

DEFAULTS = {
    "robot": {"ip": "nao.local", "port": 9559},
    "topics": {
        "joint_states": "/joint_states",
        "imu": "/imu/torso",
        "joint_angles": "/joint_angles",
    },
    "command": {
        "speed_fraction": 1.0,          # JointAnglesWithSpeed.speed (fraction of max motor speed)
        "hip_yaw_pitch_mode": "mean",
    },
    "stiffness": {
        "legs": 1.0,
        "arms_head": 0.8,
        "max": 1.0,                     # NAOqi maximum; PGRL factors above it are clamped
        "ramp_s": 1.0,
    },
    "prepare": {
        "disable_autonomous_life": True,
        "fall_manager_enabled": True,   # keep NAOqi's own fall protection on
        "smart_stiffness_enabled": False,
        "move_duration_s": 3.0,         # interpolation to the warm-up pose
        "settle_s": 1.0,
        "max_pose_error_deg": 5.0,      # abort if the robot did not reach the pose
    },
    "estimator": {
        "velocity_filter_alpha": 0.5,
        "height_offset_m": 0.0025,   # sim foot box bottom is 2.5 mm below the URDF sole
    },
    "safety": {
        "max_joint_step_deg": 12.0,
        "fall_tilt_deg": 40.0,
        "min_base_height_m": 0.20,
        "sensor_timeout_s": 0.1,
        "max_consecutive_overruns": 5,
    },
    "protocol": {
        "distance_m": 1.0,
        "auto_finish_on_odometry": False,
        "base_seed": 0,
    },
    "finish": {
        "on_success": "home",           # home | freeze | rest
        "on_timeout": "home",
        "on_fall": "rest",
        "on_aborted": "freeze",
        "home_duration_s": 1.5,
    },
    "logging": {"run_root": "~/nao_robot_runs"},
}

FINISH_ACTIONS = ("home", "freeze", "rest")


class RobotConfigError(ValueError):
    pass


def _merge(base: dict, override: dict, where: str):
    for k, v in override.items():
        if k not in base:
            raise RobotConfigError(f"unknown key '{where}{k}'")
        if isinstance(base[k], dict):
            if not isinstance(v, dict):
                raise RobotConfigError(f"'{where}{k}' must be a mapping")
            _merge(base[k], v, f"{where}{k}.")
        else:
            base[k] = v


def load_robot_config(path=None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        raw = yaml.safe_load(Path(path).expanduser().read_text()) or {}
        if not isinstance(raw, dict):
            raise RobotConfigError(f"{path}: top level must be a mapping")
        _merge(cfg, raw, "")
    c = cfg
    if c["command"]["hip_yaw_pitch_mode"] not in HIP_YAW_PITCH_MODES:
        raise RobotConfigError(f"command.hip_yaw_pitch_mode must be one of {HIP_YAW_PITCH_MODES}")
    if not 0.0 < float(c["command"]["speed_fraction"]) <= 1.0:
        raise RobotConfigError("command.speed_fraction must be in (0, 1]")
    for k in ("legs", "arms_head", "max"):
        if not 0.0 <= float(c["stiffness"][k]) <= 1.0:
            raise RobotConfigError(f"stiffness.{k} must be in [0, 1]")
    for k in ("on_success", "on_timeout", "on_fall", "on_aborted"):
        if c["finish"][k] not in FINISH_ACTIONS:
            raise RobotConfigError(f"finish.{k} must be one of {FINISH_ACTIONS}")
    if float(c["protocol"]["distance_m"]) <= 0:
        raise RobotConfigError("protocol.distance_m must be > 0")
    return cfg
