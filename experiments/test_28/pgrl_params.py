"""
pgrl_params.py — the 12-parameter PGRL policy (report §4.5.3, Table 4-6).

Pure Python / NumPy (no Isaac Sim imports), so it can be used by the training
loop, by analysis scripts and by the report tables alike.

The policy is a vector theta in [0, 1]^12 (normalised). PGRL perturbs and
updates it in that normalised space; `to_physical()` maps it to the values the
controller actually uses.

Parameter order (fixed — every log / checkpoint uses it):
    0  K_ank_pitch   ankle-pitch gain on forward tilt g_x        [rad/rad]
    1  K_hip_pitch   hip-pitch gain on forward tilt g_x          [rad/rad]
    2  K_ank_roll    ankle-roll gain on lateral tilt g_y         [rad/rad]
    3  K_hip_roll    hip-roll gain on lateral tilt g_y           [rad/rad]
    4  f             gait frequency factor (playback speed)      [-]
    5  a             gait amplitude factor (around home pose)    [-]
    6  s_hip_yawpitch  stiffness factor, HipYawPitch (both legs) [-]
    7  s_hip_roll      stiffness factor, HipRoll                 [-]
    8  s_hip_pitch     stiffness factor, HipPitch                [-]
    9  s_knee_pitch    stiffness factor, KneePitch               [-]
   10  s_ankle_pitch   stiffness factor, AnklePitch              [-]
   11  s_ankle_roll    stiffness factor, AnkleRoll               [-]

Torso-tilt compensator (decision 2026-11-17): the gains act on the horizontal
components of projected gravity measured by the torso IMU, instead of the ZMP
error. g_x/g_y are ~sin(tilt), so the gains are rad/rad and the range is
[-1, 1] (a 0.1 rad lean gives at most a 0.1 rad correction).
"""
import numpy as np

# (name, physical_low, physical_high, physical_initial, unit)
PARAM_SPECS = [
    ("K_ank_pitch",     -1.0, 1.0, 0.0, "rad/rad"),
    ("K_hip_pitch",     -1.0, 1.0, 0.0, "rad/rad"),
    ("K_ank_roll",      -1.0, 1.0, 0.0, "rad/rad"),
    ("K_hip_roll",      -1.0, 1.0, 0.0, "rad/rad"),
    ("f",                0.7, 1.3, 1.0, "-"),
    ("a",                0.7, 1.3, 1.0, "-"),
    ("s_hip_yawpitch",   0.2, 1.2, 1.0, "-"),
    ("s_hip_roll",       0.2, 1.2, 1.0, "-"),
    ("s_hip_pitch",      0.2, 1.2, 1.0, "-"),
    ("s_knee_pitch",     0.2, 1.2, 1.0, "-"),
    ("s_ankle_pitch",    0.2, 1.2, 1.0, "-"),
    ("s_ankle_roll",     0.2, 1.2, 1.0, "-"),
]

N_PARAMS = len(PARAM_SPECS)
NAMES = [p[0] for p in PARAM_SPECS]
LOW = np.array([p[1] for p in PARAM_SPECS], dtype=np.float64)
HIGH = np.array([p[2] for p in PARAM_SPECS], dtype=np.float64)
INIT_PHYS = np.array([p[3] for p in PARAM_SPECS], dtype=np.float64)

# Index helpers used by the controller.
IDX_GAINS = slice(0, 4)
IDX_F, IDX_A = 4, 5
IDX_STIFF = slice(6, 12)


def to_physical(theta):
    """Normalised theta in [0,1]^12 -> physical parameter values."""
    theta = np.clip(np.asarray(theta, dtype=np.float64), 0.0, 1.0)
    return LOW + theta * (HIGH - LOW)


def to_normalized(phys):
    """Physical parameter values -> normalised theta in [0,1]^12."""
    phys = np.asarray(phys, dtype=np.float64)
    return np.clip((phys - LOW) / (HIGH - LOW), 0.0, 1.0)


def initial_theta():
    """Starting policy: gains 0, f = a = 1, stiffness 1 (= the test_27 run)."""
    return to_normalized(INIT_PHYS)


def as_dict(theta):
    """{name: physical value} for logging."""
    return {n: float(v) for n, v in zip(NAMES, to_physical(theta))}
