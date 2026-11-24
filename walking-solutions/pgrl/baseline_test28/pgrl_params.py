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
error. g_x/g_y are ~sin(tilt), so the gains are rad/rad.

Gain range history (normalised θ only has a meaning together with its range,
so every run stores the specs it used in runs/<run>/config.json and
`load_run_specs()` reads them back):
  * pgrl_v1:          [-1, 1] rad/rad  (PARAM_SPECS_V1). Three of the four
                      gains ended at a bound (K_ank_pitch = +1,
                      K_ank_roll = K_hip_roll = -1), i.e. the search wanted
                      stronger corrections than allowed.
  * current (v2 on):  [-3, 3] rad/rad  (PARAM_SPECS). A 0.1 rad lean can now
                      produce up to a 0.3 rad correction. Note: with the
                      normalised ε = 0.05, one perturbation of a gain is now
                      0.3 rad/rad instead of 0.1 rad/rad.
"""
import json
from pathlib import Path

import numpy as np

# (name, physical_low, physical_high, physical_initial, unit)
PARAM_SPECS = [
    ("K_ank_pitch",     -3.0, 3.0, 0.0, "rad/rad"),
    ("K_hip_pitch",     -3.0, 3.0, 0.0, "rad/rad"),
    ("K_ank_roll",      -3.0, 3.0, 0.0, "rad/rad"),
    ("K_hip_roll",      -3.0, 3.0, 0.0, "rad/rad"),
    ("f",                0.7, 1.3, 1.0, "-"),
    ("a",                0.7, 1.3, 1.0, "-"),
    ("s_hip_yawpitch",   0.2, 1.2, 1.0, "-"),
    ("s_hip_roll",       0.2, 1.2, 1.0, "-"),
    ("s_hip_pitch",      0.2, 1.2, 1.0, "-"),
    ("s_knee_pitch",     0.2, 1.2, 1.0, "-"),
    ("s_ankle_pitch",    0.2, 1.2, 1.0, "-"),
    ("s_ankle_roll",     0.2, 1.2, 1.0, "-"),
]

# Ranges used by pgrl_v1 (kept so that run can still be evaluated even if its
# config.json were missing).
PARAM_SPECS_V1 = [(n, -1.0, 1.0, i, u) if u == "rad/rad" else (n, lo, hi, i, u)
                  for (n, lo, hi, i, u) in PARAM_SPECS]

N_PARAMS = len(PARAM_SPECS)
NAMES = [p[0] for p in PARAM_SPECS]

# Index helpers used by the controller.
IDX_GAINS = slice(0, 4)
IDX_F, IDX_A = 4, 5
IDX_STIFF = slice(6, 12)


def _arrays(specs=None):
    specs = PARAM_SPECS if specs is None else specs
    low = np.array([p[1] for p in specs], dtype=np.float64)
    high = np.array([p[2] for p in specs], dtype=np.float64)
    init = np.array([p[3] for p in specs], dtype=np.float64)
    return low, high, init


# Physical initial values (identical in every spec version).
INIT_PHYS = _arrays()[2]


def to_physical(theta, specs=None):
    """Normalised theta in [0,1]^12 -> physical parameter values."""
    low, high, _ = _arrays(specs)
    theta = np.clip(np.asarray(theta, dtype=np.float64), 0.0, 1.0)
    return low + theta * (high - low)


def to_normalized(phys, specs=None):
    """Physical parameter values -> normalised theta in [0,1]^12."""
    low, high, _ = _arrays(specs)
    phys = np.asarray(phys, dtype=np.float64)
    return np.clip((phys - low) / (high - low), 0.0, 1.0)


def initial_theta(specs=None):
    """Starting policy: gains 0, f = a = 1, stiffness 1 (= open-loop looped gait)."""
    return to_normalized(_arrays(specs)[2], specs)


def as_dict(theta, specs=None):
    """{name: physical value} for logging."""
    return {n: float(v) for n, v in zip(NAMES, to_physical(theta, specs))}


def load_run_specs(run_dir):
    """Parameter specs a run was trained with (from its config.json).

    Falls back to PARAM_SPECS_V1 for 'pgrl_v1' and to the current
    PARAM_SPECS otherwise."""
    cfg = Path(run_dir) / "config.json"
    if cfg.is_file():
        specs = json.loads(cfg.read_text()).get("param_specs")
        if specs:
            return [tuple(s) for s in specs]
    return PARAM_SPECS_V1 if Path(run_dir).name == "pgrl_v1" else PARAM_SPECS
