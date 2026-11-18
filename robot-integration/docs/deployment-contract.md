# Deployment contract (`deployment.yaml`, schema v1)

Every solution in `walking-solutions/<algorithm>/<variant>/` that is meant to
run on the real NAO ships a `deployment.yaml` next to its code. It states,
as data, everything the robot side has to reproduce exactly as it was in
training. The robot side (`robot-integration/`) reads **only** this file and
the artefacts it points to; it never imports the Isaac Sim environment.

Why: the training env needs Isaac Sim (Python 3.10) and changes between
experiments, while the robot runs ROS 2 Jazzy (Python 3.12) with no
simulator. A policy only works on the robot if its inputs are built exactly
as in training, so those choices must be written down rather than implied
by the env code.

Implementation and validation:
`nao_rl_bridge/nao_rl_bridge/core/deployment_spec.py`.
Check every solution with:

```bash
python3 robot-integration/tools/validate_deployment.py                     # all solutions
python3 robot-integration/tools/validate_deployment.py --require-artifacts # also require trained models
```

**Keep it in sync.** If the env of a solution changes (limits, home pose,
gait, observation), its `deployment.yaml` must change in the same commit.

## Fields

Relative paths are resolved from the folder that contains the YAML; `~` is
expanded. Angles in the file are in **degrees**; the loader converts to
radians.

| Field | Meaning |
|---|---|
| `schema_version` | `1` |
| `name` | Identifier used in exported files and run logs |
| `algorithm` | `ppo` \| `sac` \| `pgrl` |
| `description` | Free text; mention anything unusual about training |
| `policy.format` | `sb3` (PPO, SAC) or `pgrl_run` (PGRL) |
| `policy.sb3_class` | `PPO` \| `SAC` (sb3 only, must match `algorithm`) |
| `policy.model` | SB3 `.zip` (sb3 only; gitignored, may not exist yet) |
| `policy.run_dir` | PGRL run folder with `state.json` + `config.json` (pgrl only) |
| `control.control_hz` | Policy rate (30 Hz in all current envs) |
| `control.warmup_steps` | Steps holding the home pose before the gait starts |
| `control.max_episode_steps` | Episode limit (also the evaluation limit) |
| `joints.leg_order` | The 12 leg joints in policy order (action and observation) |
| `joints.limits_deg` | `[lower, upper]` per leg joint, as used for clipping and normalisation |
| `joints.home_pose_deg` | Leg pose during warm-up; must equal the gait's first frame (±0.5°) |
| `joints.fixed_pose_deg` | Pose of non-leg joints, held for the whole run |
| `reference_gait.keyframes / timestamps / joint_names` | Gait clip (`.npy`, `.npy`, one name per line) |
| `reference_gait.after_end` | `hold_last` (literal Webots replay) or `wrap` (looped gait) |
| `reference_gait.phase_period_s` | Period of the phase signal; `null` = clip length |
| `action.type` | `residual_on_gait`: `q* = clip(q_ref(t) + scale_rad·clip(a,−1,1), limits)` (sb3 only) |
| `action.scale_rad` | Correction for `|a| = 1` |
| `observation.quaternion_convention` | `wxyz` (correct) or `legacy_xyzw` (the pre-fix bug, see below) |
| `observation.spawn_yaw_deg` | World yaw of the robot at spawn (180: faces world −x) |
| `observation.nonfinite_to_zero` | Replace NaN/inf with 0 before the policy |
| `observation.components` | Ordered list; see vocabulary below |
| `pgrl.controller` | `tilt_compensator_v1` (pgrl only) |

### Observation vocabulary

| Component | Dim | Options | Real-robot source |
|---|---|---|---|
| `projected_gravity` | 3 | — | IMU roll/pitch (+ integrated yaw for `legacy_xyzw`) |
| `angular_velocity` | 3 | `scale` | Gyroscope (body frame) |
| `linear_velocity` | 2 | `frame`: `world_xy` \| `heading` | **Estimated** (stance-leg kinematics + IMU) |
| `leg_joint_positions` | 12 | `normalization`: `limits_to_unit` | Joint encoders, `2(q−mid)/(max−min)` |
| `base_height` | 1 | — | **Estimated** (forward kinematics, stance foot → `base_link`) |
| `gait_phase` | 2 | `warmup_value` | Clock: `sin/cos(2π·t/period)`; `warmup_value` during warm-up |

`linear_velocity.frame`:
- `world_xy`: world-frame (vx, vy). With `spawn_yaw_deg: 180`, walking
  forward gives **negative** vx. Used by the PPO `baseline_test24`.
- `heading`: (forward, lateral) in the yaw-only frame. Used by SAC v3.

### The `legacy_xyzw` convention

Isaac Sim returns quaternions as (w, x, y, z). Every env up to and including
PPO `baseline_test24` unpacked them as (x, y, z, w) (see
`experiments/test_26_sac/APPROACH.md`, "Quaternion convention fix"). The
policy learned from those distorted inputs (side tilt invisible, pitch sign
flipped, yaw showing up as tilt), so on the robot the same distortion must
be reproduced on purpose: the true orientation is rebuilt from the IMU and
passed through the same faulty function. Retrained policies use `wxyz`.

## Templates for the SAC and PGRL folders

SAC (differences from the PPO file only):

```yaml
algorithm: sac
policy: {format: sb3, sb3_class: SAC, model: runs/<run>/final_model.zip}
observation:
  quaternion_convention: wxyz
  components:
    - {name: linear_velocity, dim: 2, frame: heading}   # others as in PPO
```

PGRL (no `action` / `observation` sections; the controller is fixed code,
its 12 parameters and their ranges come from the run's `config.json`):

```yaml
algorithm: pgrl
policy: {format: pgrl_run, run_dir: runs/<run>}
reference_gait:
  keyframes:   <looped gait>/looped_keyframes.npy
  timestamps:  <looped gait>/looped_timestamps.npy
  joint_names: <looped gait>/looped_joint_names.txt
  after_end: wrap
  phase_period_s: 2.64
pgrl:
  controller: tilt_compensator_v1
```

`tilt_compensator_v1` (from `experiments/test_28/pgrl_env.py`):
`q* = clip(q_home + a·(q_ref(f·t) − q_home) + Δq, limits)`, where Δq adds
`K_ank_pitch·g_x` / `K_hip_pitch·g_x` to ankle/hip pitch and
`K_ank_roll·g_y` / `K_hip_roll·g_y` to ankle/hip roll on both legs.
(g_x, g_y) is projected gravity in the correct `wxyz` convention. The six
stiffness factors are mapped to NAOqi stiffness, clamped to 1.0.
