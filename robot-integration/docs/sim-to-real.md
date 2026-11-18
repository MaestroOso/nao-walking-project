# Sim-to-real design notes

A policy is a function of its observation. On the robot it only behaves as
in simulation if every input is built with the same conventions, units,
timing and frames as in training. This document lists how each training
quantity is obtained on the NAO, what was verified, and what remains a
known difference.

## Control loop and timing

| | Simulation (training env) | Robot |
|---|---|---|
| Rate | 30 Hz policy, physics 120 Hz | 30 Hz loop on the laptop (`session.py`), overruns logged; 5 late ticks in a row abort |
| Before the episode | teleport + 85 settle steps | wake up → StandInit → 3 s interpolation to the warm-up pose → 1 s settle → pose check (≤ 5°) |
| Warm-up | 60 steps, reference = home pose, phase input (0, 1), **policy acts** | identical (`ResidualController`) |
| Sequencing | obs after step n carries the phase of step n | identical: the observation of tick n+1 uses the phase of step n |
| Gait after the clip | PPO/SAC: last frame held (`hold_last`); PGRL: looped (`wrap`) | from `deployment.yaml` |
| δ perturbation | `radians(δ)·U(−1,1)^12`, seed `base_seed + k`, clipped to limits; also the centre of PGRL's amplitude scaling | identical (`protocol.py`, tested against `evaluate_algorithms.py`) |

The fake-simulator tests (`test/test_fidelity.py`) run the real training
env classes and the robot-side controllers side by side for 200 steps and
require identical observations and joint targets (≤ 2e-5).

## Observation components

| Component | Robot source | Notes |
|---|---|---|
| Projected gravity | IMU roll/pitch (AngleX/AngleY via `/imu/torso`) + yaw | `wxyz`: yaw-invariant. `legacy_xyzw` (PPO baseline): the true orientation is rebuilt and passed through the env's faulty `_quat_rotate_inverse`, which makes the input depend on yaw |
| Angular velocity | gyroscope (body frame) | the env rotated Isaac's world-frame ω back with the same function, so for `legacy_xyzw` the robot computes `legacy(q, R(q)·ω_body)` |
| Linear velocity | **estimated**: stance-foot kinematics + IMU | `world_xy` (PPO) or `heading` (SAC) frame |
| Leg joint positions | encoders (`/joint_states`) | normalised with the training limits |
| Base height | **estimated**: leg FK from the URDF + 2.5 mm | the sim foot box bottom lies 2.5 mm below the URDF sole |
| Gait phase | clock (step count) | |

**World yaw.** At spawn the robot faces world −x (yaw π). On the robot,
`yaw_world = π + (IMU yaw − IMU yaw at episode start)`. NAO's IMU yaw is an
integrated gyro and drifts slowly; over a 1 m run (a few seconds) the drift
is negligible. Only `legacy_xyzw` and `world_xy` inputs depend on yaw.

**Verified in Isaac Sim** (`tools/check_sim_traces.py`, zero-action
episode of the PPO baseline, 120 steps):

| Check | Result |
|---|---|
| Observation rebuilt from IMU-like inputs vs env observation | max error ≤ 1.1e-6 on every component (legacy convention included) |
| Torso height, FK estimate vs sim (upright samples) | sim − estimate = 2.7 ± 0.9 mm → `height_offset_m: 0.0025` |
| Forward velocity, kinematic estimate vs sim | RMSE 0.017 m/s, r = 0.96 (filter α = 0.5) |
| Lateral velocity | RMSE 0.020 m/s, r = 0.93 |

Samples with torso tilt above 15° are excluded: when the robot is falling
the stance foot is not flat and static, and the robot side stops the run
at 40° anyway. α = 0.5 was the best of {1.0, 0.5, 0.3, 0.2} on the
walking segment. Re-run the check with each final policy (`--export`):
policies that walk faster or slide their feet may give larger errors.

## Actions and actuators

* **Command path.** Targets are published at 30 Hz on `/joint_angles`;
  naoqi_driver2 executes them with `ALMotion.setAngles` at
  `command.speed_fraction` (1.0). In simulation the PD gains (up to 300k)
  made tracking nearly instantaneous. The real motors are slower and
  `setAngles` adds latency, so the measured joints lag the targets. This is
  the largest remaining difference and the main reason to expect worse
  results than in simulation (the report notes that the simulated joints
  can exert more force than the real motors).
* **Rate limit.** Each command may move at most 12° per step (≈ 6.3 rad/s,
  about the NAO's motor speed). Clipped steps are counted
  (`rate_limited_steps` in `run.json`).
* **HipYawPitch.** One motor drives both joints on the NAO; the
  simulator modelled two. The command sends the mean (or the left value);
  the largest L/R disagreement is logged (`max_hip_yaw_gap_deg`).
* **Stiffness.** PPO/SAC legs at 1.0, arms/head at 0.8. PGRL's factors
  s_j scaled kp/kd in simulation; on the robot they scale NAOqi stiffness
  and are clamped to 1.0, so factors above 1 (stiffer than the baseline)
  cannot be reproduced.
* **Smart stiffness** is disabled (constant stiffness, as in simulation);
  NAOqi's fall manager stays enabled.

## Safety and protocol

| | Simulation | Robot |
|---|---|---|
| Fall | height < 0.10 m or rotation > ~145°, after a 50-step grace period | tilt > 40° or estimated torso height < 0.20 m, always active (operator can catch the robot) |
| Finish | distance ≥ D from simulator position | operator presses `f` at the finish line (or optional odometry finish) |
| Distance / lateral | simulator position | tape measure (`record_measurement.py`); odometry estimate logged as fallback |
| Time | steps × dt from end of warm-up | wall clock from end of warm-up |
| Other stops | — | sensor data older than 0.1 s, non-finite values, timing overruns, operator stop, e-stop → `aborted` (not a repetition) |

## Known differences that remain

1. Motor dynamics and `setAngles` latency (see above).
2. Contact and friction of the real floor and feet vs the simulated foot boxes.
3. Velocity and height are estimates (validated in simulation only).
4. IMU noise and latency; NAOqi's AngleX/AngleY filtering.
5. PGRL stiffness factors above 1.0 are clamped.
6. Mass distribution and joint calibration of the individual robot vs the URDF.
