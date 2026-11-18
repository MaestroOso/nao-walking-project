# Robot Integration

Runs the walking policies trained in `walking-solutions/` (PPO, SAC, PGRL)
on the real NAO through ROS 2, with the report's evaluation protocol
(straight 1 m course, δ perturbations, F_S and fall rate, §5.1).

```
 NAO (NAOqi) ──► naoqi_driver2 ──► /joint_states, /imu/torso
                                         │
                              ┌──────────▼───────────┐
                              │ nao_policy_node      │  30 Hz
                              │  state estimator     │  FK height + velocity, IMU yaw
                              │  observation builder │  exactly as in training
                              │  policy (NumPy)      │  PPO / SAC / PGRL
                              │  gait + action map   │  HipYawPitch rule
                              │  safety supervisor   │  rate limit, fall, watchdogs
                              └──────────┬───────────┘
                                         ▼
                       /joint_angles ──► naoqi_driver2 ──► ALMotion.setAngles
            stiffness, postures, rest, e-stop: direct NAOqi `qi` session
```

Setup: the node runs on a laptop with ROS 2 Jazzy, connected to the robot
over Ethernet, with an operator next to the robot.

## Workflow

| # | Step | Where | Command |
|---|---|---|---|
| 1 | Each solution declares how it was trained | `walking-solutions/<algo>/<variant>/deployment.yaml` | `python3 robot-integration/tools/validate_deployment.py` |
| 2 | Export the trained policy (verified against SB3) | Isaac/SB3 env | `setup/isaac_env/bin/python robot-integration/export/export_policy.py walking-solutions/<algo>/<variant>` |
| 3 | (optional) Check the estimators in simulation | Isaac env, then system Python | `export/record_sim_traces.py`, `tools/check_sim_traces.py` |
| 4 | Check the robot's sensor conventions | laptop + robot | `ros2 run nao_rl_bridge check_sensors --ros-args -p export_dir:=...` |
| 5 | Dry run (motors untouched) | laptop + robot | `ros2 launch nao_rl_bridge deploy.launch.py export_dir:=... nao_ip:=...` |
| 6 | Protocol runs | laptop + robot | same, with `dry_run:=false`, plus `ros2 run nao_rl_bridge walk_operator` |
| 7 | Measurements and summary | laptop | `tools/record_measurement.py`, `tools/summarize_robot_runs.py` |

The full robot-session checklist is in
[docs/deployment-procedure.md](docs/deployment-procedure.md); the design and
its sim-to-real assumptions are in [docs/sim-to-real.md](docs/sim-to-real.md);
the `deployment.yaml` schema is in
[docs/deployment-contract.md](docs/deployment-contract.md).

## Installation (laptop, Ubuntu 24.04 + ROS 2 Jazzy)

```bash
# 1. NAOqi ROS 2 driver and its messages, built from source in a workspace:
#    ros-naoqi/naoqi_driver2 (package naoqi_driver) and naoqi_bridge_msgs.
#    Follow their README; check with:
ros2 pkg list | grep -E "naoqi_driver|naoqi_bridge_msgs"

# 2. NAOqi Python SDK (stiffness / posture / rest / e-stop calls):
pip install qi

# 3. This package (from the repo root):
source /opt/ros/jazzy/setup.bash
colcon build --base-paths robot-integration/nao_rl_bridge
source install/setup.bash
```

Exported policies go to `robot-integration/exported/` (not committed). Run
logs go to `~/nao_robot_runs/` (one folder per repetition: `run.json` +
`ticks.csv`).

## Layout

```
robot-integration/
├── README.md
├── docs/
│   ├── deployment-contract.md      # deployment.yaml schema + SAC/PGRL templates
│   ├── deployment-procedure.md     # checklist for a session with the robot
│   └── sim-to-real.md              # how each training quantity is obtained on the robot
├── export/                         # runs in the Isaac/SB3 env (Python 3.10)
│   ├── export_policy.py            #   solution -> self-contained, verified export folder
│   ├── record_sim_traces.py        #   ground-truth traces from the training env
│   └── test_export.py              #   export tests with throwaway models
├── tools/                          # system Python
│   ├── validate_deployment.py
│   ├── check_sim_traces.py         #   robot-side reconstruction vs sim ground truth
│   ├── record_measurement.py       #   operator's tape-measure readings -> run.json
│   └── summarize_robot_runs.py     #   runs -> protocol table + CSV (sim format)
└── nao_rl_bridge/                  # ROS 2 package (ament_python)
    ├── config/robot.yaml           #   hardware-side settings
    ├── launch/deploy.launch.py
    ├── nao_rl_bridge/
    │   ├── core/                   #   pure NumPy, no ROS: unit-tested
    │   │   ├── deployment_spec.py  #     contract loader/validator
    │   │   ├── exported.py         #     export folder loader (sha256-checked)
    │   │   ├── policies.py         #     MLP (PPO/SAC) and PGRL parameters
    │   │   ├── observation.py      #     observation builder (wxyz / legacy_xyzw)
    │   │   ├── estimators.py       #     yaw, FK torso height, kinematic velocity, odometry
    │   │   ├── kinematics.py       #     leg FK from the URDF
    │   │   ├── gait.py             #     hold_last / wrap gait playback
    │   │   ├── controller.py       #     residual (PPO/SAC) and PGRL controllers
    │   │   ├── command.py          #     HipYawPitch rule, full joint command
    │   │   ├── safety.py           #     rate limit, fall detection, watchdog
    │   │   ├── protocol.py         #     δ perturbation (same seeds as sim), episode runner
    │   │   ├── robot_config.py     #     robot.yaml loader
    │   │   └── run_log.py
    │   ├── robot/naoqi_backend.py  #   qi session + dry-run stand-in
    │   ├── session.py              #   prepare -> 30 Hz loop -> finish (ROS-free)
    │   └── ros/
    │       ├── policy_node.py      #   the node (topics, services, e-stop)
    │       ├── operator_cli.py     #   single-key start / finish line / stop / e-stop
    │       └── check_sensors_node.py
    └── test/                       #   pytest suite
```

## Status

| Part | State |
|---|---|
| Deployment contract + PPO baseline `deployment.yaml` | done |
| Export (SB3 → NumPy, PGRL run → parameters), verified | done; **exports of the final policies pending** |
| Robot runtime (observation, estimators, controllers, safety, protocol) | done |
| ROS 2 node, NAOqi backend, operator CLI, sensor check, launch | done; dry-run smoke-tested with a stand-in robot |
| Measurement + summary tools (report format) | done |
| `deployment.yaml` for the final SAC and PGRL folders | pending (written when those folders are added; templates in the contract doc) |
| Runs on the robot | pending |

## Tests and checks

```bash
python3 robot-integration/tools/validate_deployment.py
python3 -m pytest robot-integration/nao_rl_bridge/test robot-integration/export -q
setup/isaac_env/bin/python robot-integration/export/test_export.py   # SB3 export tests
```

What the tests establish (none uses a trained policy):

- **Fidelity to the training envs.** The real `NaoWalkingEnv` (PPO
  baseline, legacy quaternion handling), the SAC v3 env and
  `PGRLWalkingEnv` are driven with a fake simulator; over 200 steps
  (warm-up, whole gait clip, held last frame), with and without δ
  perturbation, the robot-side observation and joint targets equal the
  env's to 2e-5.
- **Export.** NumPy policies reproduce SB3 `predict(deterministic=True)`
  (max error 1.9e-8 PPO, 2.4e-7 SAC on 4096 observations); tampered or
  unverified exports are refused.
- **Runtime.** Estimators, safety checks, HipYawPitch rule, protocol
  perturbation (identical to `evaluate_algorithms.py`), and complete
  sessions (finish line, fall, timeout, timing overrun, missing sensors,
  pose check) with a fake robot and clock.
- **In Isaac Sim** (zero-action PPO-baseline episode): observation
  reconstruction exact (≤ 1e-6); estimators while upright: torso height
  2.7 ± 0.9 mm offset (→ `height_offset_m: 0.0025`), forward velocity RMSE
  0.017 m/s (r = 0.96), lateral RMSE 0.020 m/s (r = 0.93). See
  [docs/sim-to-real.md](docs/sim-to-real.md).
