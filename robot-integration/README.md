# Robot Integration

Runs the walking policies trained in `walking-solutions/` (PPO, SAC, PGRL)
on the real NAO through ROS 2.

```
 NAO (NAOqi) ──► naoqi_driver2 ──► /joint_states, /imu/torso
                                         │
                              ┌──────────▼───────────┐
                              │ nao_policy_node      │  30 Hz
                              │  observation builder │  (as declared in deployment.yaml)
                              │  policy (numpy)      │  PPO / SAC / PGRL
                              │  gait + action map   │
                              │  safety supervisor   │
                              └──────────┬───────────┘
                                         ▼
                       /joint_angles ──► naoqi_driver2 ──► ALMotion.setAngles
            (stiffness, rest, posture: direct NAOqi `qi` session)
```

Setup: the node runs on a laptop with ROS 2 Jazzy, connected to the robot
over Ethernet. Runs follow the report's evaluation protocol (straight 1 m
course, §5.1.1), with an operator next to the robot.

## How a solution reaches the robot

1. **Contract.** Each `walking-solutions/<algo>/<variant>/` folder has a
   `deployment.yaml` describing how the policy was trained. See
   [docs/deployment-contract.md](docs/deployment-contract.md).
2. **Export.** *(next step)* Run in the Isaac/SB3 environment. It turns the
   trained model into plain NumPy weights plus a JSON file, and checks that
   they reproduce the original policy's actions.
3. **Deploy.** *(later step)* The ROS 2 node loads the exported policy and
   drives the robot.

Exported policies go in `exported/` and are not committed.

## Layout

```
robot-integration/
├── docs/deployment-contract.md     # deployment.yaml schema
├── nao_rl_bridge/                  # ROS 2 package (pure-numpy core + node)
│   ├── nao_rl_bridge/core/
│   │   └── deployment_spec.py      # load + validate deployment.yaml
│   └── test/
└── tools/validate_deployment.py    # check every solution's deployment.yaml
```

## Status

| Step | Content | State |
|---|---|---|
| A | Deployment contract + validator + PPO baseline `deployment.yaml` | done |
| B | Export tools (SB3 → npz/json, PGRL run → json, action-equivalence check) | pending |
| C | Core runtime: observation profiles, estimators, policies, action map, safety | pending |
| D | ROS 2 node, NAOqi backend, launch/config, run logging | pending |
| E | Sensor-convention check tool, deployment procedure, report text | pending |

## Checks

```bash
python3 robot-integration/tools/validate_deployment.py
python3 -m pytest robot-integration/nao_rl_bridge/test -q
```
