# NAO Walking Project

Training a NAO humanoid robot to walk using reinforcement learning in
NVIDIA Isaac Sim, as an academic investigation project.

## Objective

Train a NAO humanoid to perform forward walking on flat ground using RL,
starting from zero prior experience with physics simulators, Isaac Sim,
or hands-on RL implementation. Primary target: stable forward walking.
Possible later extensions: turning, obstacle avoidance, multi-robot
parallel training.

## Background

NAO is a small humanoid robot (42 controllable joints) commonly used in
robotics research and education. This project imports a NAO URDF model
into NVIDIA Isaac Sim, builds a Gym-compatible RL environment around it
(joint-angle/velocity/orientation observations, joint-position actions),
and trains policies (starting with PPO) to produce a stable forward gait.

## Repository Structure

```
nao-walking-project/
├── CLAUDE.md                    # session-to-session project context (source of truth for status)
├── docs/
│   ├── environment-setup/       # how to reproduce the dev environment from scratch
│   │   └── README.md            #   Isaac Sim install, NAO URDF/mesh sourcing, troubleshooting
│   └── experiment-history.md    # full narrative of all ~25 training experiments
├── experiments/                 # archival tier: code + NOTES.md for the pivotal experiments
│   ├── test_01_standing/        #   only the turning-point experiments are migrated in full —
│   ├── test_04_position/        #   see docs/experiment-history.md for the complete narrative,
│   ├── ...                      #   including experiments not migrated as folders
│   └── test_24/
└── walking-solutions/           # working/promoted tier: validated results meant to be built on
    ├── README.md                #   how this tier is organized, how to add a 2nd algorithm
    ├── shared/                  #   robot URDF/meshes, materials, reference-gait data —
    │                            #   one canonical copy, referenced by every algorithm below
    └── ppo/
        └── baseline_test24/     #   current working forward-walking baseline
```

`experiments/` and `walking-solutions/` serve different purposes: the
former is a historical record (what was tried, what broke, what was
learned — the reasoning behind decisions that isn't visible from final
code alone), the latter is what you'd actually run or build on today.

## Current Status

- **Environment & robot setup:** documented and reproducible — see
  `docs/environment-setup/` for the full Isaac Sim installation and NAO
  URDF/mesh integration process.
- **Experiment history:** ~25 training experiments run to date (physics
  stability → pure RL → hand-designed reference gait → literal Webots
  motion-capture gait). Full narrative in `docs/experiment-history.md`;
  the 12 pivotal experiments are also migrated as runnable code +
  learnings under `experiments/`.
- **Working baseline:** a single-robot PPO policy (promoted from
  `test_24`) that walks forward, currently ~20.9M of a 50M-step training
  target. Lives at `walking-solutions/ppo/baseline_test24/`, referencing
  shared robot/gait assets rather than duplicating them.
- **Next:** develop and evaluate a second RL algorithm alongside PPO for
  comparison (see `CLAUDE.md` phase 6), under a new
  `walking-solutions/<algorithm>/` folder.
- A separate, unvalidated refactor toward a modular multi-robot training
  structure (`test_25` in the original experiment set) was paused and is
  intentionally not part of this migration.

## Getting Oriented

See `CLAUDE.md` for full project context: role/working-style conventions,
system specs, phase breakdown, and session-to-session status tracking.
That file is the source of truth for where the project currently stands.
