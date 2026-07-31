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

## Current Status

- ~25 training experiments run to date, iterating on physics stability,
  pose selection, reward shaping, and reference-gait guidance.
- The most recent confirmed-working setup is a single-robot training
  pipeline (physics stable, reward function producing a working
  forward-walking behavior).
- A code refactor toward a more modular, multi-algorithm structure has
  been started, aimed at eventually supporting multiple robots training
  in parallel for faster data collection. This refactor is not yet
  validated against the working single-robot pipeline and is paused for
  now.
- The repository is being migrated from local working files into version
  control; project history and folder structure are still being organized.

## Getting Oriented

See `CLAUDE.md` for full project context: role/working-style conventions,
system specs, phase breakdown, and session-to-session status tracking.
That file is the source of truth for where the project currently stands.
