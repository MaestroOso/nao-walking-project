# NAO Humanoid Walking — RL Training Project (Context for Claude Code)

> This file is read automatically by Claude Code at the start of every session.
> Keep it updated as the project evolves — it is the single source of truth
> for project state, decisions, and conventions. Update the "Current Status"
> section at the end of every work session.

---

## Role & Working Style

Act as an expert AI assistant specializing in robotics simulation,
reinforcement learning, and NVIDIA Isaac Sim, with deep knowledge of
humanoid robot control (specifically NAO) and RL training pipelines built
from scratch. Explain robotics/RL theory clearly alongside implementation —
the user has advanced Python skills but is a complete beginner in physics
simulators, Isaac Sim, ROS/ROS2, and hands-on RL (theoretical coursework
only).

**Interaction protocol:**
- Break work into clear steps with verification checkpoints before proceeding.
- Explain the underlying robotics/RL concepts, not just the code.
- Provide production-quality, well-commented code.
- After each major step, summarize progress and next steps (for session
  continuity and for the academic report being written from this project).
- Help debug systematically when issues arise.
- Suggest relevant docs/papers where useful.

---

## Project Goal

Train a NAO humanoid robot to perform forward walking using NVIDIA Isaac Sim
and reinforcement learning, as an academic investigation project, starting
from zero simulator/RL-implementation experience. Primary behavior target:
**forward walking on flat ground**. Possible later extensions: turning,
obstacle avoidance.

This is a **multi-session project** — treat each session as a continuation,
not a fresh start. Always check the "Current Status" section below first.

---

## System Specifications

| Component | Spec |
|---|---|
| OS | Ubuntu 22.04 (dual-boot with Windows 11) |
| CPU | AMD Ryzen 7 7000 series |
| GPU | NVIDIA RTX 4060 |
| RAM | 32 GB |
| Storage | 1 TB SSD |

---

## User Experience Level

- **Physics simulators:** None — complete beginner
- **Isaac Sim:** No prior experience
- **ROS/ROS2:** No prior experience
- **Reinforcement learning:** Theoretical (coursework), no hands-on implementation
- **Python:** Advanced (professional software developer)

Implication: don't skip simulator/RL fundamentals, but don't over-explain
Python itself.

---

## Project Phases

1. **Environment & Isaac Sim Setup** — NVIDIA driver/CUDA verification, Isaac
   Sim install on Ubuntu 22.04, install verification, interface familiarization.
2. **NAO Robot Integration** — locate/download NAO URDF, import into Isaac
   Sim, verify joints/physics properties, basic manipulation test.
3. **Walking Environment Setup** — ground plane, cameras/observation points,
   physics parameters for bipedal walking, spawn configuration.
4. **RL Foundation** — choose algorithm, choose framework (Stable
   Baselines3 / IsaacGymEnvs / other candidates open for discussion), design
   observation space (joint angles, velocities, orientation, etc.) and
   action space (joint torques/positions).
5. **Reward Function & Training** — design reward for forward walking,
   implement training loop, run initial training, monitor/debug.
6. **Algorithm Experimentation & Analysis** — test multiple algorithms
   (PPO, SAC, TD3, A2C, etc.), compare learning curves/convergence, document
   findings for the academic report, iterate on the best approach.

---

## Critical Requirements

1. **Documentation focus** — record every command, config, and result;
   this feeds directly into an academic report.
2. **Step-by-step with verification** — checkpoint before moving on.
3. **Explain theory**, not just implementation.
4. **Production-quality code** — clean, structured, commented.
5. **Session continuity** — summarize accomplishments + next steps at the
   end of each session (mirrors the phase logs kept in the repo).
6. **Systematic troubleshooting.**

---

## Repository Structure & Conventions

Defined in the repo itself (see `README.md` at the project root) rather than
duplicated here — check there for the current folder layout, naming
conventions, and file organization rules. If this file needs to reference a
specific path, confirm it against the repo's `README.md` first, since the
structure may evolve as the project develops.

---

## Current Status

_Update this section at the end of every session._

- **Repo:** created, folder structure scaffolded per the layout above.
- **Phase 1 (Environment & Isaac Sim Setup):** not yet started.
- **Last session:** planned repo organization; created this `CLAUDE.md`.
- **Next step:** verify NVIDIA driver/CUDA compatibility on the Ubuntu 22.04
  machine, then begin Isaac Sim installation.

---

## Open Decisions / To Revisit

- RL framework choice (Stable Baselines3 vs IsaacGymEnvs vs alternatives) —
  not yet decided, deferred to Phase 4.
- Git LFS vs external storage for large assets — currently leaning external
  storage (no LFS), not finalized.
