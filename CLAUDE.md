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

- **Repo:** structure in place per `README.md` — `docs/` (environment
  setup guide + full experiment history), `experiments/` (archival tier,
  12 pivotal experiments migrated with code + learnings), and
  `walking-solutions/` (working tier: shared robot/gait assets, the PPO
  baseline and the PGRL baseline).
- **Phase 1 (Environment & Isaac Sim Setup):** complete. Isaac Sim 4.5.0
  installed and verified; see `docs/environment-setup/`. Note: actually
  verified on Ubuntu 24.04, not the 22.04 listed in the System
  Specifications table below — see the discrepancy note in
  `docs/environment-setup/README.md`, not yet reconciled.
- **Phase 2 (NAO Robot Integration):** complete. NAO V5.0 imported (83
  joints / 84 links / 42 controllable DOF); see
  `docs/environment-setup/03-nao-robot-integration.md`.
- **Phase 3 (Walking Environment Setup):** complete — see
  `docs/experiment-history.md` phase 1 (`test_01`–`test_05`) for how the
  physics-stability issues were found and fixed.
- **Phase 4 (RL Foundation):** PPO (Stable Baselines3) is the algorithm
  behind the current working baseline. A second algorithm for comparison
  is not yet chosen — see Open Decisions below.
- **Phase 5 (Reward Function & Training):** working baseline established
  at `walking-solutions/ppo/baseline_test24/`, ~20.9M of a 50M-step
  target. Reward formula and full config documented in that folder's
  `README.md`.
- **Phase 6 (Algorithm Experimentation & Analysis):** in progress.
  SAC is promoted to `walking-solutions/sac/baseline_test26/` (run
  `sac_pilot_v5`, 1.23M steps: stays upright headless, ~0.06 m/s, turns,
  fragile in the GUI). A v6 attempt (constant alive bonus, yaw-rate
  penalty 5.0) is uncommitted in `experiments/test_26_sac`; its run
  `sac_pilot_v6` stopped at 225k steps and can be resumed.
  PGRL is implemented and promoted to `walking-solutions/pgrl/baseline_test28/`
  (developed in `experiments/test_27_gait_loop` + `experiments/test_28`,
  run `pgrl_v1`: F_S 0.043 → 0.078, still falls after two steps). A common
  evaluation script for all
  algorithms is in `experiments/test_evaluation_script/` and implements the
  report's protocol (§4.6.2, §5.1): straight course of D m, deterministic
  policy, 10 repetitions per initial-pose perturbation δ ∈ {0,1,2,4}°,
  F_S and fall-rate metrics. It also produces Table 5-3 and Figure 5-3.
- **Report:** `docs/report/` (Spanish, LaTeX PDF). Chapter 5 results are
  mostly still `[PENDIENTE]`.
- **Last session (17 Nov 2026):** built and committed the evaluation
  script. First PPO evaluation runs: there are two copies of the PPO
  `test_24` run (27.2M steps in `~/nao_walking_project/`, the default, and
  22.35M steps in the gitignored `<repo>/nao_walking_project/`). Results and
  caveats are in that folder's `README.md`. The speeds measured (about
  0.46–0.9 m/s) are suspiciously high and need a visual check with `--gui`.
- **Robot integration (18 Nov 2026):** `robot-integration/` deploys the
  policies to the real NAO over ROS 2 Jazzy (`naoqi_driver2` for the
  30 Hz joint stream, a direct `qi` session for stiffness, posture, rest
  and e-stop). Each `walking-solutions/<algo>/<variant>/` needs a
  `deployment.yaml` (written for `ppo/baseline_test24`,
  `pgrl/baseline_test28` and `sac/baseline_test26`). The tools,
  runtime, ROS node and docs are done and tested. Still pending: the
  exports for the final versions (run `export/export_policy.py`) and the
  runs on the robot. Procedure:
  `robot-integration/docs/deployment-procedure.md`.
- **PGRL promotion (23 Nov 2026):** `walking-solutions/pgrl/baseline_test28/`
  is self-contained: it has its own copies of the PPO env, of the evaluation
  episode/F_S code and of `metrics.py`, plus `deployment.yaml`. The looped
  gait is in `walking-solutions/shared/reference_gaits/forwards_looped_*`.
  The evaluation script defaults and the robot-integration tests now point
  there. `runs/` stays gitignored; copy `experiments/test_28/runs/pgrl_v1`
  into it. Checked from the new folder: same headless result (d 0.531 m,
  F_S 0.0813) and the first training iteration reproduces bit-for-bit.
- **SAC promotion (23 Nov 2026):** `walking-solutions/sac/baseline_test26/`
  holds the `experiments/test_26_sac` code at commit `f29cb8a` (run
  `sac_pilot_v5`), with paths pointed at `shared/`. Checked from the new
  folder: same headless result (1500 steps, reward 4037.6), both verify
  scripts pass, the export matches SB3 (1.4e-5), and the common evaluation
  completes the 1 m course (F_S 0.0664, 1 repetition). New optional contract
  field `control.warmup_actions` (SAC: `false`), because the robot runtime
  applied the action during warm-up and SAC v5 was trained with it ignored;
  the fidelity test caught this. The evaluation script default and the
  robot-integration tests point to the new folder. `runs/` stays gitignored;
  copy `experiments/test_26_sac/runs/sac_pilot_v5` into it.
- **Next step:** run the common
  evaluation for PPO, SAC and PGRL (`run_all_evaluations.sh`), and fill
  chapter 5 of the report.

---

## Open Decisions / To Revisit

- RL framework choice — resolved for the current baseline: Stable
  Baselines3 PPO (`walking-solutions/ppo/baseline_test24/`). Still open:
  which second algorithm to add for the phase 6 comparison (SAC, TD3,
  A2C, or other).
- Git LFS vs external storage for large assets — resolved: no LFS. Model
  checkpoints and training logs are gitignored entirely (not committed,
  reproducible by re-running training); the NAO URDF/mesh assets are
  committed once as a shared asset under `walking-solutions/shared/`,
  while their upstream source repos are referenced by pinned commit in
  `docs/environment-setup/03-nao-robot-integration.md` rather than
  vendored directly.
- System Specifications table below vs. what was actually verified during
  setup (Ubuntu 24.04, ~24GB RAM, RTX 4060 *Laptop*, per the dated setup
  log) — see `docs/environment-setup/README.md` for the discrepancy;
  not reconciled.
