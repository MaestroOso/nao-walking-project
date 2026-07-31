# NAO Walking — Experiment History

This document reconstructs the development history of the NAO walking RL
project from the ~26 experiment folders produced during development
(`test_01` through `test_24`, plus a parked `test_25` refactor). It exists
to preserve the reasoning behind decisions that aren't visible from the
final code alone — the dead ends, the bugs found, and the lessons learned
along the way — for the academic report and for future reference.

It was reconstructed by reading source code, small config files, and
filenames only (checkpoints, TensorBoard logs, and raw console dumps were
not opened). Some details are inferred and should be spot-checked against
the actual code before being quoted as fact in the report.

**Current status:** `test_24` is the confirmed-working baseline. `test_25`
is a parked, in-progress refactor toward multi-robot training and is not
covered here.

---

## Summary table

| Experiment | Focus | Outcome |
|---|---|---|
| test_01_standing | First standing RL attempt, binary reward | Ran, but reward had no learnable gradient |
| test_02_walking | Add forward reward + NaN-safety wrapper | NaN crashes persisted; mitigated, not solved |
| test_03_walking_home | Empirically-found "home" spawn pose | Training completed multiple times |
| test_04_position | Pure physics forensics (no RL) | Robot still fell; found root causes (see below) |
| test_05_damping_sweep | Systematic PD-gain grid search | **Turning point** — found working gains |
| test_06_rl_walking | Rebuilt RL env on fixed physics, legs-only actions | 2 full training runs completed |
| test_07_reward_tuning | Reward/termination retuning + chunked training fix | Fully completed 500k-step run |
| test_08_walking | Aggressive reward shaping to force walking | Failed (twice) — agent lost balance |
| test_09_diagnostics | Diagnostic-only: is velocity observable? | Confirmed bug — no training run |
| test_10_walking | Fixed missing-velocity bug, simplified reward | Longest run so far (870k steps) |
| test_11_walking | Fixed finger/wrist PhysX instability | Completed (532k steps) |
| test_12_walking | Fixed "fall-fast" reward-hacking exploit | Completed, longer run (2.2M steps) |
| test_13_walking | Multi-robot vectorized training (throughput) | Completed 5M steps — largest so far |
| test_14_walking | Hand-designed reference gait + residual actions + BC pretrain | Pretraining only, no RL run |
| test_14b_walking | Diagnostics: spawn height/orientation/gains bugs | No training; fixed spawn geometry |
| test_15_walking | Reference gait V1 (stepping in place) | Gait design + BC data only |
| test_16_rl_training | Long RL run on gait V1 | Completed 878k steps |
| test_16b_rl | Reference gait V2 (active-stance push) — open loop only | Stability/speed trade-off identified |
| test_16c_rl | Reference gait V3 → V4 — open loop only | V4: stable **and** forward-moving |
| test_17_walking_rl | Fix world-frame vs. body-frame reward bug | Debugging session, short verification runs only |
| test_18_ds_gait | Fix "dive and slide" reward-hacking exploit | Completed 327k-step run |
| test_20_webots | Switch to literal Webots motion-capture gait | Completed run; diagnosed backward-toppling |
| test_21_webots | Tuning pass (GPU, larger batches, bigger residual) | Consolidation step |
| test_22_fullRL | Drop behavior-cloning pretraining | Completed full 5M-step run |
| test_23_backwards | Fix reward sign-convention bug | Completed full 5M-step run |
| test_24 | Add lateral-drift penalty, scale to 50M-step target | **Working baseline**, ~20.9M/50M steps so far |

---

## Phase 1 — Making NAO stand at all (test_01–05)

The earliest attempts (`test_01`, `test_02`) tried standing/walking RL
directly and hit a wall: NAO fell regardless of pose or reward tuning.
`test_03` patched the spawn pose using an empirical sweep, which helped
but didn't address the underlying cause.

`test_04` stepped away from RL entirely to do pure physics forensics, and
found the URDF import was silently broken in several ways:
- Foot collision geometry was empty (the robot had no way to physically
  contact the ground).
- `RHipYawPitch` and all finger joints had zero drive stiffness.
- Torque limits on some joints couldn't resist gravity.
- USD `DriveAPI` gain edits didn't survive `world.reset()` unless applied
  through `controller.set_gains()`.

None of this was visible from the Python-level joint APIs — it required
inspecting the underlying USD scene graph directly.

`test_05` turned that diagnosis into a systematic PD-gain grid search and
found a working configuration. The key insight: once damping was actually
applied and initialization order was fixed, both low gains (Kp=50, Kd=5)
and the URDF's original high gains (Kp=35810, Kd=500) worked — **the fix
was correct damping, not any particular gain magnitude.**

## Phase 2 — Learning to walk via pure RL (test_06–14)

With stable physics, `test_06` rebuilt the RL environment with a
leg-only action space (12 joints) and delta actions from a home pose.
Most of the progress in this phase came from debugging the reward signal
itself rather than tuning it:

- `test_09` found the agent was being rewarded for forward velocity it
  couldn't perceive — the observation vector had no velocity term.
  `test_10` fixed this and simplified the reward from ~6 components down
  to 3, explicitly to reduce gradient noise.
- `test_11` traced PhysX numerical instability to 20 previously
  uncontrolled finger/wrist joints and fixed it with dedicated low-gain
  home poses and disabled collision on those meshes.
- `test_12` found and fixed a reward-hacking exploit: falling forward
  quickly earned more reward than balancing carefully, because the alive
  bonus wasn't scaled by episode length. Fix: `reward = base × (1 +
  step/100)`.
- `test_13` shifted focus to pure throughput, running two robots in
  parallel in one Isaac Sim physics step, reaching 5M steps — the
  largest run to that point.
- `test_14` introduced a new strategy: a hand-designed sinusoidal
  reference gait with RL learning small residual corrections, plus
  behavior-cloning pretraining. This immediately surfaced spawn-pose
  bugs; `test_14b` spent a dedicated session re-deriving the robot's
  spawn height and orientation (an initial overcorrection estimated
  1.76m before landing on the correct ~0.33m).

## Phase 3 — Hand-designed gait iteration (test_15–19)

This phase is the most numerically rigorous part of the project: four
successive versions of a hand-authored reference gait (V1–V4), each
validated open-loop (no RL) for stability and forward progress before
being trusted as a base for training.

- V1 (`test_15`/`test_16`): simple stepping-in-place pattern. Trained for
  878k steps but had no real forward-progress mechanism.
- V2 (`test_16b`): added active-stance push-off. Surfaced a clear
  stability/speed trade-off — a fast version fell over after ~37cm of
  progress in 300 steps, while a stable version crawled at 0.4cm/cycle.
- V3/V4 (`test_16c`): further iteration; V4 finally achieved both
  stability and forward progress (+0.55cm/cycle over 6 cycles, stable).

`test_17` then trained RL on top of V4 but got derailed fixing a
fundamental bug: the reward used world-frame velocity as "forward,"
which only holds if the robot's local axes align with the world's — PPO
had learned to walk in the robot's mechanical backward direction. This
required a body-frame velocity transform and a manual empirical
sign-verification protocol.

`test_18` reverted to the simpler V1 gait and fixed a second exploit:
the agent learned to dive forward and slide along the ground, collecting
~470 units of spurious reward before terminating. Fixed by gating the
forward-velocity reward on the robot staying upright, plus a height-drop
penalty and tighter fall-termination thresholds.

## Phase 4 — Webots motion-capture gait → working baseline (test_20–24)

`test_20` replaced the hand-authored gait with literal playback of real
Webots-captured walking keyframes (`Forwards.motion`). Most effort here
went into diagnosing why the robot still fell — confirmed via
foot-position/tilt telemetry as backward-toppling (mean forward velocity
of −0.25 m/s, tilt spiking to 83°, base drifting 30cm behind the feet at
the worst moment).

`test_21` was a tuning/consolidation pass (moved training to GPU, larger
PPO batches, larger allowed residual). `test_22` made an architectural
bet: drop behavior-cloning pretraining entirely and train PPO from a
random policy, using the reference gait purely as a scaffold for the
action space rather than as an imitation target. This completed a full
5M-step run successfully, validating the bet.

`test_23` found and fixed a one-line reward-sign bug: forward velocity
was being credited with the opposite sign from the robot's actual
heading. A single-character fix (`v[0]` → `-v[0]`) with an outsized
effect on what the agent was actually learning to do.

`test_24` added a lateral-drift penalty (to stop sideways wandering) and
scaled the training target 10× (5M → 50M steps). It is the current,
accepted working baseline, at approximately 20.9M of 50M target steps —
training appears to have been paused rather than stopped due to failure
(clean chunk logs throughout).

### `test_24` configuration, for the record

- **Action space:** 12-dim residual added to the literal Webots reference
  trajectory for that instant (arms/head held at fixed targets), scaled
  by 0.25 rad, clamped to joint limits.
- **Observation space (23-dim):** projected gravity vector, damped
  angular velocity, linear velocity (x/y only), 12 normalized leg joint
  positions, base height, gait-phase sin/cos.
- **Reward:** `5.0·(forward vel) + slow-ramping alive bonus −
  0.5·(orientation error) − 0.5·(lateral velocity)²` — notably, **no
  joint-tracking/imitation term**; the reference gait shapes behavior
  only through the action space, not the reward.
- **Physics:** `physics_dt=1/120s`, `control_dt=1/30s`, spawn height
  0.3344m (empirically measured), three-tier PD gains with hips/knees
  ~8× stiffer than other joints (load-bearing).
- **Algorithm:** SB3 PPO, MLP [128,128], lr=3e-4, n_steps=4096,
  batch_size=256, γ=0.99, GAE-λ=0.95, clip=0.2, GPU-trained.
- **Training infrastructure:** chunked into 50,000-step subprocess runs
  with checkpoint-based resumption, RAM-availability gating — apparently
  working around Isaac Sim's long-run memory/stability behavior.

**Known issue worth noting:** `test_24`'s auto-generated `config.txt` is
stale — it still describes a much earlier hyperparameter scheme from
`test_16` and does not reflect the actual configuration above. The real
values are only in the code. This is itself a small lesson about keeping
auto-logged run metadata in sync for reproducibility.

---

## Cross-cutting lessons learned

1. **URDF import bugs are invisible at the Python API level.** Broken
   collision geometry and zero-stiffness joints required USD-prim-level
   inspection to find; nothing about them was visible from
   `Articulation.get_joint_positions()` or similar high-level calls.
2. **Reward "design" failures were often reward *observability* bugs.**
   Multiple episodes of "the reward isn't working" turned out to be "the
   agent can't see the thing the reward depends on" (missing velocity in
   the observation vector, in this case).
3. **Reward hacking recurred in different forms and needed explicit
   fixes each time:** fast-forward-falling (test_12), diving-and-sliding
   (test_18), and implicitly, sideways "trotting" around a lenient
   heading reward (test_17) — each required a specific gating or penalty
   term, not just a weight adjustment.
4. **Coordinate-frame and sign-convention bugs were cheap to introduce
   and expensive to catch.** Two separate incidents (test_17, test_23) of
   the robot being rewarded for the wrong direction of travel, each only
   found through dedicated verification runs.
5. **Literal motion-capture replay outperformed hand-authored gaits**,
   despite requiring more diagnostic work up front. The version that
   ultimately worked was the *simplest* transformation of the raw Webots
   data — no windowing, smoothing, or symmetrizing — once the spawn pose
   was aligned to match it exactly (verified to <0.5° discontinuity).
6. **Simulator infrastructure constraints shaped the whole project's
   methodology**, independent of the RL algorithm. Nearly every
   experiment from `test_07` onward used chunked, resumable training
   with subprocess restarts between chunks — apparently required to work
   around Isaac Sim's long-run memory behavior.

---

*This document was reconstructed by an AI assistant (Claude Code) reading
project source code and filenames, cross-checked against `notes.md` and
`TRAINING_GUIDE.md`. It has not been independently verified against the
raw training logs/TensorBoard data and should be treated as a starting
point for the report, not a final source of truth.*
