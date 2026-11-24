# PGRL Baseline (from test_28)

The project's PGRL result: policy-gradient RL by finite differences (Kohl &
Stone [20], as used by Kulk & Welsh [14]; report §4.5.3). It tunes 12
parameters of a hand-built controller on top of the looped Webots gait.
Promoted from `experiments/test_28`. See that folder's `NOTES.md` for how it
was built, and `experiments/test_27_gait_loop/NOTES.md` for the looped gait.

**Status:** run `pgrl_v1` finished: 166 iterations, 1992 evaluations. The
budget was 2000, which is not a multiple of 12. It improves the score but
still falls after two steps (see Results).

**Self-contained:** everything needed to train, evaluate and deploy is in
this folder, plus `../../shared/`. Nothing is imported from `experiments/`
or from `ppo/`. Three files are copies of code that lives elsewhere. They do
**not** follow changes to their originals:

| File here | Copy of | Why it must match |
|---|---|---|
| `nao_walking_env.py`, `webots_reference_gait.py` | `ppo/baseline_test24/` (verbatim) | Same physics, PD tiers, warm-up, fall criteria and 50 s limit as PPO |
| `pgrl_episode.py` (`run_episode`, `make_evaluation_env_class`) | `experiments/test_evaluation_script/evaluate_algorithms.py` | The F_S used for training equals the evaluation score (Table 5-3) |
| `pgrl_metrics.py` | `experiments/test_evaluation_script/metrics.py` (verbatim) | Same F_S formula (Eq. 4-5) |

The copies were verified identical when this folder was created
(2026-11-23). The functions were compared ignoring docstrings and comments.

## Configuration

**Policy:** a 12-parameter vector θ, not a neural network. PGRL searches in
normalised space θ ∈ [0, 1]^12. `pgrl_params.py` maps θ to physical values.

| # | Parameter | Meaning | Range (`pgrl_v1`) | Range (new runs) | Initial |
|---|---|---|---|---|---|
| 0–3 | `K_ank_pitch`, `K_hip_pitch`, `K_ank_roll`, `K_hip_roll` | Torso-tilt compensator gains [rad/rad] | [−1, 1] | [−3, 3] | 0 |
| 4 | `f` | Gait frequency factor (playback speed) | [0.7, 1.3] | same | 1 |
| 5 | `a` | Gait amplitude factor (around the home pose) | [0.7, 1.3] | same | 1 |
| 6–11 | `s_hip_yawpitch` … `s_ankle_roll` | Leg stiffness factor per joint type (scales PD kp and kd, both legs) | [0.2, 1.2] | same | 1 |

The gain range was widened after `pgrl_v1` because three of its gains ended
at ±1. Normalised θ only has a meaning together with its ranges, so each run
stores them in `runs/<run>/config.json`. Every script reads them back with
`load_run_specs()`, so `pgrl_v1` is always interpreted with [−1, 1].

**Controller** (`pgrl_env.PGRLWalkingEnv.compute_leg_targets`), every 1/30 s
control step after the warm-up:

```
q*(t) = clip( q_home + a·(q_ref(f·t) − q_home) + Δq_comp , joint limits )

Δq_comp, same on BOTH legs:
  ankle pitch += K_ank_pitch·g_x     hip pitch += K_hip_pitch·g_x
  ankle roll  += K_ank_roll ·g_y     hip roll  += K_hip_roll ·g_y
```

- **Inputs.** `g_x` and `g_y` are the horizontal components of gravity
  projected into the torso frame, from the IMU orientation. They are
  computed with the **correct (w, x, y, z)** quaternion convention. `g_x > 0`
  when leaning forward, and ≈ +0.028 standing at the home pose.
- **What the compensator replaces.** The report's original design used the
  ZMP error; the torso-tilt compensator replaces it.
- **Warm-up.** For the first 60 steps the legs hold the home pose and the
  compensator is off, exactly as for PPO.
- **Inputs the controller ignores.** The env's `step(action)` ignores
  `action` (it keeps the Gymnasium signature). The 23-dim PPO observation is
  still computed, but not used.

**Base gait:** `../../shared/reference_gaits/forwards_looped_*`.
- It is the Webots `Forwards.motion` clip (2.60 s) tiled 20 times with a
  period of 2.64 s, giving 1320 frames and 52.76 s.
- Gait time wraps at 52.80 s, so any `f` is safe.
- The seam between copies jumps 3.3° in hip and ankle roll. Every other
  joint jumps less than 0.3°.
- Rebuild it with `build_looped_gait.py`. It is byte-identical to the
  test_27 data.

**Episode and score** (`pgrl_episode.py`):
- **Course.** 1 m (D) along the robot's forward direction (world −x),
  measured from the end of the warm-up. δ = 0° during training, so every
  episode is deterministic.
- **Score F_S (Eq. 4-5):**
  - D/t if the course is completed;
  - otherwise (d/t)·0.5·|d|/D. The distance covered while falling counts.
- **Episode end:** at success, at a fall (the baseline height and tilt
  criteria), or at the 50 s limit.

**Update rule** (`train_pgrl.py`, one iteration):
1. Make P = 12 perturbed policies θ + Δ, with each Δ_i drawn from
   {−ε, 0, +ε} and ε = 0.05.
2. Run one episode each and record F_S.
3. For each parameter, average F_S over the policies where it went up (F⁺),
   stayed the same (F⁰) and went down (F⁻). Then A_i = 0 if F⁰ is the best
   of the three, otherwise A_i = tanh(F⁺ − F⁻) (Eq. 4-6).
4. Update θ ← clip(θ + η·A/‖A‖, 0, 1) with η = 0.05. The step is normalised
   to length η as in Kohl & Stone's original algorithm. Without this, F_S
   differences of about 0.005 give steps of about 1e-4, i.e. no learning.

**Budget:** 2000 evaluations (episodes). Training runs in chunks of 10
iterations per process because Isaac Sim's memory grows over time. State is
saved after every iteration, so a crash loses at most one iteration.

## Running

```bash
source <repo>/nao_walking_project/isaac_env/bin/activate   # from the repo's environment setup
cd walking-solutions/pgrl/baseline_test28

# Watch the trained policy in the GUI (real time; 3 identical episodes):
python evaluate_policy.py --run-name pgrl_v1 --episodes 3 --hold-open

# Numbers to report (headless):
python evaluate_policy.py --run-name pgrl_v1 --headless --episodes 5
python evaluate_policy.py --initial --headless      # initial parameters, for comparison

# Sanity checks of the controller (initial params, tilt sign, gain effect):
python check_controller.py

# New training run (gains [-3, 3]), chunked and resumable:
./run_chunked_training.sh pgrl_v2

# Reproduce pgrl_v1 exactly (gains [-1, 1], seed 0):
./run_chunked_training.sh pgrl_v1_repro --specs v1
```

- **Per-run outputs** go in `runs/<run>/`:
  - `state.json`: θ, iteration, evaluation count, RNG state;
  - `config.json`: hyperparameters and parameter ranges;
  - `evaluations.csv`: one row per episode;
  - `iterations.csv`: one row per iteration, the source for Figure 5-2 and
    Table 5-2;
  - `training_log.txt`.
- **Common evaluation for the report** (δ ∈ {0, 1, 2, 4}°, 10 repetitions,
  Table 5-3): `experiments/test_evaluation_script/evaluate_algorithms.py
  --algo pgrl`. Its default is this folder's `runs/pgrl_v1`.
- **Robot:** `deployment.yaml` is validated by
  `robot-integration/tools/validate_deployment.py`. Export the policy with
  `python3 robot-integration/export/export_policy.py walking-solutions/pgrl/baseline_test28`.

**Timing:** about 4.5 s per episode while it falls, so 12 episodes take
about 55 s per iteration and the full 2000-evaluation budget takes about
2.5 h.

## Getting the trained run (`pgrl_v1`)

`runs/` is gitignored, as for PPO. The run is not in the repo. Copy it from
the original experiment:

```bash
mkdir -p runs
cp -r ../../../experiments/test_28/runs/pgrl_v1 runs/
```

The run is small: `state.json` plus `config.json` define the policy, and the
CSVs and log are the training record. If it is lost, retrain it with
`--specs v1` (see above). The first iteration was checked to reproduce
bit-for-bit from this folder.

Final θ of `pgrl_v1` (physical values), for reference:

| K_ank_pitch | K_hip_pitch | K_ank_roll | K_hip_roll | f | a |
|---|---|---|---|---|---|
| +1.000 | +0.219 | −1.000 | −1.000 | 1.272 | 1.093 |

| s_hip_yawpitch | s_hip_roll | s_hip_pitch | s_knee_pitch | s_ankle_pitch | s_ankle_roll |
|---|---|---|---|---|---|
| 0.784 | 1.176 | 1.053 | 0.926 | 1.137 | 1.173 |

## Results (`pgrl_v1`)

| Iteration | 1 | 20 | 60 | 100 | 120 | 166 |
|---|---|---|---|---|---|---|
| Mean F_S | 0.0428 | 0.0525 | 0.0641 | 0.0757 | 0.0784 | 0.0782 |

Evaluation of the final θ, headless, δ = 0° (verified from this folder):

| Policy | Outcome | d | t | Lateral | F_S |
|---|---|---|---|---|---|
| Initial parameters | fall | 0.409 m | 2.03 s | — | 0.0412 |
| `pgrl_v1` final θ | fall | 0.531 m | 1.73 s | +0.020 m | 0.0813 |

- **Score.** Mean F_S rose 83% and plateaued from about iteration 120. No
  episode completed the 1 m course; all 1992 fell.
- **What it does.** Two steps, then a fall forward after the second
  (left-foot) step. The initial parameters fall during the first step.
- **Fall exploit.** It covers more distance in less time than the initial
  parameters (0.52 m / 1.75 s against 0.41 m / 2.03 s). F_S counts the
  distance covered while falling, so PGRL learned partly to **fall forward
  further and sooner**. Three gains at the bound and f close to 1.3 fit that
  picture.

## Known issues

- **GUI ≠ headless.** The same θ gives d = 0.499 m, lateral −0.167 m,
  F_S = 0.0717 in the GUI, and d = 0.531 m, lateral +0.020 m, F_S = 0.0813
  headless. Each mode is deterministic on its own.
  - The likely cause is `world.step(render=True)` going through the Kit app
    update.
  - Training and evaluation are headless, so report the headless numbers.
    The GUI is for viewing only.
- **Baseline quaternion bug.** The copied env's `_quat_rotate_inverse` reads
  quaternions as (x, y, z, w), the bug found in `experiments/test_26_sac`.
  PGRL does not use it: the tilt is computed correctly in `pgrl_env.py`.
  Fall detection (`|quat[3]| < 0.3`) happens to be correct.
- **Harmless startup errors.** `isaacsim.asset.browser … No module named
  'requests'` and the finger "mimic joint" limit errors can be ignored.
