# test_28 — PGRL with a torso-tilt compensator

**Focus:** third algorithm for the comparison (report §4.5.3). The method is
policy-gradient RL by finite differences (Kohl & Stone [20], Kulk & Welsh
[14]), applied to 12 parameters of a controller built on the looped Webots
gait from test_27.

## Design decisions (2026-11-17)
- **Base gait:** test_27's looped Webots clip, the same reference gait as PPO
  and SAC. Gait time wraps, so any frequency factor works.
- **Compensator:** driven by **torso tilt**: horizontal components g_x, g_y of
  projected gravity, measured by the IMU. This replaces the ZMP error.
  - Why: no foot contact sensors are needed, the value already exists, and
    the real robot has the IMU.
  - Gains are in rad/rad. Range [−1, 1] for `pgrl_v1`, widened to
    [−3, 3] afterwards (see "Gain range change" below).
  - The same correction is applied to both legs.
  - Report §3.2.3, Table 3-1, §4.5.3 and Table 4-6 must be reworded
    (they still describe a ZMP compensator).
- **F_S as in the report:** distance covered while falling counts.
- **δ = 0° during training,** so episodes are fully deterministic.
- **Environment:** subclasses `walking-solutions/ppo/baseline_test24`'s
  `NaoWalkingEnv` read-only. Physics, warm-up, grace period, fall criteria
  and the 50 s limit are identical to PPO/SAC.
- **Measurement:** the episode and F_S are measured by
  `test_evaluation_script/evaluate_algorithms.run_episode`, imported
  read-only, so the training score is exactly the evaluation score.
- **Tilt quaternion:** computed with the correct (w, x, y, z) convention
  (see `test_26_sac/APPROACH.md`). The baseline's `_quat_rotate_inverse` is
  not used for it.

## Files
| File | Role |
|---|---|
| `pgrl_params.py` | 12 parameters, ranges, initial values, normalisation |
| `pgrl_env.py` | `PGRLWalkingEnv`: looped gait, f/a scaling, tilt compensator, stiffness |
| `pgrl_episode.py` | one scored episode (reuses the evaluation script) |
| `check_controller.py` | sanity checks before training |
| `train_pgrl.py` | PGRL loop, resumable, logs for Figure 5-2 / Table 5-2 |
| `run_chunked_training.sh` | runs `train_pgrl.py` in 10-iteration chunks until the budget is reached |
| `evaluate_policy.py` | watch or evaluate a trained θ (GUI in real time by default), or the initial θ with `--initial` |

## Sanity checks (`check_controller.py`, headless)
| Run | Outcome | d | t | F_S |
|---|---|---|---|---|
| Initial params (= test_27) | fall | 0.409 m | 2.03 s | 0.0412 |
| K_ank_pitch = +0.5 | fall | 0.489 m | 2.57 s | 0.0466 |
| K_ank_pitch = −0.5 | fall | 0.342 m | 1.67 s | 0.0351 |

- **Initial parameters:** reproduce test_27.
- **Tilt sign:** g_x grows positive as the torso pitches forward, as
  expected. Standing at the home pose, g_x ≈ +0.028 (≈ 1.6° forward lean),
  so the compensator also applies a small constant correction.
- **Ankle compensator:** a positive gain is corrective (the robot survives
  longer and goes further); a negative gain makes it fall sooner.
- **Roll:** the robot also rolls sideways before falling (|g_y| up to 0.34).
  With the correct quaternion, roll is now visible.

## Smoke test (1 iteration, `runs/smoke_test`)
- **Time:** 12 episodes in 54 s, so about 4.5 s per episode while it still
  falls. The full budget of 2000 evaluations (about 166 iterations) is
  about 2.5 h, and more if episodes get longer.
- **All 12 fell:** F_S between 0.037 and 0.046.
- **The update is tiny:** max |A| = 0.0002 in normalised units, against
  ε = 0.05. With F_S differences of about 0.005 m/s, η·tanh(ΔF) ≈ η·ΔF, so
  Eq. 4-6 as written barely moves θ.
- **Decision (option b):** take the direction from Eq. 4-6 (A_i = 0 or
  tanh(F⁺ − F⁻)), then normalise the step to length η as in Kohl & Stone's
  original algorithm [20]: θ ← θ + η·A/‖A‖. Report §4.5.3 needs one sentence
  on this.

## Full run `pgrl_v1`
- **Command:** `./run_chunked_training.sh pgrl_v1`
- **Settings:** 2000 evaluations, 12 policies per iteration, ε = η = 0.05,
  D = 1 m, δ = 0°, seed 0.
- **Chunks:** 10 iterations per process.
- **Log:** `runs/pgrl_v1/training_log.txt`.

```bash
source ../../nao_walking_project/isaac_env/bin/activate
python check_controller.py [--gui]
python train_pgrl.py --run-name pgrl_v1 --iterations 10   # re-run to continue
```

## Result `pgrl_v1` (finished 2026-11-17, 166 iterations, 1992 evaluations)
Training stopped at 1992 evaluations because 2000 is not a multiple of 12.

| Iteration | 1 | 20 | 60 | 100 | 120 | 166 |
|---|---|---|---|---|---|---|
| Mean F_S | 0.0428 | 0.0525 | 0.0641 | 0.0757 | 0.0784 | 0.0782 |

- **F_S:** the mean rose 83% (0.043 → 0.078) and plateaued from about
  iteration 120. The best single episode scored 0.0874 (eval 1662).
- **No episode completed the 1 m course.** All 1992 fell.
- **Fall exploit:** in the last iteration the mean distance was 0.52 m and
  the mean time 1.75 s, against 0.41 m / 2.03 s at the start. PGRL learned to
  fall forward **further and sooner**, which F_S rewards (option a: distance
  covered while falling counts). This is the reward exploit from report §4.1
  reappearing, now in the evaluation metric.
- **Final θ (physical values):**
  - Compensator gains: K_ank_pitch = +1.000 (upper bound),
    K_hip_pitch = +0.219, K_ank_roll = −1.000 (lower bound),
    K_hip_roll = −1.000 (lower bound).
  - Gait: f = 1.272 (close to the 1.3 limit), a = 1.093.
  - Stiffness: hip yaw-pitch 0.784, hip roll 1.176, hip pitch 1.053,
    knee 0.926, ankle pitch 1.137, ankle roll 1.173.
  - Interpretation: three compensator gains hit their bounds, and the gait
    plays 27% faster with larger steps. That fits "throw the body forward
    faster" more than balance.

### Visual check of the final θ (`evaluate_policy.py`, GUI)
| Outcome | d | t | Lateral | Speed | F_S |
|---|---|---|---|---|---|
| fall | 0.499 m | 1.73 s | −0.167 m | 0.288 m/s | 0.0717 |

- **Repeatability:** 3 episodes, all identical (δ = 0°, deterministic).
- **What it does:** the robot takes **two steps**, then falls after the
  second step (left foot), forward and to the right. The untrained
  parameters fall during the first step.
- **Main limit:** the fall goes to the right (−0.167 m) and both roll gains
  sit at their bound, so side-to-side balance looks like the main limit.
- **Policy vs. curve:** the final θ scores 0.072, slightly below the
  0.078 mean of the last iteration. The training curve averages the 12
  perturbed policies, not θ itself.

## Gain range change (after `pgrl_v1`, 2026-11-17)
- **Why:** three of the four compensator gains finished at the [−1, 1]
  bound, and F_S plateaued from about iteration 120. The search wanted
  stronger corrections, and the clipping discarded those steps.
- **Change:** all four gains now use **[−3, 3] rad/rad** (`PARAM_SPECS` in
  `pgrl_params.py`). Report Table 4-6 already lists [−3, 3] for the
  compensator gains.
- **Side effect:** with the normalised ε = η = 0.05, one perturbation or
  step of a gain is now 0.3 rad/rad instead of 0.1.
- **Old runs stay correct.** Normalised θ only has a meaning together with
  its ranges. Each run stores its ranges in `runs/<run>/config.json`, and
  `train_pgrl.py` (when resuming) and `evaluate_policy.py` read them with
  `load_run_specs()`. `pgrl_v1` is therefore still interpreted with
  [−1, 1] (`PARAM_SPECS_V1` is the fallback). New runs use [−3, 3].
- **Not run yet.**
