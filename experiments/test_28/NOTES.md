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
  - Gains are in rad/rad, range [−1, 1].
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
