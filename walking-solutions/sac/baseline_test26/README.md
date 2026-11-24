# SAC Baseline (from test_26_sac)

The project's SAC result: Soft Actor-Critic (Stable-Baselines3) learning
residual corrections on top of the looped Webots gait. Promoted from
`experiments/test_26_sac`, run `sac_pilot_v5`. That folder's `APPROACH.md`
records the whole path from v1 to v5: every change, why it was made, and
what each run showed.

**Status:** run `sac_pilot_v5` finished: 12 h, 1,230,000 steps. It is the
first SAC policy that does not fall in headless simulation, and it
completes the 1 m evaluation course. It walks very slowly, turns, and is
fragile (see Results and Known issues).

**Self-contained:** everything needed to train, evaluate and deploy is in
this folder, plus `../../shared/`. Nothing is imported from `experiments/`
or from the other solutions. The code is the `experiments/test_26_sac`
code at commit `f29cb8a` (the commit that trained `sac_pilot_v5`). Only
these were changed:
- asset paths, now `../../shared/...`;
- the looped gait files, now read from
  `../../shared/reference_gaits/forwards_looped_*` (byte-identical to the
  `experiments/test_27_gait_loop/data/looped_*` files used in training);
- comments, defaults (`evaluate_policy.py --model`) and the
  `run_chunked_training.sh` header and run-name argument.

The headless evaluation from this folder reproduces the original run
exactly: 1500 steps, reward 4037.6.

## Configuration

**Action** (12-dim, one per leg joint, at 30 Hz):

```
q*(t) = clip( q_ref(t) + 0.25 · clip(a, −1, 1) , joint limits )
```

- `q_ref(t)` is the looped Webots `Forwards.motion` gait, period 2.64 s.
- **Warm-up:** for the first 60 steps (2 s) the legs hold the home pose and
  the action is **ignored** (`warmup_actions=False`). The policy still
  observes and acts; its output has no effect.
- `RHipYawPitch` is a URDF mimic of `LHipYawPitch` (one motor on the real
  NAO), so its action has no effect. 11 of the 12 outputs do something.

**Observation** (23-dim): projected gravity (3), body angular velocity ×0.25
(3), forward/lateral velocity in the yaw-only heading frame (2), 12 leg
joint positions normalised to [−1, 1] by the joint limits, base height (1),
gait phase sin/cos (2; (0, 1) during warm-up). Quaternions are read in
Isaac Sim's correct (w, x, y, z) order. PPO `baseline_test24` was trained
with them read as (x, y, z, w); the two observation vectors are not
interchangeable.

**Reward** per step (`NaoWalkingEnv._compute_reward` + `step`):

```
r = 5.0 · v_fwd                    forward speed in the heading frame [m/s]
  + 1.0 · (1 + step/500)           alive bonus, grows from 1 to 4 per step
  − 0.5 · (g_x² + g_y²)            tilt (projected gravity)
  − 0.5 · v_lat²                   sideways speed
  − 0.5 · ω_yaw²                   yaw rate
  − 100   on the falling step only (terminated, not the time limit)
```

**Episode end:** a fall (base height < 0.10 m, rotated more than ~145° from
spawn, or NaN), checked only after step 50; or the 1500-step (50 s) limit.

**Algorithm:** SB3 2.7.1 SAC defaults, except `net_arch=[128, 128]` and
γ = 0.99 (matched to PPO `baseline_test24`): lr 3e-4, buffer 1M, batch 256,
τ 0.005, `train_freq=1`, `gradient_steps=1`, `learning_starts=100`,
automatic entropy coefficient (target −12), GPU.

**Physics:** same as PPO `baseline_test24`: `physics_dt` 1/120 s, control
1/30 s, spawn height 0.3344 m, hips and knees about 8× stiffer than the
other joints.

**Chunked training:** 15,000 steps per chunk, each in a fresh Isaac Sim
process. `final_model.zip` and `replay_buffer.pkl` (~244 MB) are saved at the
end of each chunk, plus `ckpt_<N>_steps.zip`. A 30-minute limit per chunk
kills and retries a hung Isaac Sim. A batch stops at its wall-clock cap
(12 h); re-running the script continues the same run.

## Running

```bash
source <repo>/nao_walking_project/isaac_env/bin/activate   # from the repo's environment setup
cd walking-solutions/sac/baseline_test26

# Numbers to report (headless, deterministic):
python evaluate_policy.py --headless --episodes 1

# Watch it in the GUI (real time; deterministic episodes are identical):
python evaluate_policy.py --episodes 3
python evaluate_policy.py --episodes 1 --slow      # slowed down

# Sanity checks of the env (quaternion convention, looped gait):
python verify_quat_fix.py
python verify_gait_loop.py

# New training run with the v5 settings, chunked and resumable (12 h batch):
./run_chunked_training.sh sac_v5_repro

# Continue the promoted run for another batch:
./run_chunked_training.sh sac_pilot_v5
```

- **Per-run outputs** go in `runs/<run>/`: `final_model.zip`,
  `replay_buffer.pkl`, `ckpt_<N>_steps.zip` every 15k steps,
  `monitor.monitor.csv` (one row per training episode: the learning curve),
  `progress.txt`, `config.txt`, `training_log.txt` and `tb/` (TensorBoard).
- **`evaluate_policy.py` output:** "Survived" counts episodes longer than the
  50-step grace period, not episodes without a fall; read the `Term` column.
- **Common evaluation for the report** (δ ∈ {0, 1, 2, 4}°, 10 repetitions,
  Table 5-3): `experiments/test_evaluation_script/evaluate_algorithms.py
  --algo sac`. Its default is this folder's `runs/sac_pilot_v5`.
- **Robot:** `deployment.yaml` is validated by
  `robot-integration/tools/validate_deployment.py`. It sets
  `control.warmup_actions: false` so the robot runtime ignores the action
  during warm-up, as in training. Export the policy with
  `python robot-integration/export/export_policy.py walking-solutions/sac/baseline_test26`
  (verified: NumPy actor matches SB3 within 1.4e-5).

**Timing:** about 10 minutes per 15k-step chunk on the RTX 4060 Laptop; a
12 h batch is about 1.1–1.2M steps.

## Getting the trained run (`sac_pilot_v5`)

`runs/` is gitignored, as for PPO and PGRL. The run is not in the repo. Copy
it from the original experiment (338 MB, most of it the replay buffer):

```bash
mkdir -p runs
cp -r ../../../experiments/test_26_sac/runs/sac_pilot_v5 runs/
```

Only `final_model.zip` is needed to evaluate or export. `replay_buffer.pkl`
is only needed to continue training the run. If the run is lost, retrain
it with `./run_chunked_training.sh <name>` (12 h).

## Results (`sac_pilot_v5`)

Learning curve (training episodes, with exploration noise):

| Steps | Mean ep length | Eps reaching 1500 | Mean reward |
|---|---|---|---|
| 0–100k | 94 | 0 | −42 |
| 200–300k | 131 | 0 | 93 |
| 300–400k | 281 | 0 (max 1262) | 282 |
| 400–500k | 839 | 26 / 118 | 1544 |
| 700–800k | 1239 | 51 / 80 | 2896 |
| 1.0–1.1M | 1305 | 49 / 77 | 3099 |
| 1.1–1.2M | 1218 | 42 / 69 | 2845 |

Evaluation of `final_model.zip`, headless, 48 s of gait after the warm-up:

| Policy | Outcome | Forward | Lateral | Speed | Heading change | \|roll\| mean / max |
|---|---|---|---|---|---|---|
| Deterministic | 1500 steps | +3.00 m | −0.74 m | 0.064 m/s | −34° | 4.4° / 13.2° |
| Stochastic ep 1 | 1500 steps | +2.95 m | −0.70 m | 0.063 m/s | +34° | 4.5° / 14.3° |
| Stochastic ep 2 | fell at 39 s | +0.79 m | −2.13 m | 0.058 m/s | +148° | 4.9° / 15.0° |
| Stochastic ep 3 | 1500 steps | +2.96 m | −1.09 m | 0.066 m/s | +39° | 4.7° / 14.9° |

For comparison, the reference gait alone (zero action) falls after 2.0 s
of gait, at 0.42 m.

Common evaluation script, smoke test (D = 1 m, δ = 0°, 1 repetition):
**success** in 15.07 s, F_S = 0.0664. The full protocol (δ ∈ {0, 1, 2, 4}°,
10 repetitions) has not been run yet.

- **Balance.** The fall penalty (v5) is what made it stay up; v4 without it
  learned to lunge forward and fall after ~1.1 s.
- **Shuffling.** The alive bonus grows to 4 per step, so a full episode
  earns ≈ 3,750 for survival against ≈ 450 for 3 m of travel. Very small
  steps are the safe optimum.
- **Turning.** Forward speed is measured in the heading frame, so walking
  in a turned direction earns full forward reward; the yaw-rate penalty
  barely charges a slow turn.
- **Lopsided gait.** Constant offsets on the left leg (hip roll +12.7°,
  ankle roll −7.5°); about 35% of the action components are at ±1.

## Known issues

- **GUI ≠ headless, and fragile.** In the GUI the same model falls after
  5.5 s of gait (225 steps, 0.64 m); headless it lasts the full 48 s. The
  only difference is rendering (`world.step(render=True)`), so the balance
  is sensitive to tiny physics differences. Report the headless numbers;
  the δ-perturbation protocol measures the fragility properly.
- **Slow and turning:** see Results. `experiments/test_26_sac` has a v6
  attempt (constant alive bonus, yaw-rate penalty 5.0), not finished.
- **Harmless startup errors.** `isaacsim.asset.browser … No module named
  'requests'` and the finger "mimic joint" limit errors can be ignored.
