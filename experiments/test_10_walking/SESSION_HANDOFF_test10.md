# SESSION HANDOFF — NAO Walking RL Project
## Date: March 25, 2026
## Experiment: test_10 (velocity observation + simplified reward)

---

## PROJECT OVERVIEW

Training a NAO V40 humanoid robot to walk using RL in NVIDIA Isaac Sim 4.2.0.
PPO (Stable-Baselines3) with custom Gymnasium environment.

**System:** Ubuntu 22.04, AMD Ryzen 7, RTX 4060, 23GB usable RAM, Isaac Sim 4.2.0

---

## WHAT CHANGED IN test_10

### 1. Linear velocity added to observations (43 → 45 values)
- `get_linear_velocity()` confirmed working in test_09 diagnostics
- Added vel_x (forward) and vel_y (lateral) to observation space
- Agent can now DIRECTLY OBSERVE the quantity it's rewarded for
- This was the critical missing piece: previous tests rewarded forward velocity
  but the agent had no way to perceive it

### 2. Simplified reward (6 components → 3)
- KEPT: alive_bonus (+1.0), forward_vel (+5.0 × vx), orientation (-0.5 × tilt)
- REMOVED: energy, smoothness, joint_limit, vel_target
- Rationale: fewer components = cleaner gradient signal
- Natural consequences (falling) already punish bad behavior

### 3. Training from scratch
- New observation space (45 vs 43) means no model resume
- Clean start avoids value function miscalibration issues from test_08

---

## PROJECT HISTORY (condensed)

- **test_01-03**: Setup, home pose, NaN safety
- **test_04**: 5 physics fixes (collision, damping, joints, FSR, ground)
- **test_05**: PD sweep, knee fix → balance must be learned
- **test_06**: First RL pipeline → falls immediately
- **test_07**: Reward tuning → **perfect standing** (but zero motion)
- **test_08a**: Aggressive walking reward → **failed** (lost balance)
- **test_08b**: Gentler curriculum → **failed** (still falls at grace+1)
- **test_09**: Diagnostics → confirmed get_linear_velocity() works
- **test_10**: Velocity obs + simple reward → **ready to train**

### Why previous walking attempts failed
1. Agent couldn't observe its own forward velocity (rewarded for invisible quantity)
2. Reward had too many components creating noisy gradients
3. Value function miscalibration when resuming from standing model

---

## HOW TO RUN

```bash
cd ~/nao_walking_project/experiments/test_10_walking/
chmod +x run_chunked_training.sh
./run_chunked_training.sh
```

### Monitor
```bash
tail -f runs/test10_walking/training_log.txt
tensorboard --logdir runs/
```

### Evaluate
```bash
python evaluate_policy.py --model runs/test10_walking/final_model.zip --headless
python evaluate_policy.py --model runs/test10_walking/model_250000.zip --headless --episodes 5
```

---

## OBSERVATION SPACE (45 values — NEW)

| Index | Content | Notes |
|-------|---------|-------|
| [0:3] | Projected gravity (body frame) | Balance signal |
| [3:6] | Angular velocity (body frame) | Scaled ×0.25 |
| [6:8] | **Linear velocity (vel_x, vel_y)** | **NEW — world frame, m/s** |
| [8:20] | Leg joint positions (12) | Normalized [-1, 1] |
| [20:32] | Leg joint velocities (12) | Scaled ×0.1 |
| [32:44] | Previous actions (12) | For temporal coherence |
| [44] | Base height | Meters |

## REWARD (simplified)

| Component | Weight | Formula |
|-----------|--------|---------|
| alive_bonus | +1.0 | constant per step |
| forward_vel | +5.0 | × vel_x (m/s) from API |
| orientation | -0.5 | × (grav_x² + grav_y²) |

Standing still: ~1.0/step. Walking at 0.1 m/s: ~1.5/step. Walking at 0.2 m/s: ~2.0/step.

## CONSTRAINTS

- Action: [-1, 1] × 0.4 rad from home pose, clamped to joint limits
- Termination: height < 0.10m, |qw| < 0.3, grace = 50 steps, max = 1500 steps
- PD: Kp=35810, Kd=500, max_effort=100 Nm
- Physics: 120 Hz, control: 30 Hz (decimation=4)

---

## FILES

1. `nao_walking_env.py` — Environment with velocity obs + simple reward
2. `train_ppo.py` — Training with file logging support
3. `run_chunked_training.sh` — Chunked training with 30s rest, log file
4. `evaluate_policy.py` — Evaluation with walking metrics
5. `nao.urdf` — NAO robot model
6. This handoff document
