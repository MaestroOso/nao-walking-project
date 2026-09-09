# PPO Baseline (from test_24)

The project's current working forward-walking result. Promoted from
`experiments/test_24` — see that folder's `NOTES.md` for how it got here
and `docs/experiment-history.md` for the full project narrative.

**Status:** ~20.9M of a 50M-step target completed. Training appears to
have been paused rather than stopped due to failure (clean chunk logs
throughout) — resumable via `run_chunked_training.sh` (see below).

## Configuration

- **Action space:** 12-dim residual added to the literal Webots reference
  trajectory for that instant (arms/head held at fixed targets), scaled
  by 0.25 rad, clamped to joint limits.
- **Observation space (23-dim):** projected gravity vector, damped
  angular velocity, linear velocity (x/y only), 12 normalized leg joint
  positions, base height, gait-phase sin/cos.
- **Reward:** `5.0·(forward vel) + slow-ramping alive bonus −
  0.5·(orientation error) − 0.5·(lateral velocity)²` — no joint-tracking/
  imitation term; the reference gait shapes behavior only through the
  action space, not the reward.
- **Physics:** `physics_dt=1/120s`, `control_dt=1/30s`, spawn height
  0.3344m, three-tier PD gains with hips/knees ~8× stiffer than other
  joints (load-bearing).
- **Algorithm:** SB3 PPO, MLP [128,128], lr=3e-4, n_steps=4096,
  batch_size=256, γ=0.99, GAE-λ=0.95, clip=0.2, GPU-trained.
- **BC pretraining:** not used — `pretrain.py` is carried over from
  earlier experiments for reference only; this architecture (since
  `test_22`) trains PPO from scratch against the reference-gait action
  scaffold.

## Running

```bash
source isaac_env/bin/activate   # from the repo's environment setup
cd walking-solutions/ppo/baseline_test24

# Full chunked training run (resumable, RAM-gated):
./run_chunked_training.sh

# Or a single manual run:
python train_ppo.py --headless --timesteps 200000 --run-name baseline_test24

# Evaluate a trained checkpoint:
python evaluate_policy.py --model runs/baseline_test24/final_model.zip --episodes 10
```

## Resuming the actual paused 20.9M-step run

The checkpoint from the real paused run is **not included in this repo**
(model checkpoints are gitignored — see the repo's `.gitignore`). To
resume it here, copy the checkpoint into this folder's expected location
before running the script:

```bash
mkdir -p runs/baseline_test24
cp <path-to-your-local-checkpoint>/final_model.zip runs/baseline_test24/final_model.zip
./run_chunked_training.sh
```

The original run's checkpoints live locally under
`nao_walking_project/experiments/test_24/runs/test24/` (not committed).

## Known issue

The `config.txt` auto-generated per run has historically been stale in
earlier experiments (still describing a `test_16`-era hyperparameter
scheme) — this was fixed in this version's `train_ppo.py` to reflect the
actual configuration above. If a checkpoint's `config.txt` still shows
the old text, it predates this fix; the real values are always the ones
in this README and in the code.
