# test_24

**Focus:** Added a lateral-drift penalty (to stop sideways wandering) on
top of `test_23`'s fixed sign convention, and scaled the training target
10× (5M → 50M steps).

**Outcome — the current, accepted working baseline.** Approximately 20.9M
of the 50M target steps completed at time of writing; training appears to
have been paused rather than stopped due to failure (clean chunk logs
throughout).

## Configuration, for the record

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
  with checkpoint-based resumption and RAM-availability gating —
  apparently working around Isaac Sim's long-run memory/stability
  behavior (see `run_chunked_training.sh`).

**Known issue:** this experiment's auto-generated `runs/*/config.txt` is
stale — it still describes a much earlier hyperparameter scheme from
`test_16` and does not reflect the actual configuration above. The real
values are only in the code (`nao_walking_env.py`, `train_ppo.py`). This
is itself a small lesson about keeping auto-logged run metadata in sync
for reproducibility — worth fixing before starting the second RL
algorithm.

## Key learning

Diff from `test_23` is small (~10 lines: a new `penalty_lateral_vel`
weight, a squared lateral-velocity penalty term, and a raised alive-bonus
scale) — the project's last few working iterations advanced through
targeted, individually-verifiable changes rather than large rewrites.
That discipline is worth carrying into the next algorithm's development.

## Note on shared assets

`nao_gait_data/` is not duplicated in this archival snapshot (same as
`test_22`/`test_23`) — see `walking-solutions/shared/reference_gaits/`.

This experiment is also the seed for the **working/promoted tier**
(`walking-solutions/ppo/`) — that migration is a separate step, where
this same configuration gets re-pointed at the shared robot/materials/
reference-gait assets instead of local copies.
