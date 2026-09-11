# test_26_sac — SAC pilot (vanilla SB3 defaults)

**Status: in progress / not yet committed.** Per project convention, this
folder is only committed once the algorithm is confirmed working in a
meaningful way — this document is written first, alongside the code, and
will be updated with results once the pilot run completes.

## Goal

Introduce Soft Actor-Critic (SB3 `SAC`) as the second RL algorithm for
NAO forward walking, directly comparable to the PPO baseline
(`test_24` / `walking-solutions/ppo/baseline_test24`).

## Why SAC

- **Off-policy, replay-buffer-based.** Isaac Sim is a slow single
  environment here (no vectorized/multi-robot stepping in this setup) —
  SAC can reuse past experience via its replay buffer instead of
  discarding a rollout after each on-policy update the way PPO does.
  With a slow env, sample efficiency per env step matters more than it
  would with cheap, highly parallel simulation.
- **Direct precedent on NAO walking.** Chenatti & Colombini's LSTM-SAC
  work applies SAC (with recurrence) specifically to NAO locomotion,
  informing this choice as a natural next algorithm to try (the LSTM
  variant is a later stretch goal, not part of this pilot).

## Controlled-comparison design

The only deliberate variable is the algorithm. Held fixed, identical to
`test_24`:
- Physics setup (`physics_dt=1/120s`, `control_dt=1/30s`, spawn height,
  three-tier PD gains).
- Observation space (23-dim) and action space (12-dim residual on the
  literal Webots reference gait, `action_scale=0.25` rad).
- `nao_walking_env.py` and `webots_reference_gait.py` are carried over
  **unmodified** except for asset paths, now pointing at
  `walking-solutions/shared/` instead of a local copy (per the project's
  shared-asset convention — no more duplicating the 12MB robot package
  per experiment).

Expected to change: the reward may eventually need retuning for SAC's
objective, but this pilot makes **no reward changes** — same reward
formula as `test_24`, to isolate the algorithm's effect first.

## Audit findings that shaped the implementation (full detail in session
notes; summarized here for the record)

1. **Chunked training breaks SAC's replay buffer if handled naively.**
   `run_chunked_training.sh` restarts the Python process every chunk.
   PPO tolerates this because its rollout buffer is transient anyway.
   SAC's replay buffer is excluded from `model.save()` by default (SB3
   2.7.1, confirmed in `common/base_class.py`) and must be persisted
   explicitly via `save_replay_buffer()` / `load_replay_buffer()`.
   Additionally, `train_ppo.py`'s resume path manually zeroes
   `model.num_timesteps` — copying that for SAC would make its
   `learning_starts` random-exploration warmup re-trigger at every chunk
   boundary instead of once at the true start of training. Fix: never
   zero `num_timesteps` on resume, and pass `reset_num_timesteps=False`
   to `learn()`.
2. **`terminated`/`truncated` are already handled correctly** in
   `nao_walking_env.py` (fall/NaN vs. time-limit are independent, never
   conflated) — verified by reading `step()`/`_check_termination()`
   directly; no changes needed, confirmed empirically in the smoke test
   below.
3. **Exploration vs. residual action space — deferred.** SAC's warmup
   phase samples uniform-random actions across the full residual range
   every control step, which is a much rougher signal than the smooth
   reference-gait trajectory the physics was tuned around. Candidate
   mitigations (gSDE with low `log_std_init`, smaller `learning_starts`,
   lower `target_entropy`) were evaluated but **deliberately not applied
   in this pilot** — we're running vanilla SB3 defaults first and will
   decide on exploration changes from the pilot's actual data, so that
   any later exploration change is a separately documented, clean
   before/after comparison for the report.
4. **PPO-specific assumptions removed:** `train_sac.py` has its own SAC
   constructor (no `n_steps`/`clip_range`/`gae_lambda`/etc.), `ent_coef`
   defaults to `"auto"` (not PPO's fixed-weight `0.01`), and
   `evaluate_policy.py` is now algorithm-aware (`--algo {ppo,sac}`).

## Pilot configuration

SB3 2.7.1 defaults, deliberately un-tuned, with only the two parameters
needed to match `test_24` for comparability:

| Param | Value | Source |
|---|---|---|
| `learning_rate` | 3e-4 | SB3 default |
| `buffer_size` | 1,000,000 | SB3 default |
| `learning_starts` | 100 | SB3 default |
| `batch_size` | 256 | SB3 default |
| `tau` | 0.005 | SB3 default |
| `train_freq` / `gradient_steps` | 1 / 1 | SB3 default |
| `ent_coef` | `"auto"` | SB3 default |
| `target_entropy` | `"auto"` (= -12 for our 12-dim action space) | SB3 default |
| `use_sde` | `False` | SB3 default — exploration change deferred (see above) |
| `net_arch` | `[128, 128]` | overridden from SB3's `[256, 256]` default, to match test_24's PPO network capacity |
| `gamma` | 0.99 | matched to test_24 |

**This is a ~2 hour pilot, not a commitment to 50M steps.** The chunked
runner is time-bounded for this run (see `run_chunked_training.sh`).

## Verification checklist (before trusting any pilot results)

- [ ] `NaNSafeWrapper` sanitizes both the returned observation *and*
      reward on a NaN/Inf step, not just the termination flag (a NaN
      reward alone, with obs otherwise fine, must not poison the
      critic's TD target).
- [ ] Replay buffer saves are atomic (write to a temp file, then
      `os.replace()`) so a RAM-gate kill mid-write can't corrupt the
      only on-disk copy.
- [ ] Replay buffer RAM footprint computed from the actual obs/action
      dims and checked against the chunk runner's `MIN_RAM_MB` gate.
- [ ] Resume uses `SAC.load(..., env=env)` with the default
      `force_reset=True` (confirmed default in SB3 2.7.1 source).
- [ ] Smoke test spans at least one chunk boundary: replay buffer
      restores, `num_timesteps` keeps counting across the restart (not
      reset to 0), and `learning_starts` warmup happens exactly once
      across the whole run, not once per chunk.

## Smoke test results (before launching the pilot)

Ran two manual 150-step chunks (fresh, then `--resume`) directly, bypassing
the chunk-runner's RAM-gate sleep loop for speed:
- `num_timesteps`: 150 after chunk 1, **300** after chunk 2 (continuous,
  not reset) — confirmed by loading the saved model directly.
- Replay buffer: 150 transitions after chunk 1, **300** after chunk 2
  (correctly appended, not overwritten) — matches `num_timesteps` exactly.
- No NaNs found in stored observations, rewards, or actions in the buffer.
- `log_ent_coef` evolved smoothly across the resume (-0.015 → -0.060),
  confirming the entropy-tuning optimizer state survived the restart
  intact, not reinitialized.
- Measured replay buffer file size: 244,001,666 bytes — matches the
  ~244MB hand-calculation above almost exactly.
- **Bug found and fixed during the smoke test:** `Monitor`'s default
  `override_existing=True` wipes `monitor.monitor.csv` back to empty on
  every chunk restart, which would have silently thrown away
  episode-level history from every chunk but the last — directly
  relevant to the "ep_len_mean/ep_rew_mean over time" deliverable below.
  Fixed with `override_existing=(args.resume is None)` (fresh + header on
  the first chunk, append on every resume). Verified fixed: a second
  two-chunk run showed all 4 episodes (2 per chunk) accumulated in one
  file.

## Launching the pilot

```bash
cd experiments/test_26_sac
source ../../nao_walking_project/isaac_env/bin/activate
./run_chunked_training.sh
```

Time-bounded to `MAX_WALLCLOCK_SECONDS=7200` (~2h) via
`run_chunked_training.sh`, with a 3,000,000-step ceiling as a backstop
that isn't expected to bind. `CHUNK_STEPS=5000` (smaller than PPO's
50,000) so the run crosses many replay-buffer persistence boundaries
during the pilot, deliberately stress-testing the resume mechanism this
audit was built around, at a measured ~15-20s Isaac Sim restart overhead
per chunk (~7-8% of the pilot's wall-clock budget).

## Pilot results

**Completed 145,000 steps in 7,324s (~2h02m) across 29 chunks, zero
crashes/retries/NaNs.** Overall throughput ~19.8 steps/sec including
per-chunk Isaac Sim restart overhead (~15-20s/chunk); pure training-loop
throughput (SB3's own `time/fps`, excluding restarts) held at ~17
steps/sec through most of training, rising to ~20-21 in the final chunk.

### `ep_len_mean` / `ep_rew_mean` over time (from TensorBoard, `rollout/*`, 630 logged points; corroborated by the raw 2,522-episode `monitor.monitor.csv`, which accumulates correctly across chunks now that the bug below is fixed)

| step (≈) | ep_rew_mean | ep_len_mean |
|---|---|---|
| 8,000 | 62.0 | 51.5 |
| 24,000 | 74.0 | 51.6 |
| 41,000 | 85.9 | 52.5 |
| 58,000 | 95.0 | 57.1 |
| 75,000 | 113.1 | 52.8 |
| 93,000 | 118.6 | 57.5 |
| 111,000 | 154.4 | 61.4 |
| 132,000 | 192.5 | 67.9 |
| 144,000 | 192.1 | 73.6 |

Clear, steady upward trend in both — reward roughly tripled and episode
length grew ~43% from start to end of the pilot. Still, 62% of all 2,522
episodes end at exactly 51 steps (the `grace_period=50` floor + 1) —
most episodes still fall almost immediately once fall-detection turns
on; a growing minority survive much longer (up to 125 steps by the end).
This reads as genuine early-stage learning, not noise or a fluke.

### Losses, entropy, and Q-values

- **`critic_loss`**: drops from a noisy ~97-177 in the first ~40k steps
  down to a stable ~25-28 plateau from ~90k steps onward — the value
  function is converging, not diverging.
- **`ent_coef`** (auto-tuned temperature): smooth, monotonic decay from
  ~0.28 to ~0.04, stabilizing after ~75k steps — textbook SAC entropy
  annealing, no instability.
- **`ent_coef_loss`**: oscillates near zero throughout, consistent with
  `ent_coef` having converged near its equilibrium.
- **`actor_loss`**: −95 → −134 (early, while `ent_coef` was still high)
  → stabilizes around −75 to −80. Not directly interpretable as
  "better/worse" on its own since it's coupled to the shrinking entropy
  term, but the stabilization pattern is consistent with the rest.
- **Mean Q-value** (SB3 doesn't log this by default, so computed
  directly: loaded `ckpt_10000`, `ckpt_75000`, and `final_model`, ran
  each through `min(critic_1, critic_2)` on the *same* fixed sample of
  512 states drawn from the final replay buffer, to isolate how each
  checkpoint's value estimate changed holding the state distribution
  fixed): **125.2 → 83.7 → 78.0**. The 10k-step estimate is the least
  trustworthy of the three (that's precisely when `critic_loss` was
  highest/noisiest), so this reads as early overestimation settling down
  as the critic converges, not the value function degrading with more
  training.

### Bug found and fixed during the smoke test (see above) — confirmed fixed in the real pilot

`Monitor`'s `override_existing=True` default would have wiped
`monitor.monitor.csv` every chunk; fixed with
`override_existing=(args.resume is None)`. Confirmed fixed: the real
145k-step run's `monitor.monitor.csv` contains all 2,522 episodes across
all 29 chunks, not just the last one.

### Second bug found while gathering these results (evaluation tooling, not training)

`evaluate_policy.py`'s plain `print()` output does not appear in a
piped/redirected shell capture of this environment's Isaac Sim
process — Isaac Sim's `omni.kit.app._impl` intercepts Python's stdout
and reroutes it into its own `[py stdout]:`-tagged lines inside a Kit
log file (`.../omni/logs/Kit/Isaac-Sim Python/4.5/kit_<timestamp>.log`)
instead of the process's original stdout. `train_sac.py` avoids this by
writing explicitly to its own opened `--log-file` handle rather than
relying on `sys.stdout`, but `evaluate_policy.py` has no such mechanism.
The evaluation itself completed correctly in both cases (exit code 0) —
only the *captured* output was affected, not by fabricating any numbers,
but the affected results were recovered by reading them out of the Kit
log directly. **Worth fixing** (add the same `--log-file`/`TeeOutput`
pattern to `evaluate_policy.py`) before this becomes a recurring
annoyance — flagged, not yet fixed, since it's a tooling issue orthogonal
to the SAC pilot itself.

### Final checkpoint, deterministic evaluation (`evaluate_policy.py --algo sac --model runs/sac_pilot/final_model.zip`, 10 episodes)

All 10 episodes were **identical** (expected: deterministic policy +
deterministic env/gait replay = deterministic trajectory):
- Survived 10/10 to termination (all ended in a fall, none hit
  `max_steps`)
- Episode length: **101 steps**
- Episode reward: **181.4**
- Forward distance: **0.506 m**
- Average velocity: **0.150 m/s**
- Termination: fell, height=0.075m (below `term_height=0.10`)
- Classified "✓ WALKING DETECTED" (>0.5m) by `evaluate_policy.py`'s own
  threshold, short of "★ SUSTAINED WALKING!" (>2.0m)

### Comparison to PPO (`test_24`) — important caveat

**A true same-step-count comparison (PPO at ~145k steps) is not
reconstructable from the surviving data.** Two independent problems, both
found while gathering this report:
1. `test_24`'s real run also has the `Monitor` override bug (never
   fixed there) — its `monitor.monitor.csv` only ever contains the *most
   recent* chunk's episodes, not history from the ~145k-step mark (the
   run is now at 22M+ steps).
2. `test_24`'s `training_log.txt` never captured per-episode reward
   output as parseable text (likely the same stdout-capture issue
   described above, compounded by `rich`'s progress-bar rendering), and
   no `tb/` TensorBoard directory exists for that run at all.
3. No checkpoint was saved near 145k steps specifically — the chunked
   runner only milestone-saves every 500k steps. **`model_500000.zip`
   is the earliest genuinely-labeled checkpoint available**, at ~3.4x
   more training than the SAC pilot.

Rather than fabricate a step-matched number, `model_500000.zip` was
evaluated deterministically instead (same procedure, 10 episodes),
clearly labeled as 500k steps, not 145k:

| | SAC @ 145k steps | PPO @ 500k steps (≈3.4x more training) |
|---|---|---|
| Episode length | 101 steps | **1,500 steps (hit `max_episode_steps` every time)** |
| Episode reward | 181.4 | **5,460.3** |
| Forward distance | 0.506 m | **11.430 m** |
| Avg. velocity | 0.150 m/s | **0.229 m/s** |
| Termination | fell (h=0.075m), all 10/10 | never fell, all 10/10 hit the step cap |

PPO at 500k steps is a fully sustained walker (every episode runs the
full 1500-step window without falling); SAC at 145k steps is still in
the early "falls almost immediately, occasionally survives longer"
regime. **This is not an apples-to-apples algorithm comparison** — the
step counts differ by ~3.4x, so it conflates "SAC vs. PPO" with
"145k steps vs. 500k steps of training." It's included because it's the
only genuinely-labeled real data available, not as a claim that PPO
outlearns SAC per-step. A true same-step comparison would need either
accepting this mismatch explicitly (as done here) or a dedicated
145k-step PPO training run purely for comparison.

## Overnight extension (same run, resumed)

Decided to continue the *same* `sac_pilot` run (not restart) overnight
rather than treat 145k steps as final, given the clean mechanism and
steady trend above. Changed in `run_chunked_training.sh`:
- `MAX_WALLCLOCK_SECONDS`: 7200 → **43200** (12h)
- `CHUNK_STEPS`: 5,000 → **15,000** — the frequent-boundary pilot already
  verified the resume mechanism across 29 clean restarts, so the
  priority shifts to cutting restart overhead (~7-10% of wall-clock at
  5,000 steps/chunk → ~2% at 15,000) for the longer run, while still
  restarting often enough for Isaac Sim's long-run memory hygiene.

Resumes from the existing `runs/sac_pilot/final_model.zip` +
`replay_buffer.pkl` (145,000 steps) — continuous learning curve, nothing
discarded. Expected to add roughly 800,000-900,000 more steps over 12h at
the improved throughput, landing around ~950,000-1,050,000 total.

**Two known issues (Monitor-override-in-PPO, evaluate_policy.py's
stdout-capture gap) were reviewed and confirmed to NOT affect this run**:
neither touches `train_sac.py`'s training path — the first only ever
affected the original PPO script, the second only affects reading
`evaluate_policy.py`'s printed output after the fact, not evaluation
correctness or training. Left unfixed for now, tracked below.

## Overnight results (145,000 → 1,165,000 steps)

Completed 1,020,000 additional steps in 43,370s (~12h03m), 68 more
chunks (97 total across both sessions), zero crashes/retries/NaNs.
Throughput improved as predicted from raising `CHUNK_STEPS`: SB3's own
`time/fps` climbed from ~17 (matching the initial pilot) to a steady
~22-24 by the run's end; overall throughput (incl. restart overhead)
~23.5 steps/sec, up from ~19.8 in the first session. Replay buffer is
now **full** (hit its 1,000,000-capacity cap and is recycling — oldest
transitions from early in the run have been evicted).

### `ep_len_mean` / `ep_rew_mean` — continued, clear improvement

| step (≈) | ep_rew_mean | ep_len_mean |
|---|---|---|
| 35,000 | 80.8 | 53.3 |
| 109,000 | 151.4 | 61.4 |
| 288,000 | 206.5 | 70.4 |
| 483,000 | 234.7 | 75.8 |
| 682,000 | 240.0 | 74.6 |
| 885,000 | 258.2 | 78.9 |
| 1,105,000 | 273.4 | 86.1 |
| 1,164,000 | 278.7 | 88.5 |

From the raw 15,822-episode monitor log: first-50 vs. last-50 episodes,
reward **55.2 → 279.5** (5x), length **51.1 → 88.9** (1.7x). Max episode
length ever seen: 141. Max episode reward ever seen: 402.

**The "falls almost immediately" regime has substantially broken down.**
In the first 145k-step pilot, 62% of all episodes ended at the
`grace_period` floor (51 steps). Across the full 1.165M-step run that
fraction is down to **19.2% overall**, and just **5.8% of the last 500
episodes** — i.e., the pattern that motivated the deferred gSDE/
exploration discussion has largely resolved on its own with more
training, without needing that change. Worth keeping in mind if the
gSDE question comes back up: vanilla SAC did get past this regime given
enough steps, on this task.

### Losses, entropy, Q-values — mostly healthy, one trend worth watching

- **`ent_coef`**: continues its smooth decay/stabilization, ~0.13 early
  in this session down to a stable ~0.029-0.033 for most of the back
  half — no instability.
- **`actor_loss`**: keeps trending more negative throughout (−76 → −108
  by the end), not just plateauing. Expected/consistent with genuine
  improvement (Q-value estimates growing as the policy reaches better
  states, outpacing the shrinking entropy term) rather than a red flag
  on its own — but flagging the trend rather than calling it "converged."
- **`critic_loss`: worth watching.** Converged to ~30 by ~110k-200k
  steps (matches the first pilot's finding), but then **crept back up
  to ~56 by the end of the run** — a real, sustained rise over hundreds
  of thousands of steps, not noise. Most likely explanation: reward
  magnitudes have grown substantially (`ep_rew_mean` roughly tripled
  this session), and since critic loss is a squared TD-error, absolute
  loss naturally scales up with the target's magnitude even for an
  equally-well-fit critic — a growing Q-value scale, not necessarily a
  growing *relative* fitting error. Consistent with mean-Q staying
  roughly stable (below) rather than diverging. Not treating this as
  resolved — worth checking again if training continues further and the
  rise doesn't level off.
- **Mean Q-value** (same method as before — fixed 512-state sample from
  the final buffer, evaluated through each checkpoint's own critic):
  131.1 (10k) → 98.2 (145k) → 108.2 (1M) → 110.8 (final, 1.165M). Settled
  and stable across the back half of training (108→111), consistent
  with the critic_loss interpretation above — not runaway.

### Final checkpoint evaluation (1,165,000 steps, deterministic, 10 episodes)

All 10 episodes identical again (expected — deterministic policy +
deterministic env/gait):
- Episode length: **93 steps**
- Episode reward: **349.0**
- Forward distance: **1.710 m**
- Average velocity: **0.552 m/s**
- Termination: fell, height=0.098m — just under `term_height=0.10`,
  i.e. very close to *not* falling
- Classification: "✓ WALKING DETECTED" (>0.5m), well short of but much
  closer to "★ SUSTAINED WALKING!" (>2.0m) than before

### Progression summary (145k → 1.165M steps)

| | 145k steps | 1.165M steps | change |
|---|---|---|---|
| Episode length (det. eval) | 101 | 93 | shorter |
| Episode reward (det. eval) | 181.4 | 349.0 | **+92%** |
| Forward distance (det. eval) | 0.506 m | 1.710 m | **+238%** |
| Avg. velocity (det. eval) | 0.150 m/s | 0.552 m/s | **+268%** |
| Fall height at termination | 0.075 m | 0.098 m | closer to surviving |
| % episodes at grace-period floor (training) | 62% | 19.2% (5.8% of last 500) | major improvement |

The deterministic episode got *shorter* but covers far more ground at
much higher speed before falling — gait quality/speed improved
substantially even though it hasn't yet crossed into sustained walking.
The fall height (0.098m, right at the 0.10m threshold) suggests it's
close to clearing that bar with more training.

## Visual inspection (final checkpoint, live viewport)

Watched the final 1.165M-step checkpoint directly in Isaac Sim
(`evaluate_policy.py --algo sac`, non-headless, `--slow`). Confirms the
deterministic-eval numbers qualitatively — it advances several steps of
real forward walking — but also surfaces something the numbers alone
didn't show: **it drifts/turns left for no apparent reason** partway
through the walk, rather than continuing straight. Not visible in any of
the logged metrics above (`penalty_lateral_vel` penalizes lateral
*velocity*, not heading drift, and there's no yaw/heading term in the
reward at all — see `test_24`'s reward formula, unchanged here), so
this is a real gap between what's being optimized and what "good
walking" means, worth keeping in mind for future reward-shaping
discussion rather than something the current metrics would ever catch.

## Status

**Committed to the repository at 1,165,000 steps** — mechanism fully
verified (chunking, replay buffer persistence, exploration warmup all
correct across 97 chunk restarts), learning trend clear and substantial,
confirmed both numerically and visually. Not yet a sustained walker, and
not yet claimed to be — see the leftward-drift finding above.

## Next steps

- Investigate the leftward-drift behavior found in visual inspection —
  the reward has no heading/yaw term at all, only a lateral-*velocity*
  penalty, so nothing currently discourages turning. Worth deciding
  whether to add a heading-deviation penalty, or whether more training
  resolves it the way the "falls almost immediately" regime did.
- Continue the SAC pilot further (more wall-clock) to see whether the
  reward/length trend keeps climbing, plateaus, or the `critic_loss`
  uptick noted above stabilizes or keeps growing.
- Fix `evaluate_policy.py`'s stdout-capture gap (add `--log-file`/
  `TeeOutput`) so future evaluations don't require digging through Kit
  logs.
- Fix the `Monitor` override bug in the original PPO script too, so
  future PPO runs don't lose the same episode history SAC's did.
- Decide whether a dedicated same-step PPO comparison run is worth the
  wall-clock cost for the academic report, or whether the trend data
  above (SAC's own learning curve) is sufficient on its own.
