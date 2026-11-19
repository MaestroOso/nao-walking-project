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

**Extended again for 24h** (same run, resumed from 1,165,000 steps) to
see whether the improvement trend continues, whether the `critic_loss`
uptick stabilizes or keeps growing, and whether the fall-height margin
(0.098m vs. the 0.10m threshold) closes further. `MAX_WALLCLOCK_SECONDS`
-> 86400, `TOTAL_STEPS` backstop raised to 5,000,000 (the old 3,000,000
ceiling would have bound partway through at current throughput);
`CHUNK_STEPS` unchanged at 15,000.

## 24h extension results (1,165,000 → 3,325,000 steps)

Completed 2,160,000 additional steps in 86,778s (~24h06m), 144 more
chunks (241 total across all three sessions), zero crashes/retries/NaNs.
Throughput kept improving: SB3's `time/fps` now steady at ~23-25;
overall throughput (incl. restart overhead) ~24.9 steps/sec, up from
~23.5 in the 12h session and ~19.8 in the initial pilot.

### `ep_len_mean` / `ep_rew_mean` — still climbing

| step (≈) | ep_rew_mean | ep_len_mean |
|---|---|---|
| 1,246,000 | 279.6 | 86.6 |
| 1,570,000 | 267.2 | 84.8 |
| 1,911,000 | 315.1 | 92.5 |
| 2,269,000 | 347.4 | 97.0 |
| 2,638,000 | 362.1 | 99.0 |
| 3,022,000 | 380.1 | 104.2 |
| 3,324,000 | 389.5 | 108.2 |

Brief dip around 1.4M-1.6M steps, then resumed climbing through the
whole rest of the session — not a plateau yet. From the raw 38,456-
episode monitor log: first-50 vs. last-50, reward **55.2 → 385.3** (7x
since the very start), length **51.1 → 110.1** (2.15x). Max episode
length ever seen: 158. Max episode reward ever seen: 539.

**The "falls almost immediately" regime is now nearly gone.** 62%
(initial pilot) → 19.2% (after 12h) → **9.5% overall, 0.4% of the last
500 episodes**. This failure mode has essentially resolved itself with
continued training.

### Losses, entropy, Q-values

- **`critic_loss`: the earlier uptick reversed.** It had crept from ~30
  to ~56 by the end of the 12h session (flagged as "worth watching," not
  yet a problem). Over this 24h session it **decreased** from ~50 back
  down to **~30**, ending the session lower than it started. Reads as
  transient — likely tied to the reward-scale still growing at that
  point — rather than a sustained degradation.
- **`ent_coef` reversed direction**: after monotonically decaying for the
  first ~1.2M steps (0.28 → ~0.03), it's now **climbing back up** (0.033
  → 0.057 over this session) — SAC's auto-tuning pushing entropy back up
  as the policy's actual entropy fell below `target_entropy` (-12). Not
  a red flag on its own (this is exactly what the auto-tuning mechanism
  is for), but a real reversal worth tracking if it continues.
- **`actor_loss`** continues trending more negative (−112 → −125),
  consistent with the pattern from the previous session.
- **Mean Q-value** (same fixed-512-state method, now sampled from the
  final full/recycling buffer): 123.9 (10k) → 101.3 (1.165M) → **123.7**
  (final, 3.325M) — went down then back up to roughly where it started.
  Bounded, not runaway, though this method evaluates checkpoints against
  a state sample that mixes old and new experience, so it's a rougher
  signal than it would be with a fixed evaluation set — treat as
  directional, not precise.

### Final checkpoint evaluation (3,325,000 steps, deterministic, 10 episodes)

All 10 episodes identical again (expected):
- Episode length: **101 steps**
- Episode reward: **401.5**
- Forward distance: **1.934 m**
- Average velocity: **0.574 m/s**
- Termination: fell, height=**0.100m** — right at the `term_height=0.10`
  threshold, the closest yet to not falling at all
- Classification: "✓ WALKING DETECTED" — 1.934m is now within 7cm of the
  "★ SUSTAINED WALKING!" (>2.0m) threshold

### Full progression (all three sessions, deterministic eval each time)

| | 145k steps | 1.165M steps | 3.325M steps |
|---|---|---|---|
| Episode reward | 181.4 | 349.0 | **401.5** |
| Forward distance | 0.506 m | 1.710 m | **1.934 m** |
| Avg. velocity | 0.150 m/s | 0.552 m/s | **0.574 m/s** |
| Fall height | 0.075 m | 0.098 m | **0.100 m** (at threshold) |
| % episodes at grace-period floor (training) | 62% | 19.2% | **9.5%** (0.4% of last 500) |

The fall-height margin has now closed to essentially zero, and forward
distance is approaching the sustained-walking threshold. The trend
across all three checkpoints is monotonic on every metric — no sign of
having plateaued yet.

## Status (updated)

**3,325,000 steps, not yet committed** (this extension). Mechanism
remains fully verified across 241 total chunk restarts. Every metric —
reward, distance, velocity, fall-height margin, grace-period-floor
fraction — has improved monotonically across all three checkpoints
evaluated so far (145k / 1.165M / 3.325M). Not yet sustained walking by
the script's own >2.0m bar, but only 0.066m short of it at the current
checkpoint. Leftward-drift issue (visual inspection, 1.165M-step
checkpoint) not yet re-checked at this checkpoint.

**Extended a third time for 12h** (resumed from 3,325,000 steps), given
the trend still hadn't plateaued and the fall-height margin had closed
to essentially zero — close enough to sustained walking to be worth
seeing whether it crosses that line with more training.
`MAX_WALLCLOCK_SECONDS` -> 43200, `TOTAL_STEPS`/`CHUNK_STEPS` unchanged
(projected ~4.4M steps by the end, still under the 5,000,000 backstop).

## 12h extension results (3,325,000 → 4,450,000 steps) — apparent breakthrough near the end

Completed 1,125,000 additional steps in 43,733s (~12h09m), 75 more
chunks (316 total across all four sessions), zero crashes/retries/NaNs.
Throughput continued improving slightly: overall ~25.7 steps/sec, up
from ~24.9 in the previous session.

For most of the session, the trend continued the same steady climb as
before: `ep_len_mean` ~109 → ~119, `ep_rew_mean` ~396 → ~448 through the
first ~90% of this session's steps. **Then, in roughly the last ~150,000
steps, something changed qualitatively**, not just incrementally:

- `ep_len_mean` jumped from ~119 to **180** in the final logged points
  (individual episodes up to 182 in that window).
- `ep_rew_mean` jumped from ~448 to **527**.
- Individual episode lengths in the raw monitor log near the end include
  **543, 266, 300, 870, and 775 steps** — far beyond anything seen in
  any prior session (previous max across the whole run was 158 steps).
- **Max episode length ever recorded: 1,263 steps** — 84% of
  `max_episode_steps=1500`, i.e. close to the full episode window
  without falling. **Max episode reward ever: 2,671.6.**

This is not yet consistent, though — in the last 300 episodes: mean
length 154, but ranging from 52 (still occasionally falls almost
immediately) to 1,263. Only 22 of the last 300 episodes exceeded 300
steps, and 10 exceeded 500. **The grace-period-floor fraction has
otherwise kept improving** (7.6% overall, 0.2% of the last 500 episodes)
— so the *typical* episode is still ~150ish steps, but the policy has
now demonstrably found the long-duration behavior at least some of the
time, which it had never done at all in the first three sessions.

Mean Q-value (same fixed-sample method): 117.2 (10k) → 114.8 (3.325M) →
**128.2** (final) — still bounded, ticking up alongside the reward jump.

### Final checkpoint evaluation (4,450,000 steps, deterministic, 10 episodes)

All 10 episodes identical again:
- Episode length: **1,500 steps — hit `max_episode_steps` every time**
- Episode reward: **3,451.6**
- Forward distance: **2.680 m**
- Average velocity: **0.054 m/s**
- Termination: none — never fell, all 10/10

By raw numbers this crosses both thresholds that have been tracked
throughout: never falls, and covers >2.0m (`evaluate_policy.py` prints
"★ SUSTAINED WALKING!"). **But this needs an important caveat, not a
victory lap:**

**Velocity dropped ~10x, from 0.574 m/s (previous checkpoint) to
0.054 m/s.** The policy isn't walking fast and reliably — it's moving
*extremely* slowly and cautiously for the full 1500-step window. Given
the reward formula (`5.0·(forward vel) + slow-ramping alive bonus −
orientation/lateral penalties`, `alive_scale=500`, ramping with step
count — see `test_24`'s reward, unchanged here, and recall `test_12`'s
own history of a *different* alive-bonus exploit earlier in this
project), a policy that survives all 1500 steps banks a large cumulative
alive bonus regardless of how much forward progress it makes. The reward
(3,451.6) is far higher than any previous checkpoint's, but that's
consistent with either "genuinely much better" or "found a way to farm
the alive bonus by barely moving instead of walking" — **the eval script's
own distance-only classification ("SUSTAINED WALKING") doesn't
distinguish between these**, which is itself worth noting: distance
crossing 2.0m was being used as the sustained-walking bar, but 2.68m
over 1500 steps at 0.054 m/s is not the same achievement as 2.68m
covered quickly. This looks like the same *category* of problem as
`test_12`'s fall-fast exploit and `test_18`'s dive-and-slide exploit —
a reward-hacking pattern specific to this alive-bonus structure — just a
different, opposite-direction instance of it (crawl-slowly-forever vs.
fall-fast).

**Not concluding this is definitely reward hacking rather than genuine
progress** — worth checking against the training-time stochastic
behavior (which showed real variance, including some fast, short, high
per-step-reward episodes) and via visual inspection before deciding.
But the velocity collapse is too large to just note in passing; this is
the main thing to resolve before treating 4.45M steps as the new
baseline.

### Visual inspection (4,450,000-step checkpoint)

Confirms the concern above. The robot **tilts over onto one leg and
holds there, statically balanced, never falling** — not walking. This
is consistent with the alive-bonus-farming hypothesis: a static,
lopsided-but-stable pose trivially avoids the fall-termination
condition and banks the survival-scaled alive bonus for the entire
1500-step episode, which apparently outweighs whatever reward genuine
(riskier) forward walking would have earned. Decided to continue
training anyway to see whether more steps (and `ent_coef`'s renewed
climb, reinjecting exploration) let it escape this local optimum on its
own, before touching the reward function. If a further extension
doesn't move off this static pose, the likely fix is reward-side —
capping/reshaping the alive bonus so survival alone can't outweigh
forward progress, the same category of fix `test_12` and `test_18`
needed for their own alive-bonus exploits.

## 24h extension #2 results (4,450,000 → 7,060,000 steps) — mixed: real training-time progress, but the deterministic policy still shows the same underlying pattern

**Incident: a genuine GPU hang, not slow training.** Partway through
this session, one chunk's Isaac Sim process deadlocked at the renderer
level (`[Fatal] rtx.scenedb.plugin: Waiting on Semaphore ... longer than
60s`, 0% GPU utilization, ~5 hours with no progress vs. the normal ~10
minutes/chunk). Killed the stuck process manually; the wrapper's
existing retry logic (`if $? -ne 0 ... retrying in 10s`) picked it up
correctly and resumed cleanly from the last saved checkpoint
(7,060,000 steps) — but by the time it retried, the ~5-hour stall had
already pushed elapsed wall-clock past the 86400s cap, so the run
stopped instead of continuing. No training data lost (checkpoints are
saved before each chunk starts), but ~5 of the ~28.5 hours were wasted
wall-clock. First and only chunk failure in 491 total chunk attempts
across the whole project — a very low rate; the existing retry
mechanism handled it exactly as designed and needs no changes.

**Training-time metrics show real, substantial movement:**
- `ep_len_mean` jumped from the ~110-180 range to a peak of **940.9**
  early this session (individual episodes up to 1077), then oscillated
  in the 700-900 range for most of the session before **declining again
  toward the end** (down to ~480-535) — not a clean monotonic
  improvement, more volatile than any prior session.
- **597 episodes total have now hit `max_episode_steps` (1500) without
  falling at all** — a real, repeatable rate, not a one-off.
- First-50 vs. last-50 raw episodes: length 51 → 554, reward 55 → 1082.
- `critic_loss` continued improving (down to ~13-17, its best yet).
  `ent_coef` climbed further (0.06-0.09 range, higher than any prior
  session) — more sustained exploration throughout.

**Deterministic evaluation (7,060,000 steps, 10 episodes) — the
important number, and it's not a clean win:**
- Episode length: **791 steps** (down from 1500 at the 4.45M
  checkpoint)
- Reward: **1417.2**
- Forward distance: **2.333 m** (down slightly from 2.680m)
- Average velocity: **0.088 m/s** (up from 0.054 m/s, but still far
  below the 0.57 m/s seen at 3.325M steps, before the static-pose
  checkpoint)
- Termination: **tilt** (orientation limit, `qw=0.300`, exactly at
  `term_orient_qw=0.3`) — not a height-based fall this time

**This still looks like a variant of the same underlying strategy, not
a breakthrough into real walking.** At 4.45M steps, the deterministic
policy found a static, extremely-tilted pose stable enough to never
trip the height-based fall condition for the full episode. At 7.06M
steps, it's still tilting (that's literally the termination reason),
still moving very slowly (0.088 m/s), and now it eventually tips past
the *orientation* limit instead of surviving the whole episode — a
worse outcome by that specific measure, even though velocity ticked up
slightly. The much longer/higher-reward episodes visible in the
*training-time* stochastic data may be the exploration noise
(`ent_coef` climbing) occasionally finding genuinely better
trajectories that the deterministic policy hasn't consolidated onto —
worth visual inspection to confirm one way or the other, same as for
the 4.45M checkpoint.

## Reward revision — body-frame velocity + yaw-rate penalty

**Visual confirmation at 7,060,000 steps** (user inspection): tilts to one
side, turns in place, no forward movement, eventually falls — the same
underlying pattern as the 4.45M checkpoint, not resolved by more
training. Decided this needed a reward-side fix rather than more
wall-clock time.

**Diagnosis.** The reward had no term that required *directed* forward
locomotion:
- `forward_vel`/`lateral_vel` were raw **world-frame** velocity
  components — "forward" meant a fixed world axis, not "the direction
  the robot is facing." A robot that spins can register `r_forward`
  credit from incidental world-frame drift without ever walking in a
  consistent heading.
- There was **no yaw term at all**. `r_orient` (the existing orientation
  penalty) uses the projected-gravity vector, which only captures
  pitch/roll — rotating around the vertical axis costs nothing.
- The alive bonus (kept, deliberately — removing/weakening it risks
  reopening `test_12`'s fall-fast exploit, which the user had already
  independently observed happening again when alive-bonus pressure is
  too weak) still rewards survival regardless of whether that survival
  involves any actual locomotion.

**Fix implemented** (`nao_walking_env.py`, `_compute_reward`):
1. `forward_vel`/`lateral_vel` now computed in the robot's own body
   frame via `_quat_rotate_inverse` (the same helper already used for
   `proj_gravity`/`ang_vel` — no new primitive).
2. New `r_yaw = -w_yaw * yaw_rate²` term (`penalty_yaw_rate=0.5`, same
   scale as the existing orientation/lateral penalties), using
   `_get_angular_velocity`'s body-frame angular velocity (its z-component
   is yaw rate), with its `×0.25` observation-display scaling undone so
   the reward operates on the real magnitude.
3. Alive bonus, orientation penalty, and the `5.0`/`0.5` weights on the
   unchanged terms are all untouched.

**Sign convention empirically verified, not assumed**
(`verify_bodyframe_sign.py`, committed alongside): ran the literal
reference gait (zero residual action — known to walk in the intended
direction, since that's the convention the original `-v[0]` world-frame
fix in `test_23` was validated against) for 120 steps and compared
`body_vel[0]` against the already-validated world-frame forward signal.
Correlation: **-0.97** — body-frame `+x` is the robot's *backward*
direction, so the reward uses `forward_vel = -v_body[0]`, mirroring the
original world-frame sign fix for the new frame.

**Reward sanity-checked** (`verify_reward_terms.py`, committed
alongside) before trusting it in training: under the smooth
zero-residual gait, `yaw_rate` is tiny (abs mean 0.0058, implied
`r_yaw ≈ -0.0001`) — negligible penalty for walking as intended. Under a
jerky random-residual rollout, `yaw_rate` is much larger (abs mean 1.56,
max 9.7), implied `r_yaw ≈ -2.68` — a real, substantial penalty. No
NaNs or crashes in either case.

**Decided to modify `test_26_sac` in place rather than fork a new
experiment folder** — accepting that this breaks the original "only the
algorithm changes vs. `test_24`" premise for everything from this point
forward, in exchange for keeping one continuous, honestly-documented
history rather than fragmenting it. Everything above this section in
`APPROACH.md` describes the original reward (unchanged from `test_24`);
everything from here on describes training under the revised reward.

**Decided to start completely fresh** (new `SAC()`, empty replay
buffer) rather than warm-start from the 7,060,000-step checkpoint — the
old buffer's stored rewards were computed under the old formula and
would be flatly wrong under the new one, and the old actor/critic
weights are tuned to both the old reward's value scale and to the
degenerate tilt-and-spin behavior being moved away from. New run lives
under `runs/sac_pilot_v2/`; `runs/sac_pilot/` (the original 7.06M-step
lineage) is left untouched as the historical record this section
documents.

**Smoke-tested** (two 150-step chunks, fresh start then resume) before
trusting this in a real run: `num_timesteps` and replay buffer size both
continuous across the resume (300 after 150+150), zero NaNs, reward
range sane (early random-policy rewards are quite negative, -50 to
-157, consistent with an untrained policy wobbling into the new yaw
penalty on top of the existing terms — expected, not a bug).

**v2 run outcome (`runs/sac_pilot_v2/`, 0 → 4,995,000 steps):** recorded
so far only in `run_chunked_training.sh`'s header comment. The
deterministic checkpoint eval was the same at 3,255,000 and 4,290,000
steps: falls at step 51, fall height 0.062 m, with forward speed
0.238 → 0.319 m/s. The section below explains why the v2 revision could
not have worked as intended.

## Quaternion convention fix (v3) — 2026-11-17

**Bug.** Isaac Sim's `get_world_pose()` / `set_world_pose()` use
**scalar-first (w, x, y, z)** quaternions (confirmed in the installed
Isaac Sim 4.5 source: `isaacsim.core.prims/.../articulation.py`,
`get_world_poses` docstring). Every env in this project, back to
`test_10` and including `baseline_test24` (PPO), unpacked them as
`(x, y, z, w)` in `_quat_rotate_inverse` and in the finite-difference
angular-velocity fallback.

**Consequences** (derived analytically, then confirmed in a numpy
reproduction):

| True motion (20°) | Projected gravity, old code | Correct |
|---|---|---|
| Roll (side tilt) | `[0, 0, -1]`, **invisible** | `[0, -0.342, -0.940]` |
| Pitch | `[-0.342, 0, -0.940]` (sign flipped) | `[0.342, 0, -0.940]` |
| Yaw (turning) | `[0, 0.342, -0.940]`, **looks like a tilt** | `[0, 0, -1]` |

1. The policy could not observe side tilt, and the orientation penalty
   never charged for it. This matches the 4.45M-step checkpoint
   "tilting over onto one leg and holding there": that posture cost
   nothing.
2. The "orientation penalty" was in practice a *heading-deviation*
   penalty. Report Eq. 4-2 / Table 4-3 describe it as torso inclination.
3. The spawn orientation `[0, 0, 0, 1]` is read by Isaac as **(w, x, y, z)**,
   i.e. a 180° yaw. That is the real reason the robot's forward
   direction is world −x. The `test_23` world-frame sign fix (`-v[0]`)
   was correct, but for this reason, not a URDF axis convention.
4. v2's "body-frame" velocity: with the robot near upright, the
   misread quaternion is ≈ identity, so `v_body` ≈ world-frame
   velocity. The −0.97 correlation in `verify_bodyframe_sign.py` was a
   symptom of the bug, not a sign convention. The v2 reward change was
   therefore effectively only the yaw-rate penalty.
5. The termination check `|quat[3]| < 0.3` happened to be correct.
   Because q_spawn = (0,0,0,1), `quat[3]` equals the scalar part of the
   rotation relative to spawn, so ">~145° from the spawn pose" held.

**Fix (this experiment only — PPO baseline intentionally untouched for
now, so SAC no longer shares PPO's env):**
- `_quat_rotate_inverse` and the angular-velocity fallback now read
  (w, x, y, z). The fallback also now computes the body-frame delta
  `conj(q_prev) ⊗ q_curr`; the old formula computed the world-frame one.
- New `SPAWN_QUAT_WXYZ` constant (spawn unchanged, so world −x is still
  forward for evaluation/report purposes); termination now written as
  `|<q, q_spawn>| < 0.3`, which is numerically identical but explicit.
- New `_get_heading_velocity()`: (forward, lateral) velocity in the
  **heading frame** (world rotated by yaw only). It is used in the
  **reward**, with no sign flip, since URDF +x is forward. It is also
  used in the **observation**, replacing world-frame (vx, vy). Without
  that, once the gravity vector stops leaking yaw, the policy would have
  no way to interpret world-frame velocity after turning. A heading
  frame is used rather than the full body frame so a pitched/falling
  torso cannot convert vertical velocity into "forward" credit.
- Yaw-rate penalty (w=0.5) kept; it now reads a correct body-frame z
  angular velocity.
- Observation is still 23-dim, but components 0-7 change meaning. Old
  checkpoints are **not** compatible. Fresh run: `runs/sac_pilot_v3/`.

**Verification:**
- Offline (numpy, mock robot): heading-frame forward velocity = 0.2 m/s
  for a 0.2 m/s walk at robot yaw 0°, 45°, −90°. A pure vertical drop
  while pitched 30° gives 0 forward. A finite-difference fallback
  1° roll per step → 0.524 rad/s, as expected.
- In Isaac Sim (`verify_quat_fix.py`): **PASS**.
  - Spawn: heading −180°, pg = (0, 0, −1).
  - Zero-residual gait: heading-frame forward correlates **+0.967**
    with the validated −v_world_x signal; mean +0.086 m/s.
  - Teleported 20° tilts: roll → pg = (0, −0.342, −0.94);
    pitch → (0.342, 0, −0.94); yaw → (0, 0, −1).

**Observation worth keeping in mind.** In that same zero-residual run,
the robot's heading drifted **+115.7°** in 120 steps before falling. The
raw reference gait turns strongly in this simulator. The old buggy
orientation term was implicitly penalizing that drift; the new reward
only penalizes yaw *rate*, and forward credit is heading-relative. So
the policy can in principle earn reward while walking in a slow circle.
If v3 shows that, add a heading-deviation penalty (e.g. on
`1 - cos(yaw - yaw_spawn)`).

## Looped reference gait (v3) — 2026-11-18

**Problem.** `WebotsReferenceGait.get_targets()` played the 2.6 s Webots
`Forwards.motion` clip **once**, then held its last frame for the rest of
the 50 s episode (first noted in `test_27/NOTES.md`). Only the phase
signal (sin/cos) kept cycling. After 2.6 s, every step therefore had to
come from the ±0.25 rad (≈14°) residuals around a frozen pose. The policy
was also given a clock saying "mid-stride" while the reference stood still.
Report §4.3 describes a cycling gait, so the code did not match the report.

**Change.** `webots_reference_gait.py` gains `loop=True` (the default, and
explicit in the env). Targets come from the `test_27` looped clip
(`experiments/test_27_gait_loop/data/looped_*.npy`: back-to-back copies
of the same clip, one every 2.64 s = 2.60 s + one 40 ms frame step). It
is wrapped in time with exactly the logic of `test_28`'s PGRL `LoopedGait`.
SAC and PGRL now follow the **identical base gait**. The phase period is
2.64 s, so the phase signal stays in sync with the targets. The loader
refuses a looped file that doesn't start with the shared clip.
- **Known seam:** at each copy boundary, the hip-roll / ankle-roll targets
  jump ≈3.3°. It is deliberately not smoothed, to stay identical to PGRL.
- **First 2.6 s unchanged:** the open-loop gait still falls at about 2 s,
  so SAC must still learn to correct during the first stride.

**Verification:**
- Offline: the SAC targets equal an independent re-implementation of the
  PGRL wrap logic (max diff 0.00000° over 48 s). The first 2.6 s are
  identical to the old held version. The phase is 0 at t = 0 and 2.64 s,
  and 0.5 at 1.32 s.
- In Isaac Sim (`verify_gait_loop.py`, full 1500-step episode, zero
  actions, fall termination disabled): **PASS**.
  - L hip-pitch target range after t = 40 s: 37.6° (a held gait gives 0).
  - Targets repeat exactly every 5 periods (396 steps; max diff 0.0000°).
  - 18 phase wraps, in sync.
  - No NaN/Inf across all seams.
  - Height < 0.10 m at gait time 2.00 s, 0.415 m forward (`test_27`:
    about 2.0 s, 0.41 m).

**Caveat for analysis.** `sac_pilot_v3` changes two things at once: the
quaternion fix (above) and the looped gait. If v3 walks, the run alone
can't separate their contributions.

**Evaluation results removed.** `test_evaluation_script/results/run_SAC.json`
and `episodes_SAC.csv` evaluated the **v2 model** (4,995,000 steps) with
the **v3 env file** (sha256 matched commit `f86b6a5`). That is an
observation mismatch, and gave 100% falls and 0 m at every δ, so the
files were deleted and `summary.csv`/`robustness.png` regenerated without
SAC. Only v3 checkpoints are valid with the current env.

## sac_pilot_v3 outcome and warmup action masking (v4) — 2026-11-18

**v3 run (8 h, 0 → 675,000 steps, 45 chunks, no failed/timed-out chunk).**
Training-time mean episode length rose 55 → ~80 steps by ~300k steps,
then plateaued (~78 steps); no episode came close to 1500 (max 142).
Deterministic eval of `final_model.zip`:

| Mode | Steps | Distance | Outcome |
|---|---|---|---|
| GUI (`evaluate_policy.py --slow`, 10 eps) | 51 (all identical) | 0.607 m | fell, h = 0.066 m |
| Headless (action-logging diagnostic, 1 ep) | 105 | — | fell after gait start |

The GUI vs headless difference (same model, same spawn) matches the
discrepancy already noted for PGRL; training is headless, so headless is
the representative number. Visually (GUI): the robot attempts steps, but
one leg moves more than the other and it tips to one side.

**Action-usage diagnostic** (1 deterministic + 5 stochastic episodes,
headless). Residual range is ±0.25 rad (±14.3°) per leg joint; joint
limits clipped the target in only 0–2 % of steps; overall saturation
(|a| > 0.9) was 9 % (deterministic) / 15 % (stochastic). But the policy
holds **constant offsets** on a few joints rather than making corrections:
LHipRoll −7.2°, RHipPitch −7.4°, LHipYawPitch +4.3° (mean), with |a| > 0.5
on 70–90 % of steps — an asymmetric gait by construction. Roll was already
+5–8° during the 60-step **warmup stand** (before the reference gait
starts), i.e. the policy uses the warmup to lean the robot. Also found:
RHipYawPitch is a URDF mimic of LHipYawPitch (actual R follows L within
0.6°, up to 22° off its own command), so that action output has no effect.

**Change (v4):** `NaoWalkingEnv(warmup_actions=False)` (new default) zeroes
the residual action during warmup, so the robot holds the home pose exactly
until the gait starts. The policy still observes and outputs actions there;
only their effect is removed. Nothing else changed (action scale, reward,
grace period, obs). Fresh run `runs/sac_pilot_v4/` (8 h); v3 left
untouched. Note that v3 models evaluated with the v4 env behave slightly
differently during warmup (their warmup actions are now ignored).

## Next steps

- **Superseded by the quaternion fix above:** the v2 body-frame revision
  did not do what it intended. Train v3 (`runs/sac_pilot_v3/`) and check
  the deterministic eval for real forward walking, and for slow circling
  (heading drift).
- Launch a real training run under the new reward (`runs/sac_pilot_v3/`)
  and watch for the same two things that mattered before: whether
  `ep_len_mean`/`ep_rew_mean` climb the way they did originally, and
  whether the *deterministic* checkpoint evaluation shows an actual
  forward-walking gait rather than another degenerate local optimum —
  don't trust training-time aggregate stats alone this time, given what
  happened at 4.45M/7.06M steps.
- Watch `critic_loss` — it had an unexplained-but-self-resolving uptick
  in the original run; worth checking it behaves similarly here rather
  than assuming the pattern repeats.
- Fix `evaluate_policy.py`'s stdout-capture gap (add `--log-file`/
  `TeeOutput`) so future evaluations don't require digging through Kit
  logs.
- Fix the `Monitor` override bug in the original PPO script too, so
  future PPO runs don't lose the same episode history SAC's did.
- Decide whether a dedicated same-step PPO comparison run is worth the
  wall-clock cost for the academic report, or whether SAC's own
  learning curve (under the revised reward) is sufficient on its own.
