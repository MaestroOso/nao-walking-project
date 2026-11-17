# test_evaluation_script — common evaluation protocol for all algorithms

One evaluation procedure, used for every walking algorithm (PPO, SAC, and later
PGRL), so the comparison in chapter 5 of the report (Table 5-3, Figure 5-3)
measures every algorithm in the same way. This implements §4.6.2 and §5.1 of
the report.

## Files

| File | Purpose |
|---|---|
| `evaluate_algorithms.py` | Runs the protocol for **one** algorithm in Isaac Sim and writes `results/episodes_<LABEL>.csv` and `results/run_<LABEL>.json`. |
| `metrics.py` | Metric definitions (F_S, fall rate, heading drift, mean/std). No Isaac Sim dependency. |
| `summarize_results.py` | Merges every `episodes_*.csv` (plus external CSVs) into `results/summary.csv` (Table 5-3) and `results/robustness.png` (Figure 5-3). No Isaac Sim needed. |
| `run_all_evaluations.sh` | Runs `evaluate_algorithms.py` for each algorithm (one process each), then `summarize_results.py`. |
| `external_results_template.csv` | Column layout for results recorded outside this script (e.g. the manufacturer's walking routine). |

## Protocol

| Item | Value |
|---|---|
| Course | Straight line of **D** m (`--distance`, default 1.0) along the robot's forward direction (world **−x**) |
| Policy | Deterministic (mean action, no exploration noise) |
| Repetitions | N = 10 per δ (`--repetitions`) |
| Perturbation | δ ∈ {0, 1, 2, 4}° (`--deltas`). Repetition k adds `δ · u_k` to the 12 leg joints, `u_k ~ U[−1,1]¹²` from seed `base_seed + k`. The same `u_k` is used for every δ and every algorithm. Offsets are clipped to the joint limits. |
| Where the offset applies | The pose the PD controller holds during the warm-up (teleport, settle, warm-up targets). At the end of the warm-up the reference gait starts from its own first frame, as in training. |
| Start of measurement | End of the warm-up (the instant the reference gait starts, after control step 59). Distance and time are measured from there. |
| Stop conditions | First of: forward distance ≥ D → `success`; env termination → `fall_height` / `fall_tilt` / `invalid` (NaN); env truncation at 1500 steps = **50 s from the start of the episode** → `timeout` (not a fall). |
| Fall criteria | Exactly the env's own `_check_termination()`: height < 0.10 m, `|quat[3]| < 0.3`, or NaN, after the 50-step grace period. Uses the project's existing quaternion convention. |

The training reward is **not** used. PPO's env and the revised SAC env compute
different rewards, so every metric comes from the robot's physical state.

## Metrics (`metrics.py`)

- **F_S** (report Eq. 4-5): `D / t` if the course is completed, otherwise
  `(d/t) · 0.5 · (|d|/D)`. Using `|d|` keeps backwards walking **negative**.
  For d ≥ 0 this is identical to the report's formula. When the course is
  completed, `t` is interpolated within the last control step to the exact
  moment d = D. A run that falls during the warm-up has t = 0 and F_S = 0.
- **TC** (Eq. 5-1): `100 · falls / N`. `invalid` (NaN state) counts as a fall;
  `timeout` does not.
- **Auxiliary:** walking time, forward distance `d` (may be negative),
  `lateral_left_m` (final displacement to the robot's left = world −y), and
  `heading_drift_deg` = direction of travel relative to forward (positive =
  left). Heading drift is computed from the displacement, not from the
  quaternion. Means use sample std (ddof = 1).

## Usage

Inside the Isaac Sim Python environment:

```bash
cd experiments/test_evaluation_script
source ../../nao_walking_project/isaac_env/bin/activate

# Everything (PPO + SAC, D = 1 m, δ = 0 1 2 4, N = 10), then summary + figure
./run_all_evaluations.sh
DISTANCE=1.5 ./run_all_evaluations.sh --overwrite

# A single algorithm / custom checkpoint
python evaluate_algorithms.py --algo sac --distance 1.0
python evaluate_algorithms.py --algo sac --model ../test_26_sac/runs/sac_pilot_v2/ckpt_4995000_steps.zip --label SAC_4.995M
python evaluate_algorithms.py --algo ppo --deltas 0 --repetitions 3 --gui    # watch it

# Rebuild table/figure only (no Isaac Sim needed)
python summarize_results.py
python summarize_results.py --extra manufacturer_results.csv
```

Default models (latest version of each algorithm, overridable with `--model` / `--env-dir`):

| `--algo` | Model | Environment (`nao_walking_env.py` folder) |
|---|---|---|
| `ppo` | `~/nao_walking_project/experiments/test_24/runs/test24/final_model.zip` (27.2M steps, outside the repo; see "PPO model choice") | `walking-solutions/ppo/baseline_test24/` |
| `sac` | `experiments/test_26_sac/runs/sac_pilot_v2/final_model.zip` (still training — re-run when it finishes) | `experiments/test_26_sac/` |
| `pgrl` | not implemented yet | — |

The script changes into the env folder before creating the env, because the
env files resolve the URDF and reference-gait paths relative to the working
directory. It also checks that the model's observation/action spaces match the env.

Results for an existing label are not overwritten unless `--overwrite` is
given. Use `--label` to keep several checkpoints of the same algorithm.

## Outputs (`results/`)

- `episodes_<LABEL>.csv`: one row per repetition (label, δ, repetition, seed,
  D, outcome, success, fell, fell_during_warmup, steps, walk_time_s,
  distance_m, lateral_left_m, heading_drift_deg, mean_speed_mps, fs,
  max_abs_offset_deg, wall_time_s). Written row by row, so a crash keeps
  the finished repetitions.
- `run_<LABEL>.json`: provenance: command, protocol, env parameters, model
  path + SHA-256 + `num_timesteps`, env file SHA-256, git commit and dirty flag.
- `summary.csv`: per (configuration, D, δ): N, successes, falls, TC, and
  mean/std of F_S, time, distance, lateral deviation and heading drift.
- `robustness.png`: TC and F_S (mean ± std) against δ, for configurations
  evaluated at two or more δ values.

## Results recorded elsewhere (manufacturer's routine)

Fill a CSV with the columns of `external_results_template.csv`, one row per
repetition. `success` and `fell` are `True`/`False`, and `walk_time_s` /
`distance_m` are measured as in the protocol above. Use `delta_deg = 0`,
because the routine can't be started from a perturbed pose. Then run
`python summarize_results.py --extra <file>.csv`. F_S is recomputed from these
columns with the same formula, so it is directly comparable.

## PPO model choice

There are two copies of the PPO `test_24` run (`runs/test24/`). Neither is
inside `experiments/test_24/`, which holds only the archived code (checkpoints
are gitignored):

| Option | Folder | Steps (`progress.txt`) | Last update |
|---|---|---|---|
| **A (default)** | `~/nao_walking_project/experiments/test_24/runs/test24/` | 27.2M | 17 Aug 2026 |
| B | `<repo>/nao_walking_project/experiments/test_24/runs/test24/` | 22.35M | 8 Sep 2026 |

Both models report `num_timesteps ≈ 53k`, because `train_ppo.py` resets
`num_timesteps` to 0 on every chunk resume. Use `progress.txt` for the true
step count. The environment code in `experiments/test_24/` is identical to
`walking-solutions/ppo/baseline_test24/` except for asset paths, so the
baseline env is used for evaluation.

## First results (17 Nov 2026, PPO only, D = 1 m, N = 10)

These were the first full runs of the script. They are kept locally in
`results/` (not committed) as `episodes_PPO.csv` (option B) and
`episodes_PPO_A_27.2M.csv` (option A).

| δ | F_S B | F_S A | Falls B | Falls A |
|---|---|---|---|---|
| 0° | 0.587 | 0.906 | 0 % | 0 % |
| 1° | 0.509 ± 0.160 | 0.544 ± 0.193 | 0 % | 20 % |
| 2° | 0.399 ± 0.216 | 0.523 ± 0.217 | 30 % | 20 % |
| 4° | 0.536 ± 0.112 | 0.483 ± 0.235 | 0 % | 20 % |

Observations, still to be confirmed:
- Option A is faster from the nominal pose, but it turns about 14° to the
  right over the first metre and falls more often when the starting pose is
  perturbed.
- The speeds are much higher than a real NAO's (about 0.1 m/s). A full 50 s
  trace of option B stayed upright (base height about 0.31 m, no tilt) and
  covered 22.3 m in 48 s, about 0.46 m/s. Watch it with `--gui` to rule out
  shuffling or sliding (an exploit listed in report §4.1) before using
  these numbers in the report.

## Known issues

- Once Isaac Sim starts it takes over stdout, so the per-episode table
  printed by `evaluate_algorithms.py` may not show in the console or in a
  redirected log. The CSV and JSON files are complete; use
  `summarize_results.py` to see the results.

## Adding PGRL

`PGRLAdapter` in `evaluate_algorithms.py` is a stub. Its docstring, and the
`PolicyAdapter` contract above it, describe what the controller and its env
must provide. Once it's implemented, add `pgrl` to `ALGOS` in
`run_all_evaluations.sh`. Evaluate the base gait without the compensator with
the same adapter (gains = 0) and a different `--label`, e.g. `PGRL_base`.
