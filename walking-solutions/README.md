# Walking Solutions

This is the **working/promoted tier** — validated results meant to be
built on, as opposed to `experiments/`, which is the archival history of
how the project got here (see `docs/experiment-history.md`).

Everything under here is expected to actually run, referencing the shared
assets below rather than duplicating them per algorithm/experiment.

## Layout

```
walking-solutions/
├── shared/                # assets common to every algorithm — see below
└── <algorithm>/            # one folder per RL algorithm, e.g. ppo/
    └── <variant_name>/     # a specific trained/training result for that algorithm
```

## `shared/`

- `shared/robot/` — the Isaac-Sim-ready NAO package (URDF + meshes).
  See `docs/environment-setup/03-nao-robot-integration.md` for where this
  comes from and how it was assembled; the upstream source repos it's
  derived from are **not** committed here.
- `shared/materials/` — texture assets. Note: as of `baseline_test24`,
  nothing in the code actually references these — carried forward for
  now in case they're used manually inside the Isaac Sim UI.
- `shared/reference_gaits/` — captured Webots motion-capture keyframes
  (currently just the one `forwards` gait) used as an action-space
  scaffold by the PPO baseline. Regeneratable via
  `nao_walking_project/webots/extract_forward_motion.py` if needed.

Every algorithm/variant folder references these via `../../shared/...`
relative paths — see any `<algorithm>/<variant>/README.md` for specifics.

## Adding a second RL algorithm

This tier exists specifically to support comparing multiple algorithms on
the same walking task (see `CLAUDE.md` phase 6). To add one:

1. Create `walking-solutions/<algorithm_name>/<variant_name>/`.
2. Point its environment/training code at `../../shared/robot/nao.urdf`
   and, if it uses the reference gait, `../../shared/reference_gaits/`.
3. Add a `README.md` in that folder documenting its config and results,
   following the pattern in `ppo/baseline_test24/README.md`.

## Current contents

- [`ppo/baseline_test24/`](ppo/baseline_test24/README.md) — the current
  working baseline (~20.9M/50M target steps), promoted from
  `experiments/test_24`.
