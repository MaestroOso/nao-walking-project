# test_16c_rl

**Focus:** Continue iterating the hand-designed reference gait, V3 → V4,
validated **open-loop** (no RL training) for stability and forward
progress before trusting it as a base for training.

**Context:** V1 (`test_15`/`test_16`) was a simple stepping-in-place
pattern with no real forward-progress mechanism. V2 (`test_16b`) added
active-stance push-off and surfaced a clear stability/speed trade-off — a
fast version fell over after ~37cm of progress in 300 steps, while a
stable version crawled at 0.4cm/cycle.

**What we did:** Iterated the gait design through V3 and V4
(`debug_gait_sim.py`, `test_gait_v3_math.py`), checking the resulting
trajectories at each version (`debug_v3_iter*` / `debug_v4_iter*` —
individual iteration logs not migrated, see note below).

**Outcome:** V4 finally achieved **both** stability and forward
progress — +0.55cm/cycle over 6 cycles, stable. This became the gait fed
into RL training in `test_17`, and ultimately (after further revert/fixes)
into the working baseline.

**Key learning:** Validating a reference gait open-loop, before ever
spending RL training time on it, caught the V1/V2 problems cheaply. This
phase (`test_15`–`test_19`) was the most numerically rigorous part of the
project specifically because of this discipline.

**Note on raw data:** The per-iteration debug logs (`debug_v3_iter*.txt`,
`debug_v4_iter*.txt`) are not included in this snapshot; the key measured
numbers are captured above and in `docs/experiment-history.md`.
