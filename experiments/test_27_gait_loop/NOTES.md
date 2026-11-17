# test_27 — looped Webots gait, open loop

**Focus:** Can the Webots `Forwards.motion` gait (2.6 s, 66 frames) be used
as PGRL's base gait? PPO/SAC only use it for the first 2.6 s:
`WebotsReferenceGait.get_targets()` holds the LAST frame after the clip ends,
so from then on the walking comes from the policy's corrections. PGRL has
no network, so the gait itself has to keep cycling.

**Method:** `build_looped_gait.py` appends 20 copies of the clip back to back
(period 2.64 s = 2.60 s + one 40 ms step, 52.76 s in total) and writes them to
`data/`. The shared gait file is not modified. At each join between copies,
hip roll and ankle roll jump 3.3°; all other joints jump < 0.3°.
`run_open_loop.py` plays it in the unmodified PPO baseline env
(imported read-only, gait object swapped) with **zero actions**.

**Outcome:** the robot falls forward before completing one cycle, so it never
reaches the first join.

| Gait time | Forward | Lateral | Height |
|---|---|---|---|
| 0.0 s | 0.000 m | 0.000 m | 0.308 m |
| 1.0 s | 0.026 m | −0.041 m | 0.306 m |
| 2.0 s | 0.410 m | +0.048 m | 0.083 m (fell) |

About 0.38 of the 0.41 m happens while falling, so it is not walking. It still
scores F_S ≈ 0.04, because distance covered while falling counts as distance.

**Decision:** use this looped gait as PGRL's base gait anyway (test_28), so
all three algorithms share the same reference gait and the failure can be
analysed.

**Gotcha:** `SimulationApp.close()` hard-exits the process without flushing
Python's stdout buffer. Use `python -u` or line buffering, and write results
to disk before `env.close()`.

```bash
source ../../nao_walking_project/isaac_env/bin/activate
python build_looped_gait.py
python run_open_loop.py                 # GUI; --headless for numbers only
```
