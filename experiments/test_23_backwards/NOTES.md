# test_23_backwards

**Focus:** Found and fixed a one-line reward-sign bug.

**What we found:** Forward velocity was being credited with the
**opposite sign** from the robot's actual heading — the reward term used
`v[0]` where it needed `-v[0]`.

**Fix:** A single-character change:

```diff
- forward_vel = float(v[0])
+ forward_vel = -float(v[0])
```

**Outcome:** Completed a full 5M-step run, continuing from `test_22`'s
architecture.

**Key learning:** An outsized effect from a tiny change — this single
sign flip determined which direction the agent was actually being
trained to walk in. Never assume a reward term's sign convention matches
intuition; verify it empirically (as `test_17` also had to, for a
different frame-related bug).

**Relationship to `test_22`:** This experiment's code is otherwise a
near-exact copy of `test_22` — the entire functional diff is the one line
above. It's kept as a full folder here (rather than a diff-only note)
per this repo's migration policy of preserving each pivotal experiment as
a runnable snapshot.

**Note on shared assets:** Same as `test_22` — `nao_gait_data/` is not
duplicated in this snapshot; see `walking-solutions/shared/reference_gaits/`
(once set up) or regenerate via `webots/extract_forward_motion.py`.
