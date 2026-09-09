# test_10_walking

**Focus:** Fix a bug diagnosed in `test_09` (diagnostics-only, not
migrated here — 40-line script confirming the agent had no way to
perceive velocity), and simplify the reward function.

**What we did:** Added the missing velocity terms to the observation
vector — the agent had been rewarded for forward velocity it structurally
could not perceive. Also simplified the reward from ~6 components down to
3, explicitly to reduce gradient noise.

**Outcome:** Longest run to that point in the project (870k steps).

**Key learning:** Several "the reward isn't working" episodes across this
project turned out to be reward *observability* bugs, not reward *design*
bugs — the agent couldn't see the thing the reward depended on. Before
retuning reward weights, verify what's actually in the observation
vector. This lesson recurred later in `test_17` (world-frame vs.
body-frame velocity) in a different form.

**Debug scripts included** (`debug_env.py`, `debug_learn.py`,
`debug_model.py`, `debug_vecenv.py`, `debug_wrapped.py`) trace the
investigation that led to finding the missing-velocity bug.
