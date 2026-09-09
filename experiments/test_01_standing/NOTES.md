# test_01_standing

**Focus:** First attempt at standing via RL — binary reward (1 if upright,
0 otherwise).

**What happened:** Training ran without crashing, but the reward signal
had no learnable gradient — there was no way for the policy to tell
"almost standing" from "completely fallen," so there was nothing to
climb.

**Outcome:** No usable standing behavior. This is the project's starting
point, not a working result.

**Key learning:** A binary/sparse reward is not enough to bootstrap
learning from a random policy — the reward needs to be dense enough to
give a gradient toward the desired behavior from the very first episodes.
This shaped every reward design decision from here on (see `test_10` for
where reward observability itself became the actual bug).

See `docs/experiment-history.md` for full project narrative and how this
connects to the later experiments.
