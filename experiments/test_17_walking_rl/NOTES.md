# test_17_walking_rl

**Focus:** Train RL on top of the V4 reference gait from `test_16c`. The
session got derailed fixing a fundamental bug rather than completing a
full training run.

**What we found:** The reward used **world-frame velocity** as "forward,"
which only holds if the robot's local axes align with the world's. PPO
had actually learned to walk in the robot's mechanical *backward*
direction — the policy was working, the reward was measuring the wrong
thing.

**Fix:** A body-frame velocity transform, plus a manual empirical
sign-verification protocol (`verify4.log`–`verify7.log`,
`verify_dsgait.log` in the original run — not migrated, see note below)
to confirm the fix actually pointed the reward the right way.

**Outcome:** Debugging session with short verification runs only — no
full training run completed here. The fix carried forward into later
experiments.

**Key learning:** Coordinate-frame and sign-convention bugs are cheap to
introduce and expensive to catch — a policy can be "working" while being
rewarded for the wrong thing, and this only surfaces through dedicated
verification, not code review. This recurred again in `test_23` (a
reward-sign bug of a different kind).

**Note on raw data:** The `verify*.log` empirical verification logs are
not included in this snapshot; the finding and fix are captured above and
in `docs/experiment-history.md`.
