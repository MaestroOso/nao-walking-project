# test_18_ds_gait

**Focus:** Reverted to the simpler V1 gait and fixed a second
reward-hacking exploit (distinct from `test_12`'s).

**What we found:** The agent learned to dive forward and slide along the
ground, collecting ~470 units of spurious reward before the episode
terminated — "falling forward" was, again, cheaper to reward-maximize
than walking.

**Fix:**
- Gated the forward-velocity reward on the robot staying upright (no
  reward for forward motion while falling).
- Added a height-drop penalty.
- Tightened the fall-termination thresholds so a dive ends the episode
  sooner.

**Outcome:** Completed a 327k-step run.

**Key learning:** This is the same underlying failure mode as `test_12`
(reward hacking) in a new form — gating/penalty terms need to be
re-examined every time the gait or reward structure changes
meaningfully, since a fix for one exploit doesn't prevent a different
one.

**Note on raw data:** `verify19.log` from the original run is not
included in this snapshot; the finding is captured above and in
`docs/experiment-history.md`.
