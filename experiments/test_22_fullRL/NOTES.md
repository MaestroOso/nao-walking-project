# test_22_fullRL

**Focus:** An architectural bet. `test_20`/`test_21` had switched from a
hand-designed gait to literal playback of real Webots-captured walking
keyframes (`Forwards.motion`). This experiment drops behavior-cloning
(BC) pretraining entirely and trains PPO from a random policy, using the
reference gait purely as a scaffold for the **action space** — the
policy's action is a residual correction on top of the reference
trajectory — rather than as an imitation-learning target.

**What we did:** Removed `pretrain.py`'s role in seeding the policy;
trained PPO directly against the reward, with the Webots reference
trajectory only shaping what "zero action" means at each timestep.

**Outcome:** Completed a full 5M-step run successfully, validating the
bet — this became the architecture carried forward into `test_23` and
`test_24`.

**Key learning:** Literal motion-capture replay outperformed
hand-authored gaits, and imitation pretraining wasn't necessary once the
reference gait was used as an action-space scaffold rather than a reward
target — RL alone could still learn effectively with that scaffold in
place.

**Note on shared assets:** This experiment depends on `nao_gait_data/`
(the Webots forward-motion keyframes, via `webots_reference_gait.py`).
That data is byte-identical across `test_20`–`test_24` and is not
duplicated in this archival snapshot — it will live as a shared asset
under `walking-solutions/shared/reference_gaits/` once the working tier
is set up, or can be regenerated via `webots/extract_forward_motion.py`.
