"""
SUPERSEDED by verify_quat_fix.py (2026-11-17). This script's -0.97 result
was a symptom of reading Isaac's (w,x,y,z) quaternion as (x,y,z,w), not a
real sign convention. After the fix it would report ~+1 (and the env no
longer uses v_body[0] directly). Kept as historical record.

One-off verification script (not part of the training pipeline):
confirms the sign of body-frame forward velocity against the known-good
world-frame convention (-v[0], validated by test_23/test_24/the whole
SAC run so far) before trusting it in the reward. Zero residual action
= pure reference-gait replay, which is known to walk in the intended
forward direction under that world-frame convention.
"""
import numpy as np
from nao_walking_env import NaoWalkingEnv

env = NaoWalkingEnv(urdf_path="../../walking-solutions/shared/robot/nao.urdf",
                     render_mode="headless")
obs, _ = env.reset()

world_fwd_signal = []   # -v[0], the already-validated "correct forward is positive" signal
body_fwd_signal = []    # quat_rotate_inverse(quat, v)[0], sign to be determined
x_positions = []

zero_action = np.zeros(env.n_legs, dtype=np.float32)

for i in range(200):
    obs, r, term, trunc, info = env.step(zero_action)
    pos, quat = env._nao.get_world_pose()
    v = env._nao.get_linear_velocity()
    if pos is None or quat is None or v is None:
        continue
    x_positions.append(float(pos[0]))
    world_fwd_signal.append(-float(v[0]))
    v_body = env._quat_rotate_inverse(quat, v)
    body_fwd_signal.append(float(v_body[0]))
    if term or trunc:
        break

x_positions = np.array(x_positions)
world_fwd_signal = np.array(world_fwd_signal)
body_fwd_signal = np.array(body_fwd_signal)

net_world_x_displacement = x_positions[-1] - x_positions[0]
print(f"steps run: {len(x_positions)}")
print(f"net world x displacement: {net_world_x_displacement:+.4f} m "
      f"(negative = moved in the already-validated 'forward' direction)")
print(f"mean world-frame forward signal (-v[0], known-good convention): "
      f"{world_fwd_signal.mean():+.4f}")
print(f"mean body-frame v_body[0]: {body_fwd_signal.mean():+.4f}")
corr = np.corrcoef(world_fwd_signal, body_fwd_signal)[0, 1]
print(f"correlation(world_fwd_signal, body_fwd_signal): {corr:+.4f}")
print()
if corr > 0.5:
    print("=> body-frame v_body[0] has the SAME sign convention as -v[0]: "
          "reward should use forward_vel = +v_body[0]")
elif corr < -0.5:
    print("=> body-frame v_body[0] has the OPPOSITE sign convention: "
          "reward should use forward_vel = -v_body[0]")
else:
    print("=> INCONCLUSIVE (|corr| too small) - investigate further before trusting either sign")

env.close()
