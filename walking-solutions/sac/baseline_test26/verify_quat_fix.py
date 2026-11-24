"""
One-off verification script (not part of the training pipeline):
checks the quaternion convention fix INSIDE Isaac Sim before trusting it
in training. Supersedes verify_bodyframe_sign.py, whose -0.97 result was a
symptom of the (x,y,z,w) misreading, not a real sign convention.

Checks:
  1. At spawn: heading = 180° (robot faces world -x), projected gravity
     ~ (0, 0, -1), termination |w| ~ 1.
  2. Zero-residual reference gait: heading-frame forward velocity is
     positive on average and tracks the validated world signal -v[0]
     with correlation ~ +1 (not -1), while heading stays ~ constant.
  3. Deliberate tilt: teleporting the robot with a 20° roll / 20° pitch /
     20° yaw shows roll in pg[1], pitch in pg[0], and yaw NOT in pg
     (the old code showed the exact opposite for roll and yaw).

Run from walking-solutions/sac/baseline_test26/:
    python verify_quat_fix.py
"""
import math
import numpy as np
from nao_walking_env import NaoWalkingEnv


def qmul(a, b):
    """Hamilton product of two (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def axis_angle(axis, deg):
    half = math.radians(deg) / 2
    return np.r_[math.cos(half), math.sin(half) * np.asarray(axis, float)]


env = NaoWalkingEnv(urdf_path="../../shared/robot/nao.urdf",
                    render_mode="headless")
env.reset()
G = np.array([0.0, 0.0, -1.0])
ok = True

# ── 1. Spawn ────────────────────────────────────────────────────────────
pos, quat = env._nao.get_world_pose()
yaw0 = math.degrees(env._yaw_from_quat(quat))
pg0 = env._quat_rotate_inverse(quat, G)
w_rel = abs(float(np.dot(quat, env.SPAWN_QUAT_WXYZ)))
print(f"[1] spawn quat (w,x,y,z) = {np.round(quat, 4)}")
print(f"    heading = {yaw0:+.1f} deg (expect ±180), pg = {np.round(pg0, 3)}, |w_rel| = {w_rel:.3f}")
ok &= abs(abs(yaw0) - 180) < 10 and pg0[2] < -0.95 and w_rel > 0.95

# ── 2. Zero-residual gait ───────────────────────────────────────────────
fwd, world_fwd, yaws = [], [], []
zero = np.zeros(env.n_legs, dtype=np.float32)
for _ in range(200):
    obs, r, term, trunc, _ = env.step(zero)
    _, quat = env._nao.get_world_pose()
    v = env._nao.get_linear_velocity()
    fwd.append(float(env._get_heading_velocity(quat)[0]))
    world_fwd.append(-float(v[0]))
    yaws.append(math.degrees(env._yaw_from_quat(quat)))
    if term or trunc:
        break
fwd, world_fwd = np.array(fwd), np.array(world_fwd)
yaws = np.unwrap(np.radians(yaws))
corr = np.corrcoef(fwd, world_fwd)[0, 1]
print(f"[2] {len(fwd)} gait steps: mean heading-frame forward = {fwd.mean():+.4f} m/s, "
      f"corr with -v_world_x = {corr:+.3f} (expect ~+1), "
      f"heading drift = {math.degrees(yaws[-1] - yaws[0]):+.1f} deg")
ok &= corr > 0.9

# ── 3. Deliberate tilts (static teleport, no stepping) ──────────────────
for name, axis, idx in [("roll", [1, 0, 0], 1), ("pitch", [0, 1, 0], 0), ("yaw", [0, 0, 1], None)]:
    env.reset()
    q_target = qmul(env.SPAWN_QUAT_WXYZ, axis_angle(axis, 20))
    env._nao.set_world_pose(position=np.array([0.0, 0.0, env.spawn_height]), orientation=q_target)
    _, quat = env._nao.get_world_pose()
    pg = env._quat_rotate_inverse(quat, G)
    print(f"[3] {name:5s} 20 deg: pg = {np.round(pg, 3)}")
    if idx is None:
        ok &= abs(pg[0]) < 0.05 and abs(pg[1]) < 0.05      # yaw must NOT look like tilt
    else:
        ok &= abs(abs(pg[idx]) - math.sin(math.radians(20))) < 0.05

print("\nRESULT:", "PASS" if ok else "FAIL — do not train on this env until resolved")
env.close()
