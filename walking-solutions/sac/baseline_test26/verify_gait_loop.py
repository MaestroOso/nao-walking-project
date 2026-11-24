"""
One-off verification script (not part of the training pipeline):
checks the looped reference gait INSIDE Isaac Sim before training on it.

Runs one full 1500-step episode with ZERO residual actions and the fall
termination disabled (term_height=-1, term_orient_qw=0), so the episode
keeps going after the robot falls and we can confirm that:
  1. the commanded leg targets keep cycling with the 2.64 s period for the
     whole episode (not held at the last frame after 2.6 s);
  2. the phase signal stays in sync with the targets;
  3. the simulation stays finite (no NaN) across all ~18 seam crossings;
  4. the open-loop outcome matches test_27 (robot falls forward at about
     2 s of gait time, since the first cycle is unchanged).

Run from walking-solutions/sac/baseline_test26/:
    python -u verify_gait_loop.py
"""
import numpy as np
from nao_walking_env import NaoWalkingEnv

env = NaoWalkingEnv(urdf_path="../../shared/robot/nao.urdf",
                    render_mode="headless", term_height=-1.0, term_orient_qw=0.0)
env.reset()
gait = env._gait
zero = np.zeros(env.n_legs, dtype=np.float32)

targets, phases, heights, gait_times = [], [], [], []
start_x = float(env._nao.get_world_pose()[0][0])
fell_at = None
nan_seen = False
for _ in range(env.max_episode_steps):
    obs, r, term, trunc, _ = env.step(zero)
    if env._step_count >= env.gait_warmup_steps:
        t = (env._step_count - env.gait_warmup_steps) * env.control_dt
        targets.append(gait.get_targets(env._phase, t_since_warmup=t))
        phases.append(env._phase)
        gait_times.append(t)
        pos, _ = env._nao.get_world_pose()
        heights.append(float(pos[2]))
        if fell_at is None and pos[2] < 0.10:
            fell_at = (t, -(float(pos[0]) - start_x))
    nan_seen |= bool(np.any(~np.isfinite(obs))) or not np.isfinite(r)
    if trunc:
        break

targets = np.degrees(np.array(targets))
gait_times = np.array(gait_times)
ok = True

# 1. Cycling: the hip-pitch target must keep varying late in the episode,
#    and repeat itself exactly every 5 periods. One period (2.64 s) is
#    79.2 control steps — not an integer, so samples one period apart are
#    up to 1/30 s out of phase. Five periods = 13.2 s = exactly 396 steps.
late = targets[gait_times > 40.0]
late_range = late[:, 2].max() - late[:, 2].min()
held_tail = gait.loop is False
period_steps = gait._period / env.control_dt
lag = int(round(5 * period_steps))                       # 396
repeat_err = np.abs(targets[lag:] - targets[:-lag]).max()
print(f"[1] gait loop={gait.loop}, period={gait._period:.3f}s ({period_steps:.1f} steps)")
print(f"    L hip-pitch target range after t=40s: {late_range:.1f} deg (held gait would be 0)")
print(f"    target(t) vs target(t + 5 periods = {lag} steps), whole episode: "
      f"max diff {repeat_err:.4f} deg")
ok &= (not held_tail) and late_range > 10.0 and repeat_err < 0.01

# 2. Phase in sync: phase ~0 at every multiple of the period
wrap_steps = np.where(np.diff(np.array(phases)) < -0.5)[0]
wrap_times = gait_times[wrap_steps + 1]
print(f"[2] phase wraps: {len(wrap_times)} (first few at {np.round(wrap_times[:4], 3).tolist()} s)")
ok &= len(wrap_times) >= 15 and abs(wrap_times[0] - gait._period) < 0.05

# 3. Numerical stability
print(f"[3] steps run: {env._step_count}, any NaN/Inf in obs/reward: {nan_seen}")
ok &= (not nan_seen) and env._step_count == env.max_episode_steps

# 4. Open-loop outcome vs test_27
if fell_at:
    print(f"[4] height < 0.10 m at gait time {fell_at[0]:.2f} s, forward {fell_at[1]:.3f} m "
          f"(test_27: fell at ~2.0 s, 0.41 m)")
else:
    print("[4] never dropped below 0.10 m (unexpected vs test_27 — inspect visually)")

print("\nRESULT:", "PASS" if ok else "FAIL — do not train on this env until resolved")
env.close()
