"""
evaluate_policy.py — Evaluate trained NAO walking policy
==========================================================
test_10: Includes velocity obs verification and walking metrics

USAGE:
  python evaluate_policy.py --model runs/test10_walking/final_model.zip --headless
  python evaluate_policy.py --model runs/test10_walking/model_250000.zip --headless --episodes 5
"""

import argparse
import numpy as np
import time

parser = argparse.ArgumentParser(description="Evaluate trained NAO walking policy")
parser.add_argument("--model", type=str, default="runs/test10_walking/final_model.zip")
parser.add_argument("--urdf", type=str, default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--deterministic", action="store_true", default=True)
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--slow", action="store_true")
args = parser.parse_args()

deterministic = not args.stochastic

print("=" * 70)
print("NAO Walking — Policy Evaluation (test_10)")
print(f"  Model: {args.model}")
print(f"  Episodes: {args.episodes}")
print(f"  Deterministic: {deterministic}")
print(f"  GUI: {not args.headless}")
print("=" * 70)

from stable_baselines3 import PPO
model = PPO.load(args.model)
print(f"Model loaded. Observation space expected: {model.observation_space.shape}")

from nao_walking_env import NaoWalkingEnv
render_mode = "headless" if args.headless else "human"
env = NaoWalkingEnv(urdf_path=args.urdf, render_mode=render_mode)

episode_stats = []

print(f"\n{'Ep':>4} {'Steps':>6} {'Reward':>8} {'R/step':>8} "
      f"{'X dist':>8} {'Avg vel':>8} {'Survived':>10} {'Termination':<15}")
print("-" * 85)

for ep in range(args.episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    step = 0
    start_pos = env._get_base_position().copy()

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated

        if args.slow and not args.headless:
            time.sleep(0.05)

    final_pos = env._get_base_position()
    x_distance = final_pos[0] - start_pos[0]
    survived_grace = step > env.grace_period
    avg_reward = total_reward / max(step, 1)
    avg_vel = x_distance / (step * env.control_dt) if step > 0 else 0.0

    pos, quat = env._nao.get_world_pose()
    if pos is None or np.any(np.isnan(pos)):
        term_reason = "NaN"
    elif step >= env.max_episode_steps:
        term_reason = "max_steps"
    elif pos[2] < env.term_height:
        term_reason = f"fell (h={pos[2]:.3f})"
    elif quat is not None and abs(float(quat[3])) < env.term_orient_qw:
        term_reason = f"tilted (qw={abs(float(quat[3])):.3f})"
    else:
        term_reason = "unknown"

    episode_stats.append({
        "steps": step,
        "reward": total_reward,
        "avg_reward": avg_reward,
        "x_distance": x_distance,
        "avg_vel": avg_vel,
        "survived_grace": survived_grace,
        "term_reason": term_reason,
    })

    survived_str = "YES" if survived_grace else "no"
    print(f"{ep+1:>4} {step:>6} {total_reward:>8.1f} {avg_reward:>8.2f} "
          f"{x_distance:>7.3f}m {avg_vel:>7.3f}m/s {survived_str:>10} {term_reason:<15}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("EVALUATION SUMMARY (test_10)")
print("=" * 85)

steps = [s["steps"] for s in episode_stats]
rewards = [s["reward"] for s in episode_stats]
distances = [s["x_distance"] for s in episode_stats]
velocities = [s["avg_vel"] for s in episode_stats]
survived = sum(1 for s in episode_stats if s["survived_grace"])

print(f"\nEpisodes:            {args.episodes}")
print(f"Survived past grace: {survived}/{args.episodes} "
      f"({100*survived/args.episodes:.0f}%)")

print(f"\nEpisode length:      {np.mean(steps):>7.1f} mean  |  "
      f"{np.min(steps):>4} min  |  {np.max(steps):>4} max  |  "
      f"{np.std(steps):>6.1f} std")
print(f"Total reward:        {np.mean(rewards):>7.1f} mean  |  "
      f"{np.min(rewards):>7.1f} min  |  {np.max(rewards):>7.1f} max")
print(f"Reward per step:     {np.mean([s['avg_reward'] for s in episode_stats]):>7.2f} mean")

print(f"\n── Walking Metrics ──")
print(f"X distance (m):      {np.mean(distances):>7.3f} mean  |  "
      f"{np.min(distances):>7.3f} min  |  {np.max(distances):>7.3f} max")
print(f"Avg velocity (m/s):  {np.mean(velocities):>7.3f} mean  |  "
      f"{np.min(velocities):>7.3f} min  |  {np.max(velocities):>7.3f} max")

any_walking = any(d > 0.05 for d in distances)
good_walking = any(d > 0.5 for d in distances)
great_walking = any(d > 2.0 for d in distances)

if great_walking:
    print(f"\n★ SUSTAINED WALKING — agent covers >2m in at least one episode!")
elif good_walking:
    print(f"\n✓ WALKING DETECTED — agent covers >0.5m")
elif any_walking:
    print(f"\n◐ SOME MOVEMENT — >5cm but not sustained walking")
else:
    print(f"\n✗ NO SIGNIFICANT FORWARD MOTION")

print(f"\nTermination reasons:")
from collections import Counter
reasons = Counter(s["term_reason"].split(" ")[0] for s in episode_stats)
for reason, count in reasons.most_common():
    print(f"  {reason:<20} {count:>3} ({100*count/args.episodes:.0f}%)")

mean_time = np.mean(steps) / 30.0
max_time = np.max(steps) / 30.0
print(f"\nSimulated time:      {mean_time:.1f}s mean  |  {max_time:.1f}s max")

env.close()
print("\nDone.")
