"""
debug_env.py — Quick diagnostic for test_10 environment
=========================================================
Runs the environment for a few steps with random actions and prints
everything to find why episodes are 1 step long.
"""

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

from nao_walking_env import NaoWalkingEnv

env = NaoWalkingEnv(urdf_path=args.urdf, render_mode="headless" if args.headless else "human")

print("\n" + "=" * 70)
print("DEBUG: Testing environment step-by-step")
print("=" * 70)

# Reset
obs, info = env.reset()
print(f"\nReset obs shape: {obs.shape}")
print(f"Reset obs: {obs}")
print(f"Any NaN in obs: {np.any(np.isnan(obs))}")
print(f"Any Inf in obs: {np.any(np.isinf(obs))}")
print(f"Obs range: [{obs.min():.4f}, {obs.max():.4f}]")

# Check initial state
pos, quat = env._nao.get_world_pose()
print(f"\nInitial position: {pos}")
print(f"Initial quaternion: {quat}")
print(f"Initial height: {pos[2]:.4f}")

lin_vel = env._nao.get_linear_velocity()
ang_vel = env._nao.get_angular_velocity()
print(f"Initial linear velocity: {lin_vel}")
print(f"Initial angular velocity: {ang_vel}")

# Run 20 steps with zero action (stay at home)
print(f"\n{'Step':>4} {'Height':>8} {'Obs NaN':>8} {'Obs Inf':>8} "
      f"{'Reward':>8} {'Term':>6} {'Trunc':>6} {'vel_x':>8} {'vel_y':>8}")
print("-" * 80)

for step in range(20):
    action = np.zeros(12, dtype=np.float32)  # zero action = stay at home
    obs, reward, terminated, truncated, info = env.step(action)

    has_nan = np.any(np.isnan(obs))
    has_inf = np.any(np.isinf(obs))

    # Get velocity from obs (indices 6:8)
    vel_x = obs[6]
    vel_y = obs[7]
    height = obs[44]  # base height

    pos, quat = env._nao.get_world_pose()
    actual_height = pos[2] if pos is not None else -1

    print(f"{step+1:>4} {actual_height:>8.4f} {str(has_nan):>8} {str(has_inf):>8} "
          f"{reward:>8.3f} {str(terminated):>6} {str(truncated):>6} "
          f"{vel_x:>8.4f} {vel_y:>8.4f}")

    if terminated or truncated:
        print(f"  → Episode ended at step {step+1}!")
        if has_nan:
            print(f"  → NaN detected in observation!")
            # Print which indices have NaN
            nan_indices = np.where(np.isnan(obs))[0]
            print(f"  → NaN at indices: {nan_indices}")
        break

# Run 5 steps with random actions
print(f"\n{'Step':>4} {'Height':>8} {'Obs NaN':>8} {'Reward':>8} {'Term':>6} "
      f"{'vel_x':>8} {'Action range':>15}")
print("-" * 70)

obs, info = env.reset()
for step in range(5):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)

    has_nan = np.any(np.isnan(obs))
    pos, _ = env._nao.get_world_pose()
    actual_height = pos[2] if pos is not None else -1
    vel_x = obs[6]

    print(f"{step+1:>4} {actual_height:>8.4f} {str(has_nan):>8} {reward:>8.3f} "
          f"{str(terminated):>6} {vel_x:>8.4f} "
          f"[{action.min():.2f}, {action.max():.2f}]")

    if terminated or truncated:
        print(f"  → Episode ended at step {step+1}!")
        break

env.close()
print("\nDone.")
