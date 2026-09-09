"""
debug_wrapped.py — Debug with exact same wrappers as training
==============================================================
Tests the environment with NaNSafeWrapper + Monitor, exactly
like train_ppo.py sets it up, to find why episodes are 1 step.
"""

import numpy as np
import gymnasium as gym
from nao_walking_env import NaoWalkingEnv


class NaNSafeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._nan_episodes = 0
        self._total_episodes = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            self._nan_episodes += 1
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = 0.0
            terminated = True
            info["nan_termination"] = True
            print(f"    [NaN CAUGHT] step={self.env._step_count}")
        if np.isnan(reward) or np.isinf(reward):
            print(f"    [NaN REWARD] step={self.env._step_count}, reward was {reward}")
            reward = 0.0
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._total_episodes += 1
        obs, info = self.env.reset(**kwargs)
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print(f"    [NaN ON RESET]")
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, info


print("Creating environment...")
env = NaoWalkingEnv(
    urdf_path="../../robots/nao_isaac_sim/nao.urdf",
    render_mode="headless",
)
env = NaNSafeWrapper(env)

print("\n" + "=" * 70)
print("Test: Running 100 steps with random actions (wrapped env)")
print("=" * 70)

obs, info = env.reset()
print(f"Reset OK. Obs shape: {obs.shape}, NaN: {np.any(np.isnan(obs))}")

total_steps = 0
episode = 0

for i in range(100):
    action = np.random.uniform(-1, 1, size=12).astype(np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    total_steps += 1

    if i < 10 or terminated or truncated:
        base_env = env.env  # unwrap one level
        pos, quat = base_env._nao.get_world_pose()
        h = pos[2] if pos is not None else -1
        sc = base_env._step_count
        print(f"  step={i+1:>3} env_step={sc:>3} h={h:.4f} r={reward:.3f} "
              f"term={terminated} trunc={truncated} "
              f"obs_nan={np.any(np.isnan(obs))}")

    if terminated or truncated:
        episode += 1
        print(f"  → Episode {episode} ended after {env.env._step_count} env steps "
              f"(total loop steps: {total_steps})")
        obs, info = env.reset()
        print(f"  → Reset OK. Obs NaN: {np.any(np.isnan(obs))}")

print(f"\nSummary: {total_steps} steps, {episode} episodes, "
      f"avg len = {total_steps/max(episode,1):.1f}")
print(f"NaN episodes: {env._nan_episodes}/{env._total_episodes}")

env.close()
print("Done.")
