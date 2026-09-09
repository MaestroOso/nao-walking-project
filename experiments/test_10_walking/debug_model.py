"""
debug_model.py — Check if the saved model works with the environment
=====================================================================
Loads model_10000.zip and runs it for a few episodes to check
if the model-environment interaction is the problem.
"""

import numpy as np
from stable_baselines3 import PPO
from nao_walking_env import NaoWalkingEnv
import gymnasium as gym


class NaNSafeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = 0.0
            terminated = True
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        return obs, reward, terminated, truncated, info
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, info


import sys
import os

# Try to load the model
model_path = "runs/test10_walking/model_10000.zip"
if not os.path.exists(model_path):
    model_path = "runs/test10_walking/final_model.zip"

print(f"Loading model: {model_path}")
model = PPO.load(model_path, device="cpu")

print(f"Model observation space: {model.observation_space}")
print(f"Model action space: {model.action_space}")

# Create environment
env = NaoWalkingEnv(
    urdf_path="../../robots/nao_isaac_sim/nao.urdf",
    render_mode="headless",
)
env = NaNSafeWrapper(env)

print(f"Env observation space: {env.observation_space}")
print(f"Env action space: {env.action_space}")

# Check shape match
print(f"\nObs space match: {model.observation_space.shape == env.observation_space.shape}")
print(f"Act space match: {model.action_space.shape == env.action_space.shape}")

# Run the model
obs, info = env.reset()
print(f"\nReset obs shape: {obs.shape}")

for step in range(60):
    # Get action from model
    action, _states = model.predict(obs, deterministic=False)
    
    if step < 5:
        print(f"\nStep {step+1}:")
        print(f"  Action shape: {action.shape}")
        print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
        print(f"  Action: {action}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if step < 5 or terminated or truncated:
        pos, _ = env.env._nao.get_world_pose()
        h = pos[2] if pos is not None else -1
        print(f"  → h={h:.4f} r={reward:.3f} term={terminated} trunc={truncated}")
    
    if terminated or truncated:
        print(f"\nEpisode ended at step {step+1}")
        break

env.close()
print("\nDone.")
