"""
debug_vecenv.py — Test with SB3's DummyVecEnv (replicates model.learn)
========================================================================
This creates the exact same wrapper chain as train_ppo.py and runs
the rollout collection loop manually to find the 1-step episode bug.
"""

import numpy as np
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
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
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._total_episodes += 1
        obs, info = self.env.reset(**kwargs)
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, info


# Create env exactly like train_ppo.py
print("Creating environment with full wrapper chain...")
raw_env = NaoWalkingEnv(
    urdf_path="../../robots/nao_isaac_sim/nao.urdf",
    render_mode="headless",
)
wrapped_env = NaNSafeWrapper(raw_env)
monitored_env = Monitor(wrapped_env, filename=None)

print(f"Env obs space: {monitored_env.observation_space.shape}")
print(f"Env act space: {monitored_env.action_space.shape}")

# Test 1: Manual stepping with the monitored environment
print("\n" + "=" * 70)
print("TEST 1: Manual stepping with Monitor wrapper")
print("=" * 70)

obs, info = monitored_env.reset()
print(f"Reset obs shape: {obs.shape}, NaN: {np.any(np.isnan(obs))}")

for i in range(55):
    action = np.random.uniform(-1, 1, 12).astype(np.float32)
    obs, reward, terminated, truncated, info = monitored_env.step(action)
    
    if i < 3 or terminated or truncated:
        print(f"  step {i+1}: r={reward:.3f} term={terminated} trunc={truncated} "
              f"info={info}")
    
    if terminated or truncated:
        print(f"  → Ended at step {i+1}")
        break

# Test 2: Create PPO and do a single rollout collection
print("\n" + "=" * 70)
print("TEST 2: PPO model.collect_rollouts (what model.learn does)")
print("=" * 70)

# Re-create env for PPO
raw_env2 = NaoWalkingEnv(
    urdf_path="../../robots/nao_isaac_sim/nao.urdf",
    render_mode="headless",
)
wrapped_env2 = NaNSafeWrapper(raw_env2)
monitored_env2 = Monitor(wrapped_env2, filename=None)

model = PPO(
    "MlpPolicy",
    monitored_env2,
    policy_kwargs={"net_arch": [128, 128]},
    learning_rate=3e-4,
    n_steps=256,  # small buffer for quick test
    batch_size=64,
    n_epochs=1,
    device="cpu",
    seed=42,
    verbose=0,
)

print(f"PPO created. Collecting 256 steps of rollout data...")
print(f"Before collect: last_obs = {model._last_obs}")
print(f"Before collect: last_obs shape = {model._last_obs.shape if model._last_obs is not None else None}")

# Manually call the rollout collection
from stable_baselines3.common.callbacks import BaseCallback

class DebugCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.step_count = 0
        self.episode_ends = 0
    
    def _on_step(self):
        self.step_count += 1
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_ends += 1
                ep_r = info["episode"]["r"]
                ep_l = info["episode"]["l"]
                print(f"    [Episode {self.episode_ends}] len={ep_l} reward={ep_r:.2f}")
        
        # Print first few steps
        if self.step_count <= 5:
            dones = self.locals.get("dones", [])
            new_obs = self.locals.get("new_obs", [None])
            rewards = self.locals.get("rewards", [None])
            print(f"  rollout step {self.step_count}: "
                  f"done={dones} reward={rewards}")
        
        return True

debug_cb = DebugCallback()

try:
    model.learn(total_timesteps=256, callback=debug_cb)
except Exception as e:
    print(f"Error during learn: {e}")
    import traceback
    traceback.print_exc()

print(f"\nRollout result: {debug_cb.step_count} steps, "
      f"{debug_cb.episode_ends} episode ends")
print(f"Expected: ~5 episodes (256 steps / ~51 steps per episode)")
print(f"If >> 5 episodes: something is causing early termination during learn()")

# Cleanup
monitored_env.close()
monitored_env2.close()
print("\nDone.")
