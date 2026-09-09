"""
debug_learn.py — Test model.learn() with small step count
==========================================================
Single env, single PPO, 256 steps. Prints every episode end.
"""

import numpy as np
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
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


class DebugCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.step_count = 0
        self.episode_ends = 0

    def _on_step(self):
        self.step_count += 1
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        # Print first 10 steps
        if self.step_count <= 10:
            print(f"  step {self.step_count}: done={dones} reward={rewards}")

        for info in infos:
            if "episode" in info:
                self.episode_ends += 1
                ep_r = info["episode"]["r"]
                ep_l = info["episode"]["l"]
                if self.episode_ends <= 20:
                    print(f"    [Episode {self.episode_ends}] "
                          f"len={ep_l} reward={ep_r:.2f}")

        return True


print("Creating environment...")
env = NaoWalkingEnv(
    urdf_path="../../robots/nao_isaac_sim/nao.urdf",
    render_mode="headless",
)
env = NaNSafeWrapper(env)
env = Monitor(env, filename=None)

print(f"Obs space: {env.observation_space.shape}")
print(f"Act space: {env.action_space.shape}")

print("\nCreating PPO model...")
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs={"net_arch": [128, 128]},
    learning_rate=3e-4,
    n_steps=512,  # ~10 episodes worth
    batch_size=64,
    n_epochs=1,
    device="cpu",
    seed=42,
    verbose=1,
)

debug_cb = DebugCallback()

print(f"\nCalling model.learn(512 steps)...")
print(f"Expected: ~10 episodes of ~51 steps each")
print(f"If >> 10 episodes: bug confirmed\n")

model.learn(total_timesteps=512, callback=debug_cb)

print(f"\n{'='*50}")
print(f"Result: {debug_cb.step_count} steps, {debug_cb.episode_ends} episodes")
print(f"Avg episode length: {debug_cb.step_count / max(debug_cb.episode_ends, 1):.1f}")
print(f"NaN episodes: {env.env._nan_episodes}/{env.env._total_episodes}")

env.close()
print("Done.")
