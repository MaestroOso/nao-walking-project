"""
train_ppo.py — Train NAO to walk using PPO (Stable-Baselines3)
================================================================
test_12: Survival-scaled alive bonus
         Training from scratch

USAGE:
  python train_ppo.py --headless --timesteps 10000  # single chunk
  python train_ppo.py --resume runs/test10_walking/final_model.zip
"""

import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--timesteps", type=int, default=200_000)
parser.add_argument("--checkpoint-freq", type=int, default=50_000)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--run-name", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--ent-coef", type=float, default=0.01)
parser.add_argument("--log-file", type=str, default=None,
                    help="Path to append training logs")
args = parser.parse_args()

if args.run_name is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_name = f"nao_ppo_{timestamp}"

run_dir = os.path.join("runs", args.run_name)
os.makedirs(run_dir, exist_ok=True)

# ── Set up logging to file ────────────────────────────────────────────────────
class TeeOutput:
    """Write to both stdout and a log file."""
    def __init__(self, log_path, stream):
        self.terminal = stream
        self.log_file = open(log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def fileno(self):
        return self.terminal.fileno()

    def close(self):
        self.log_file.close()

if args.log_file:
    tee_out = TeeOutput(args.log_file, sys.stdout)
    tee_err = TeeOutput(args.log_file, sys.stderr)
    sys.stdout = tee_out
    sys.stderr = tee_err

print(f"\n{'='*70}")
print(f"[{datetime.now().isoformat()}]")
print(f"NAO Walking — PPO Training (test_12: survival-scaled alive bonus)")
print(f"  Run: {args.run_name}")
print(f"  Dir: {run_dir}")
print(f"  Timesteps: {args.timesteps:,}")
print(f"  Headless: {args.headless}")
print(f"  Resume: {args.resume}")
print(f"  LR: {args.lr}, Entropy: {args.ent_coef}")
print(f"{'='*70}")

# ── Create environment ────────────────────────────────────────────────────────
from nao_walking_env import NaoWalkingEnv

render_mode = "headless" if args.headless else "human"
env = NaoWalkingEnv(
    urdf_path=args.urdf,
    render_mode=render_mode,
)

# ── NaN safety wrapper ───────────────────────────────────────────────────────
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor


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

    @property
    def nan_rate(self):
        if self._total_episodes == 0:
            return 0.0
        return self._nan_episodes / self._total_episodes


env = NaNSafeWrapper(env)
env = Monitor(env, os.path.join(run_dir, "monitor"))

# ── PPO Setup ─────────────────────────────────────────────────────────────────
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback


class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_freq=10, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._episode_count = 0
        self._episode_rewards = []
        self._episode_lengths = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_count += 1
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self._episode_rewards.append(ep_reward)
                self._episode_lengths.append(ep_length)

                if self._episode_count % self.log_freq == 0:
                    recent_r = self._episode_rewards[-self.log_freq:]
                    recent_l = self._episode_lengths[-self.log_freq:]
                    mean_r = np.mean(recent_r)
                    std_r = np.std(recent_r)
                    mean_l = np.mean(recent_l)
                    max_l = np.max(recent_l)

                    nan_wrapper = self.training_env.envs[0]
                    while hasattr(nan_wrapper, 'env'):
                        if isinstance(nan_wrapper, NaNSafeWrapper):
                            break
                        nan_wrapper = nan_wrapper.env
                    nan_info = ""
                    if isinstance(nan_wrapper, NaNSafeWrapper):
                        nan_info = f"  NaN={nan_wrapper.nan_rate:.1%}"

                    print(f"  [Ep {self._episode_count:>5}] "
                          f"R={mean_r:>7.1f}±{std_r:.1f}  "
                          f"len={mean_l:>5.0f} (max={max_l:>4})  "
                          f"step={self.num_timesteps:>8,}{nan_info}")
        return True


checkpoint_cb = CheckpointCallback(
    save_freq=args.checkpoint_freq,
    save_path=run_dir,
    name_prefix="checkpoint",
)
reward_cb = RewardLoggingCallback(log_freq=10)

if args.resume:
    print(f"Resuming from: {args.resume}")
    model = PPO.load(args.resume, env=env, device="cpu")
    model.learning_rate = args.lr
    model.ent_coef = args.ent_coef
    # Force clean state for new chunk — clear stale obs from previous process
    model._last_obs = None
    model.num_timesteps = 0
    print(f"  Reset model state for new chunk (cleared _last_obs, num_timesteps)")
else:
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={
            "net_arch": [128, 128],
        },
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
        seed=args.seed,
        verbose=1,
        tensorboard_log=os.path.join(run_dir, "tb"),
    )

# ── Save configuration ───────────────────────────────────────────────────────
base_env = env.unwrapped
config_path = os.path.join(run_dir, "config.txt")
with open(config_path, "w") as f:
    f.write(f"Run: {args.run_name}\n")
    f.write(f"Experiment: test_12_walking\n")
    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
    f.write(f"\nKey changes from test_11:\n")
    f.write(f"  - Survival-scaled alive bonus: base × (1 + step/100)\n")
    f.write(f"  - Makes long episodes far more valuable than short topples\n")
    f.write(f"  - Same obs space (45), same action space, same termination\n")
    f.write(f"\nObservation space: {base_env.n_obs} values\n")
    f.write(f"  [0:3]  proj_gravity, [3:6] ang_vel, [6:8] lin_vel (NEW)\n")
    f.write(f"  [8:20] joint_pos, [20:32] joint_vel\n")
    f.write(f"  [32:44] prev_actions, [44] height\n")
    f.write(f"\nReward (simplified):\n")
    f.write(f"  alive:       +{base_env.w_alive}\n")
    f.write(f"  forward_vel: +{base_env.w_forward} × vel_x\n")
    f.write(f"  orientation: -{base_env.w_orient} × tilt\n")
    f.write(f"  REMOVED: energy, smoothness, joint_limit, vel_target\n")
    f.write(f"\nPPO: lr={args.lr}, ent={args.ent_coef}, arch=[128,128]\n")
    f.write(f"  n_steps=2048, batch=64, epochs=10, gamma=0.99\n")

print(f"Config saved to: {config_path}")

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"\nStarting training for {args.timesteps:,} timesteps...")
start_time = time.time()

try:
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, reward_cb],
        progress_bar=True,
    )
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user.")

elapsed = time.time() - start_time
fps = args.timesteps / elapsed if elapsed > 0 else 0
print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"Effective FPS: {fps:.1f}")

final_path = os.path.join(run_dir, "final_model")
model.save(final_path)
print(f"Final model saved to: {final_path}.zip")

nan_wrapper = env
while hasattr(nan_wrapper, 'env'):
    if isinstance(nan_wrapper, NaNSafeWrapper):
        break
    nan_wrapper = nan_wrapper.env
if isinstance(nan_wrapper, NaNSafeWrapper):
    print(f"\nNaN episodes: {nan_wrapper._nan_episodes} / "
          f"{nan_wrapper._total_episodes} "
          f"({nan_wrapper.nan_rate:.1%})")

env.close()
print("Done.")

# Restore stdout if we were logging
if args.log_file:
    sys.stdout = tee_out.terminal
    sys.stderr = tee_err.terminal
    tee_out.close()
    tee_err.close()
