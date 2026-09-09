"""
Train NAO - With Episode Logging
"""

import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from nao_env_reduced import NaoStandingEnv

print("="*80)
print("NAO STANDING TRAINING")
print("="*80)
print()

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_DIR = f"results_{TIMESTAMP}"
os.makedirs(RESULTS_DIR, exist_ok=True)

LOG_FILE = f"{RESULTS_DIR}/episodes.csv"

print(f"📁 Results directory: {RESULTS_DIR}")
print(f"📊 Episode log: {LOG_FILE}")
print()

env = NaoStandingEnv(headless=False, log_file=LOG_FILE)
env = Monitor(env, RESULTS_DIR)

model = PPO(
    "MlpPolicy", env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    verbose=1,
    tensorboard_log=RESULTS_DIR
)

print("="*80)
print("🚀 STARTING TRAINING")
print("="*80)
print()
print("Progress will be logged to CSV file")
print("Console output reduced (every 10 episodes)")
print()

try:
    model.learn(total_timesteps=50_000, progress_bar=True)
    model.save(f"{RESULTS_DIR}/final")
    print("\n✅ TRAINING COMPLETE!")
except KeyboardInterrupt:
    print("\n⚠️ INTERRUPTED BY USER")
except Exception as e:
    print(f"\n❌ TRAINING ERROR: {e}")

env.close()

print("\n" + "="*80)
print("📊 RESULTS")
print("="*80)
print(f"Episode log: {LOG_FILE}")
print(f"TensorBoard: tensorboard --logdir {RESULTS_DIR}")
print("="*80)
