"""
train_ppo.py — PPO Training for test16_rl (single-robot, ref gait + residuals)
"""
import os, sys, argparse, time
import numpy as np
from datetime import datetime
import gymnasium as gym

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
parser.add_argument("--log-file", type=str, default=None)
args = parser.parse_args()

if args.run_name is None:
    args.run_name = f"nao_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run_dir = os.path.join("runs", args.run_name)
os.makedirs(run_dir, exist_ok=True)

class TeeOutput:
    def __init__(self, p, s):
        self.terminal = s; self.log_file = open(p, "a")
    def write(self, m):
        self.terminal.write(m); self.log_file.write(m); self.log_file.flush()
    def flush(self): self.terminal.flush(); self.log_file.flush()
    def fileno(self): return self.terminal.fileno()
    def close(self): self.log_file.close()

if args.log_file:
    tee_out = TeeOutput(args.log_file, sys.stdout)
    tee_err = TeeOutput(args.log_file, sys.stderr)
    sys.stdout = tee_out; sys.stderr = tee_err

print(f"\n{'='*70}")
print(f"NAO Walking — PPO (test16_rl: ref gait + residuals, single robot)")
print(f"  Run: {args.run_name}, Steps: {args.timesteps:,}, Resume: {args.resume}")
print(f"{'='*70}")

from nao_walking_env import NaoWalkingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

env = NaoWalkingEnv(urdf_path=args.urdf,
                    render_mode="headless" if args.headless else "human")

class NaNSafeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env); self._nan_ep = 0; self._total_ep = 0
    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            self._nan_ep += 1; obs = np.zeros(self.observation_space.shape, np.float32)
            r = 0.0; term = True; info["nan_termination"] = True
        if np.isnan(r) or np.isinf(r): r = 0.0
        return obs, r, term, trunc, info
    def reset(self, **kw):
        self._total_ep += 1; obs, info = self.env.reset(**kw)
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.zeros(self.observation_space.shape, np.float32)
        return obs, info
    @property
    def nan_rate(self): return self._nan_ep / max(self._total_ep, 1)

env = NaNSafeWrapper(env)
env = Monitor(env, os.path.join(run_dir, "monitor"))

class RewardLog(BaseCallback):
    def __init__(self, freq=10): super().__init__(1); self.freq=freq; self.ec=0; self.er=[]; self.el=[]
    def _on_step(self):
        for i in self.locals.get("infos",[]):
            if "episode" in i:
                self.ec+=1; self.er.append(i["episode"]["r"]); self.el.append(i["episode"]["l"])
                if self.ec % self.freq == 0:
                    rr,ll=self.er[-self.freq:],self.el[-self.freq:]
                    print(f"  [Ep {self.ec:>5}] R={np.mean(rr):>7.1f}±{np.std(rr):.1f}  len={np.mean(ll):>5.0f} (max={np.max(ll):>4})  step={self.num_timesteps:>8,}")
        return True

ckpt_cb = CheckpointCallback(save_freq=args.checkpoint_freq, save_path=run_dir, name_prefix="ckpt")
rlog_cb = RewardLog(freq=10)

if args.resume:
    print(f"Resuming from: {args.resume}")
    model = PPO.load(args.resume, env=env, device="cuda")
    model.learning_rate = args.lr; model.ent_coef = args.ent_coef
    model._last_obs = None; model.num_timesteps = 0
else:
    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch":[128,128]},
                learning_rate=args.lr, n_steps=4096, batch_size=256, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=args.ent_coef,
                vf_coef=0.5, max_grad_norm=0.5, device="cuda", seed=args.seed, verbose=1,
                tensorboard_log=os.path.join(run_dir,"tb"))

with open(os.path.join(run_dir,"config.txt"),"w") as f:
    f.write(f"test16_rl: single-robot, ref gait 0.5Hz + residual ±0.2rad, obs=23\n")
    f.write(f"PPO: lr={args.lr}, ent={args.ent_coef}, arch=[128,128], n_steps=2048\n")

print(f"\nTraining {args.timesteps:,} steps...")
t0 = time.time()
try: model.learn(total_timesteps=args.timesteps, callback=[ckpt_cb,rlog_cb], progress_bar=True)
except KeyboardInterrupt: print("\nStopped.")
elapsed = time.time()-t0
print(f"\nDone in {elapsed:.0f}s, FPS={args.timesteps/elapsed:.1f}")
model.save(os.path.join(run_dir,"final_model"))
print(f"Saved: {run_dir}/final_model.zip")
env.close()
if args.log_file:
    sys.stdout=tee_out.terminal; sys.stderr=tee_err.terminal; tee_out.close(); tee_err.close()
