"""
train_sac.py — SAC pilot training for test_26_sac (single-robot, Webots
reference gait + residual actions — same env/reward/physics as
test_24/baseline_test24; only the algorithm differs).

Chunked-training note: run_chunked_training.sh restarts this script as a
fresh process every chunk. Unlike PPO's transient rollout buffer, SAC's
replay buffer must survive that restart explicitly:
  - the replay buffer is saved/loaded via save_replay_buffer()/
    load_replay_buffer() (excluded from model.save() by default in SB3),
  - model.num_timesteps is NEVER manually reset on resume (SAC.load()
    already restores it), so learning_starts' random-exploration warmup
    triggers exactly once across the whole run, not once per chunk,
  - learn() is called with reset_num_timesteps=False so it doesn't zero
    num_timesteps back out again.
See APPROACH.md for the full audit this is based on.
"""
import os, sys, argparse, time
import numpy as np
from datetime import datetime
import gymnasium as gym

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../walking-solutions/shared/robot/nao.urdf")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--timesteps", type=int, default=200_000, help="env steps to run THIS chunk")
parser.add_argument("--checkpoint-freq", type=int, default=50_000)
parser.add_argument("--resume", type=str, default=None, help="path to a previous chunk's final_model.zip")
parser.add_argument("--run-name", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--ent-coef", type=str, default="auto")
parser.add_argument("--log-file", type=str, default=None)
args = parser.parse_args()

if args.run_name is None:
    args.run_name = f"nao_sac_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run_dir = os.path.join("runs", args.run_name)
os.makedirs(run_dir, exist_ok=True)
replay_buffer_path = os.path.join(run_dir, "replay_buffer.pkl")


def save_replay_buffer_atomic(model, path):
    """Write to a temp file then rename, so a kill mid-write (e.g. the
    chunk runner's RAM gate) can never leave the on-disk buffer corrupt —
    os.replace() is atomic on POSIX."""
    tmp_path = path + ".tmp"
    model.save_replay_buffer(tmp_path)
    os.replace(tmp_path, path)


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
print(f"NAO Walking — SAC pilot (test_26_sac: ref gait + residuals, single robot)")
print(f"  Run: {args.run_name}, Steps: {args.timesteps:,}, Resume: {args.resume}")
print(f"{'='*70}")

from nao_walking_env import NaoWalkingEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

env = NaoWalkingEnv(urdf_path=args.urdf,
                    render_mode="headless" if args.headless else "human")

class NaNSafeWrapper(gym.Wrapper):
    """Sanitizes BOTH the observation and the reward on a NaN/Inf step —
    not just the termination flag. A NaN reward alone (obs otherwise
    fine) would still poison SAC's critic TD target even with correct
    done-masking, since the target is bootstrapped from (reward, next_obs,
    done) regardless of whether the episode ended."""
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
# override_existing=False on resume so episode history accumulates across
# chunk restarts instead of each chunk wiping monitor.monitor.csv back to
# empty (Monitor opens in "w" mode by default) — needed for ep_len_mean/
# ep_rew_mean-over-time to be reconstructable from the full run, not just
# the last chunk. First-ever chunk still writes fresh (with header).
env = Monitor(env, os.path.join(run_dir, "monitor"), override_existing=(args.resume is None))

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

# save_replay_buffer=False here deliberately: the buffer is saved once,
# atomically, at the end of this script (see below) rather than via this
# callback's own non-atomic periodic save.
ckpt_cb = CheckpointCallback(save_freq=args.checkpoint_freq, save_path=run_dir, name_prefix="ckpt")
rlog_cb = RewardLog(freq=10)

if args.resume:
    print(f"Resuming from: {args.resume}")
    # force_reset=True is SB3's default (confirmed SB3 2.7.1) — kept
    # explicit here as a deliberate choice, not an oversight.
    model = SAC.load(args.resume, env=env, device="cuda", force_reset=True)
    if os.path.exists(replay_buffer_path):
        model.load_replay_buffer(replay_buffer_path)
        print(f"Loaded replay buffer: {model.replay_buffer.size():,} transitions")
    else:
        print("WARNING: no replay_buffer.pkl found next to the resumed model — "
              "continuing with an EMPTY buffer even though num_timesteps and "
              "learning_starts have already advanced. This should only happen "
              "on a first-ever resume after a run that predates buffer "
              "persistence; if it happens repeatedly, something is wrong.")
    # num_timesteps is intentionally left untouched — SAC.load() already
    # restored it, and learning_starts depends on it staying correct.
else:
    model = SAC("MlpPolicy", env, policy_kwargs={"net_arch": [128, 128]},
                learning_rate=args.lr,
                buffer_size=1_000_000,
                learning_starts=100,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef=args.ent_coef,
                target_entropy="auto",
                use_sde=False,
                device="cuda", seed=args.seed, verbose=1,
                tensorboard_log=os.path.join(run_dir, "tb"))

with open(os.path.join(run_dir, "config.txt"), "w") as f:
    f.write("test_26_sac pilot: single-robot, literal Webots ref gait + residual "
            "(scale=0.25rad), obs=23 — SAME env/reward/physics as test_24\n")
    f.write(f"SAC (SB3 2.7.1 defaults except net_arch/gamma matched to test_24): "
            f"lr={args.lr}, net_arch=[128,128], buffer_size=1_000_000, "
            f"learning_starts=100, batch_size=256, tau=0.005, gamma=0.99, "
            f"train_freq=1, gradient_steps=1, ent_coef={args.ent_coef}, "
            f"target_entropy=auto, use_sde=False\n")

print(f"\nTraining {args.timesteps:,} steps (this chunk)...")
t0 = time.time()
try:
    model.learn(total_timesteps=args.timesteps, callback=[ckpt_cb, rlog_cb],
                reset_num_timesteps=False, progress_bar=True)
except KeyboardInterrupt:
    print("\nStopped.")
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s, FPS={args.timesteps/elapsed:.1f}, "
      f"cumulative num_timesteps={model.num_timesteps:,}")

model.save(os.path.join(run_dir, "final_model"))
save_replay_buffer_atomic(model, replay_buffer_path)
print(f"Saved: {run_dir}/final_model.zip + {replay_buffer_path} "
      f"({model.replay_buffer.size():,} transitions)")

env.close()
if args.log_file:
    sys.stdout = tee_out.terminal; sys.stderr = tee_err.terminal; tee_out.close(); tee_err.close()
