"""
pretrain.py — Behavior cloning pre-training for test16_rl
=========================================================
TWO-PHASE: --collect (Isaac Sim) then --train (no Isaac Sim)

USAGE:
  python pretrain.py --collect --headless --episodes 20
  python pretrain.py --train --epochs 50
"""

import os, sys, argparse
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--output", type=str, default="runs/test14b_walking/pretrained_model.zip")
parser.add_argument("--data-file", type=str, default="runs/test14b_walking/pretrain_data.npz")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--collect", action="store_true")
parser.add_argument("--train", action="store_true")
args = parser.parse_args()

if not args.collect and not args.train:
    print("No --collect or --train flag. Running --collect only.")
    args.collect = True

os.makedirs(os.path.dirname(args.data_file), exist_ok=True)
os.makedirs(os.path.dirname(args.output), exist_ok=True)

print(f"\n{'='*70}")
print(f"NAO Walking — Behavior Cloning (test16_rl, single-robot gym.Env)")
print(f"  Collect: {args.collect}, Train: {args.train}")
print(f"{'='*70}\n")

if args.collect:
    print(f"── PHASE 1: Data Collection ({args.episodes} episodes) ──\n")
    from nao_walking_env import NaoWalkingEnv
    render_mode = "headless" if args.headless else "human"
    env = NaoWalkingEnv(urdf_path=args.urdf, render_mode=render_mode)

    observations, target_actions, episode_info = [], [], []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done, step = False, 0
        while not done:
            zero_action = np.zeros(12, dtype=np.float32)
            observations.append(obs.copy())
            target_actions.append(zero_action.copy())
            obs, reward, terminated, truncated, info = env.step(zero_action)
            step += 1
            done = terminated or truncated

        pos, quat = env._nao.get_world_pose()
        h = float(pos[2]) if pos is not None else -1.0
        proj_grav = env._quat_rotate_inverse(quat, np.array([0,0,-1.0]))
        tilt = float(np.sqrt(proj_grav[0]**2 + proj_grav[1]**2)) if pos is not None else -1.0
        episode_info.append({"length": step, "height": h, "tilt": tilt})
        print(f"  Ep {ep+1}/{args.episodes}: len={step}  h={h:.3f}m  tilt={tilt:.3f}")

    observations = np.array(observations, dtype=np.float32)
    target_actions = np.array(target_actions, dtype=np.float32)
    lengths = [e["length"] for e in episode_info]
    survived = sum(1 for l in lengths if l > 51)

    print(f"\nCollected {len(observations)} samples")
    print(f"Ep lengths: mean={np.mean(lengths):.1f} min={min(lengths)} max={max(lengths)}")
    print(f"Survived past grace: {survived}/{len(lengths)}")
    if survived == 0:
        print(f"\n⚠ WARNING: Reference gait not stable — all die at/before grace")

    np.savez(args.data_file, observations=observations, target_actions=target_actions,
             episode_lengths=np.array(lengths, dtype=np.int32))
    print(f"✓ Data saved to: {args.data_file}")
    env.close()
    print(f"\nNext: python pretrain.py --train --epochs {args.epochs}")
    sys.exit(0)

if args.train:
    print(f"── PHASE 2: Supervised Training ──\n")
    if not os.path.exists(args.data_file):
        print(f"Data not found: {args.data_file}"); sys.exit(1)

    data = np.load(args.data_file)
    observations, target_actions = data["observations"], data["target_actions"]
    print(f"Loaded {len(observations)} samples")

    import torch, torch.nn as nn, torch.optim as optim
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import gymnasium as gym

    class DummyEnv(gym.Env):
        def __init__(self, od, ad):
            super().__init__()
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (od,), np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (ad,), np.float32)
        def reset(self, **k): return np.zeros(self.observation_space.shape, np.float32), {}
        def step(self, a): return np.zeros(self.observation_space.shape, np.float32), 0., False, False, {}

    dummy_env = DummyVecEnv([lambda: DummyEnv(observations.shape[1], target_actions.shape[1])])
    model = PPO("MlpPolicy", dummy_env, policy_kwargs={"net_arch": [128, 128]},
                learning_rate=args.lr, device="cpu", seed=args.seed, verbose=0)

    obs_t = torch.tensor(observations, dtype=torch.float32)
    tgt_t = torch.tensor(target_actions, dtype=torch.float32)
    optimizer = optim.Adam(model.policy.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        perm = torch.randperm(len(obs_t))
        losses = []
        for s in range(0, len(obs_t), args.batch_size):
            idx = perm[s:s+args.batch_size]
            features = model.policy.extract_features(obs_t[idx])
            if model.policy.share_features_extractor:
                lp, _ = model.policy.mlp_extractor(features)
            else:
                pf, _ = features; lp = model.policy.mlp_extractor.forward_actor(pf)
            pred = model.policy.action_net(lp)
            loss = loss_fn(pred, tgt_t[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3}/{args.epochs}  loss={np.mean(losses):.6f}")

    with torch.no_grad():
        features = model.policy.extract_features(obs_t[:100])
        if model.policy.share_features_extractor:
            lp, _ = model.policy.mlp_extractor(features)
        else:
            pf, _ = features; lp = model.policy.mlp_extractor.forward_actor(pf)
        p = model.policy.action_net(lp)
        print(f"\nVerify — action mag: mean={p.abs().mean():.6f} max={p.abs().max():.6f}")

    op = args.output[:-4] if args.output.endswith(".zip") else args.output
    model.save(op)
    print(f"\n✓ Saved: {op}.zip")
    print(f"Next: python evaluate_policy.py --model {op}.zip --episodes 5 --headless")
