"""
evaluate_policy.py — Evaluate trained NAO walking policy (test_26_sac /
algorithm-agnostic: also used for the PPO baseline via --algo ppo)
"""
import argparse, numpy as np, time

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, choices=["sac", "ppo"], default="sac")
parser.add_argument("--model", type=str, default="runs/sac_pilot/final_model.zip")
parser.add_argument("--urdf", type=str, default="../../walking-solutions/shared/robot/nao.urdf")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--slow", action="store_true")
args = parser.parse_args()
deterministic = not args.stochastic

print("="*70)
print(f"NAO Walking — Evaluation ({args.algo.upper()})")
print(f"  Model: {args.model}, Episodes: {args.episodes}, Det: {deterministic}")
print("="*70)

from stable_baselines3 import SAC, PPO
AlgoClass = {"sac": SAC, "ppo": PPO}[args.algo]
model = AlgoClass.load(args.model, device="cpu")
print(f"Loaded {args.algo.upper()}. Obs space: {model.observation_space.shape}")

from nao_walking_env import NaoWalkingEnv
env = NaoWalkingEnv(urdf_path=args.urdf,
                    render_mode="headless" if args.headless else "human")
stats = []

print(f"\n{'Ep':>4} {'Steps':>6} {'Reward':>8} {'R/step':>8} {'X dist':>8} {'Avg vel':>8} {'Term':<15}")
print("-"*75)

for ep in range(args.episodes):
    obs, _ = env.reset()
    done, total_r, step = False, 0.0, 0
    start_pos = env._get_base_position().copy()

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(action)
        total_r += r; step += 1; done = term or trunc
        if args.slow and not args.headless: time.sleep(0.05)

    final_pos = env._get_base_position()
    xd = -(final_pos[0] - start_pos[0])
    avg_r = total_r / max(step,1)
    avg_v = xd / (step * env.control_dt) if step > 0 else 0.0

    pos, quat = env._nao.get_world_pose()
    if pos is None or np.any(np.isnan(pos)): tr = "NaN"
    elif step >= env.max_episode_steps: tr = "max_steps"
    elif pos[2] < env.term_height: tr = f"fell(h={pos[2]:.3f})"
    elif quat is not None and abs(float(quat[3])) < env.term_orient_qw: tr = f"tilt(qw={abs(float(quat[3])):.3f})"
    else: tr = "unknown"

    stats.append({"steps":step,"reward":total_r,"x_dist":xd,"avg_vel":avg_v,"term":tr,
                  "survived":step>env.grace_period})
    print(f"{ep+1:>4} {step:>6} {total_r:>8.1f} {avg_r:>8.2f} {xd:>7.3f}m {avg_v:>7.3f}m/s {tr:<15}")

print("\n"+"="*75)
print(f"SUMMARY ({args.algo.upper()})")
steps=[s["steps"] for s in stats]; rewards=[s["reward"] for s in stats]
dists=[s["x_dist"] for s in stats]; vels=[s["avg_vel"] for s in stats]
surv=sum(1 for s in stats if s["survived"])
print(f"Survived: {surv}/{args.episodes}")
print(f"Ep length: {np.mean(steps):.1f} mean, {min(steps)} min, {max(steps)} max")
print(f"Reward:    {np.mean(rewards):.1f} mean")
print(f"X dist:    {np.mean(dists):.3f}m mean, {max(dists):.3f}m max")
print(f"Velocity:  {np.mean(vels):.3f} m/s mean")

if any(d>2.0 for d in dists): print("\n★ SUSTAINED WALKING!")
elif any(d>0.5 for d in dists): print("\n✓ WALKING DETECTED")
elif any(d>0.05 for d in dists): print("\n◐ SOME MOVEMENT")
else: print("\n✗ NO SIGNIFICANT MOTION")

from collections import Counter
for r,c in Counter(s["term"].split("(")[0] for s in stats).most_common():
    print(f"  {r:<15} {c}")
env.close()
