"""
evaluate_policy.py — Evaluate trained NAO walking policy (test16_rl)
"""
import argparse, numpy as np, time

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="runs/exp17_rl/final_model.zip")
parser.add_argument("--urdf", type=str, default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--slow", action="store_true")
args = parser.parse_args()
deterministic = not args.stochastic

print("="*70)
print(f"NAO Walking — Evaluation (exp17: V4 iter 7 ref gait + PPO residuals)")
print(f"  Model: {args.model}, Episodes: {args.episodes}, Det: {deterministic}")
print("="*70)

from stable_baselines3 import PPO
model = PPO.load(args.model, device="cpu")
print(f"Loaded. Obs space: {model.observation_space.shape}")

from nao_walking_env import NaoWalkingEnv
env = NaoWalkingEnv(urdf_path=args.urdf,
                    render_mode="headless" if args.headless else "human",
                    gait_version="v4")   # exp17: V4 iter 7 reference gait
stats = []

print(f"\n{'Ep':>4} {'Steps':>6} {'Reward':>8} {'R/step':>7} {'X':>8} {'Y':>8} {'Vx':>8} {'Tilt':>7} {'Term':<15}")
print("-"*88)

for ep in range(args.episodes):
    obs, _ = env.reset()
    done, total_r, step = False, 0.0, 0
    start_pos = env._get_base_position().copy()
    tilt_samples = []

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(action)
        total_r += r; step += 1; done = term or trunc

        # Sample tilt (proj_gravity x/y magnitude). Skip the first
        # grace_period steps so warmup transients don't pollute the average.
        if step > env.grace_period:
            _, q = env._nao.get_world_pose()
            if q is not None and not np.any(np.isnan(q)):
                pg = env._quat_rotate_inverse(q, np.array([0, 0, -1.0]))
                tilt_samples.append(float(np.sqrt(pg[0]**2 + pg[1]**2)))

        if args.slow and not args.headless: time.sleep(0.05)

    final_pos = env._get_base_position()
    xd = final_pos[0] - start_pos[0]
    yd = final_pos[1] - start_pos[1]
    avg_r = total_r / max(step,1)
    avg_v = xd / (step * env.control_dt) if step > 0 else 0.0
    mean_tilt = float(np.mean(tilt_samples)) if tilt_samples else float("nan")

    pos, quat = env._nao.get_world_pose()
    if pos is None or np.any(np.isnan(pos)): tr = "NaN"
    elif step >= env.max_episode_steps: tr = "max_steps"
    elif pos[2] < env.term_height: tr = f"fell(h={pos[2]:.3f})"
    elif quat is not None and abs(float(quat[3])) < env.term_orient_qw: tr = f"tilt(qw={abs(float(quat[3])):.3f})"
    else: tr = "unknown"

    stats.append({"steps":step,"reward":total_r,"x_dist":xd,"y_dist":yd,"avg_vel":avg_v,
                  "mean_tilt":mean_tilt,"term":tr,"survived":step>env.grace_period})
    print(f"{ep+1:>4} {step:>6} {total_r:>8.1f} {avg_r:>7.2f} {xd:>+7.3f}m {yd:>+7.3f}m {avg_v:>+7.3f} {mean_tilt:>6.3f} {tr:<15}")

print("\n"+"="*88)
print("SUMMARY (exp17)")
steps=[s["steps"] for s in stats]; rewards=[s["reward"] for s in stats]
xdists=[s["x_dist"] for s in stats]; ydists=[s["y_dist"] for s in stats]
vels=[s["avg_vel"] for s in stats]; tilts=[s["mean_tilt"] for s in stats if not np.isnan(s["mean_tilt"])]
surv=sum(1 for s in stats if s["survived"])
print(f"Survived:      {surv}/{args.episodes}")
print(f"Ep length:     {np.mean(steps):.1f} mean,  {min(steps)} min,  {max(steps)} max")
print(f"Reward:        {np.mean(rewards):.1f} mean")
print(f"Forward (X):   {np.mean(xdists):+.4f}m mean,  {max(xdists):+.4f}m max")
print(f"Lateral (Y):   {np.mean(ydists):+.4f}m mean,  abs max {max(abs(min(ydists)), abs(max(ydists))):.4f}m")
print(f"Velocity:      {np.mean(vels):+.4f} m/s mean")
if tilts:
    print(f"Mean tilt:     {np.mean(tilts):.3f}  (lower is better; iter 7 ref ~0.18)")

# ── Comparison against iter 7 reference gait baseline ──
# These numbers come from debug_v4_iter7_long.txt (4000 steps, 6 cycles):
# - Forward: ~+1.0 cm/cycle = 0.00049 m/s = +5.5 cm over 1500 steps
# - Lateral: ~+1.75 cm/cycle leftward = +9.2 cm over 6 cycles
# - Mean tilt: ~0.18 (growing slowly)
print(f"\n  iter 7 ref baseline (open-loop, no RL):")
print(f"    Forward:  ~+0.055m / 1500 steps  (=~+0.0005 m/s)")
print(f"    Lateral:  ~+0.092m / 6 cycles    (=~+1.5 cm/s leftward — the problem)")
print(f"    Tilt:     ~0.18 mean")

# Quality bands based on reference + reasonable RL improvement
mean_x = np.mean(xdists); mean_y_abs = np.mean([abs(y) for y in ydists])
if mean_x > 0.5 and mean_y_abs < 0.05:
    verdict = "★ STRAIGHT WALKING — RL learned forward + lateral correction"
elif mean_x > 0.5:
    verdict = "✓ FORWARD WALKING — but Y drift not corrected"
elif mean_x > 0.05:
    verdict = "◐ SOME FORWARD MOTION — slow but RL is doing better than reference"
elif mean_x > 0.005:
    verdict = "◑ AT REFERENCE BASELINE — RL hasn't improved beyond open-loop gait"
else:
    verdict = "✗ NO FORWARD MOTION — policy may be standing still or worse than ref"
print(f"\n{verdict}")

from collections import Counter
print(f"\nTermination reasons:")
for r,c in Counter(s["term"].split("(")[0] for s in stats).most_common():
    print(f"  {r:<15} {c}")
env.close()
