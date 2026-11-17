"""
One-off sanity check (not part of the training pipeline): confirms the
revised reward runs without crashing/NaNs and prints a term-by-term
breakdown for both a smooth (zero-residual, follows reference gait) and
a jerky (random residual, should spin/wobble more) rollout, to eyeball
that r_yaw actually responds to spinning rather than being a no-op.
"""
import numpy as np
from nao_walking_env import NaoWalkingEnv

env = NaoWalkingEnv(urdf_path="../../walking-solutions/shared/robot/nao.urdf",
                     render_mode="headless")

def run(label, action_fn, n=80, seed=0):
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    yaw_rates, rewards = [], []
    for i in range(n):
        action = action_fn(rng)
        obs, r, term, trunc, info = env.step(action)
        _, quat = env._nao.get_world_pose()
        if quat is not None and not np.any(np.isnan(quat)):
            yaw_rate = float(env._get_angular_velocity(quat)[2]) / 0.25
        else:
            yaw_rate = float('nan')
        yaw_rates.append(yaw_rate)
        rewards.append(r)
        if term or trunc:
            break
    yaw_rates = np.array(yaw_rates)
    rewards = np.array(rewards)
    print(f"--- {label} ({len(rewards)} steps) ---")
    print(f"  reward: mean={rewards.mean():.3f} min={rewards.min():.3f} max={rewards.max():.3f} "
          f"any_nan={bool(np.isnan(rewards).any())}")
    print(f"  yaw_rate: mean={np.nanmean(yaw_rates):.4f} abs_mean={np.nanmean(np.abs(yaw_rates)):.4f} "
          f"max_abs={np.nanmax(np.abs(yaw_rates)):.4f} any_nan={bool(np.isnan(yaw_rates).any())}")
    print(f"  implied mean r_yaw = -{env.w_yaw} * mean(yaw_rate^2) = "
          f"{-env.w_yaw * np.nanmean(yaw_rates**2):.4f}")

run("zero residual (follows reference gait)", lambda rng: np.zeros(env.n_legs, dtype=np.float32))
run("random residual (jerky, should wobble/spin more)", lambda rng: rng.uniform(-1, 1, env.n_legs).astype(np.float32))

env.close()
print("DONE - no crash")
