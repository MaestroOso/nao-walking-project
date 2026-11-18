#!/usr/bin/env python3
"""
check_sim_traces.py — compare robot-side reconstruction with simulator ground truth.

Input: a traces file from export/record_sim_traces.py. Two checks:

1. Observation reconstruction (PPO / SAC). From the ground truth, build the
   inputs the robot will have (IMU roll/pitch/yaw, body-frame gyro, true
   velocity and height) and rebuild the observation with the robot-side
   ObservationBuilder. It must equal the env's observation; differences are
   reported per component.
2. Estimators. Feed joint angles + IMU-like orientation through the robot's
   StateEstimator (leg kinematics) and compare its torso height and
   heading-frame velocity with the simulator's. The mean height difference
   is the value to put in robot.yaml `estimator.height_offset_m`. Only samples
   with the torso tilted less than --max-tilt-deg count: once the robot is
   falling the stance foot is no longer flat and static, and no kinematic
   estimate is meaningful (the robot side stops the run long before that).

    python3 robot-integration/tools/check_sim_traces.py robot-integration/exported/traces_<name>.npz
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROBOT_DIR = HERE.parent
REPO_ROOT = ROBOT_DIR.parent
sys.path.insert(0, str(ROBOT_DIR / "nao_rl_bridge"))

from nao_rl_bridge.core.deployment_spec import load_deployment  # noqa: E402
from nao_rl_bridge.core.estimators import RobotState, SensorSample, StateEstimator  # noqa: E402
from nao_rl_bridge.core.gait import ReferenceGait  # noqa: E402
from nao_rl_bridge.core.kinematics import LegKinematics  # noqa: E402
from nao_rl_bridge.core.observation import ObservationBuilder  # noqa: E402
from nao_rl_bridge.core.rotations import rotation_matrix, rpy_from_quat, tilt_angle  # noqa: E402


def heading_velocity(quat, v_world):
    _, _, yaw = rpy_from_quat(quat)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * v_world[0] + s * v_world[1], -s * v_world[0] + c * v_world[1]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", type=Path)
    ap.add_argument("--urdf", type=Path, default=REPO_ROOT / "walking-solutions/shared/robot/nao.urdf")
    ap.add_argument("--alpha", type=float, default=0.5, help="velocity filter alpha to test")
    ap.add_argument("--max-tilt-deg", type=float, default=15.0)
    a = ap.parse_args()

    tr = np.load(a.traces)
    meta = json.loads(str(tr["meta"]))
    spec = load_deployment(meta["deployment"])
    n = len(tr["step"])
    dt = spec.control_dt
    print(f"{a.traces.name}: {n} steps of {spec.name}")

    # ── 1. observation reconstruction ──
    if spec.observation is not None:
        builder = ObservationBuilder(spec)
        gait = ReferenceGait(spec)
        warm = np.asarray(spec.observation.component("gait_phase").options["warmup_value"], float)
        errs = {c.name: 0.0 for c in spec.observation.components}
        for i in range(n):
            q, step = tr["quat"][i], int(tr["step"][i])
            roll, pitch, yaw = rpy_from_quat(q)
            st = RobotState(leg_q=tr["leg_q"][i], roll=roll, pitch=pitch, yaw_world=yaw,
                            quat_world=q, gyro_body=rotation_matrix(q).T @ tr["ang_vel"][i],
                            v_heading=heading_velocity(q, tr["lin_vel"][i]),
                            base_height=float(tr["pos"][i][2]), stance_foot="", odometry=np.zeros(2))
            phase = warm if step < spec.warmup_steps else gait.phase_signal(
                gait.phase((step - spec.warmup_steps) * dt))
            parts = builder.components(st, phase)
            k = 0
            for c in spec.observation.components:
                errs[c.name] = max(errs[c.name], float(np.max(np.abs(parts[c.name] - tr["obs"][i][k:k + c.dim]))))
                k += c.dim
        print("\n1. Observation reconstruction (max |robot - env|):")
        for name, e in errs.items():
            print(f"   {name:22s} {e:.2e}  {'OK' if e < 1e-4 else 'MISMATCH'}")

    # ── 2. estimators vs ground truth ──
    est = StateEstimator(LegKinematics(a.urdf), spec.leg_order,
                         math.radians(spec.observation.spawn_yaw_deg) if spec.observation else math.pi,
                         velocity_filter_alpha=a.alpha)
    h_est, h_sim, v_est, v_sim, upright = [], [], [], [], []
    for i in range(n):
        q = tr["quat"][i]
        roll, pitch, yaw = rpy_from_quat(q)
        s = SensorSample(joints=dict(zip(spec.leg_order, tr["leg_q"][i])), roll=roll, pitch=pitch,
                         yaw=yaw, gyro=rotation_matrix(q).T @ tr["ang_vel"][i], stamp=i * dt)
        st = est.update(s)
        h_est.append(st.base_height)
        h_sim.append(tr["pos"][i][2])
        v_est.append(st.v_heading)
        v_sim.append(heading_velocity(q, tr["lin_vel"][i]))
        upright.append(i > 0 and math.degrees(tilt_angle(roll, pitch)) < a.max_tilt_deg)
    keep = np.array(upright)
    h_est, h_sim = np.array(h_est)[keep], np.array(h_sim)[keep]
    v_est, v_sim = np.array(v_est)[keep], np.array(v_sim)[keep]
    dh = h_sim - h_est
    print(f"\n2. Estimators vs simulator ground truth ({keep.sum()}/{n} samples with tilt "
          f"< {a.max_tilt_deg:g} deg):")
    print(f"   torso height: sim - estimate = {dh.mean() * 1000:+.1f} mm mean, "
          f"{dh.std() * 1000:.1f} mm std  -> suggested estimator.height_offset_m: {dh.mean():.4f}")
    for j, name in enumerate(("forward", "lateral")):
        rmse = float(np.sqrt(np.mean((v_est[:, j] - v_sim[:, j]) ** 2)))
        corr = float(np.corrcoef(v_est[:, j], v_sim[:, j])[0, 1]) if np.std(v_sim[:, j]) > 1e-9 else float("nan")
        print(f"   {name:8s} velocity: RMSE {rmse:.3f} m/s, correlation {corr:.2f} "
              f"(sim mean {v_sim[:, j].mean():+.3f}, estimate mean {v_est[:, j].mean():+.3f})")


if __name__ == "__main__":
    main()
