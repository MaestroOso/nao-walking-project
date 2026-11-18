#!/usr/bin/env python3
"""
record_sim_traces.py — record a simulated episode for tools/check_sim_traces.py.

Runs a solution's training env in Isaac Sim and logs, every control step, the
simulator's ground truth (torso pose, world-frame angular and linear
velocity, leg joint angles) together with the observation the env built.
check_sim_traces.py then rebuilds the observation and the estimates (torso
height, velocity) the way the robot will, and compares.

Run inside the Isaac Sim Python env:

    setup/isaac_env/bin/python robot-integration/export/record_sim_traces.py \
        walking-solutions/ppo/<variant> --export robot-integration/exported/<name>
    # without a trained policy (zero residual / initial PGRL parameters):
    setup/isaac_env/bin/python robot-integration/export/record_sim_traces.py \
        walking-solutions/ppo/baseline_test24 --zero-actions

The env module defaults to `nao_walking_env.NaoWalkingEnv` (PPO, SAC) or
`pgrl_env.PGRLWalkingEnv` (PGRL) in the solution folder; override with
--env-file / --env-class / --cwd if a solution is organised differently.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROBOT_DIR = HERE.parent
sys.path.insert(0, str(ROBOT_DIR / "nao_rl_bridge"))

from nao_rl_bridge.core.deployment_spec import load_deployment  # noqa: E402
from nao_rl_bridge.core.exported import load_export  # noqa: E402
from nao_rl_bridge.core.policies import PGRL_PARAM_NAMES  # noqa: E402

DEFAULT_ENV = {"sb3": ("nao_walking_env.py", "NaoWalkingEnv"),
               "pgrl_run": ("pgrl_env.py", "PGRLWalkingEnv")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("solution", type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--export", type=Path, help="exported policy folder that drives the robot")
    g.add_argument("--zero-actions", action="store_true",
                   help="zero residual (PPO/SAC) or initial parameters (PGRL)")
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--env-file", type=Path)
    ap.add_argument("--env-class")
    ap.add_argument("--cwd", type=Path, help="working dir for the env (default: solution folder)")
    ap.add_argument("--gui", action="store_true")
    a = ap.parse_args()

    yaml_path = a.solution / "deployment.yaml" if a.solution.is_dir() else a.solution
    spec = load_deployment(yaml_path)
    env_file = (a.env_file or spec.path.parent / DEFAULT_ENV[spec.policy.format][0]).resolve()
    env_cls_name = a.env_class or DEFAULT_ENV[spec.policy.format][1]
    out = (a.out or ROBOT_DIR / "exported" / f"traces_{spec.name}.npz").resolve()
    exported = load_export(a.export) if a.export else None
    if exported and exported.spec.name != spec.name:
        sys.exit(f"export '{exported.spec.name}' does not belong to solution '{spec.name}'")

    # SB3 is not needed: the exported NumPy policy drives the env.
    os.chdir(a.cwd or spec.path.parent)
    sys.path.insert(0, str(env_file.parent))
    mspec = importlib.util.spec_from_file_location("solution_env", env_file)
    mod = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(mod)
    env = getattr(mod, env_cls_name)(render_mode="human" if a.gui else "headless")
    if spec.policy.format == "pgrl_run":
        phys = (None if exported is None
                else [exported.pgrl.values[n] for n in PGRL_PARAM_NAMES])
        env.set_params(phys=phys)

    obs, _ = env.reset()
    rec = {k: [] for k in ("pos", "quat", "ang_vel", "lin_vel", "leg_q", "obs", "step",
                           "action", "terminated")}
    for _ in range(a.steps):
        pos, quat = env._nao.get_world_pose()
        jp = env._nao.get_joint_positions()
        rec["pos"].append(pos)
        rec["quat"].append(quat)
        rec["ang_vel"].append(env._nao.get_angular_velocity())
        rec["lin_vel"].append(env._nao.get_linear_velocity())
        rec["leg_q"].append(np.asarray(jp)[env._leg_dof_indices])
        rec["obs"].append(obs)
        rec["step"].append(env._step_count)
        if exported is not None and exported.mlp is not None:
            action = exported.mlp.act(obs)
        else:
            action = np.zeros(12, dtype=np.float32)
        rec["action"].append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        rec["terminated"].append(terminated)
        if terminated or truncated:
            break

    meta = {"deployment": str(spec.path), "env_file": str(env_file),
            "env_file_sha256": hashlib.sha256(env_file.read_bytes()).hexdigest(),
            "export": str(a.export) if a.export else None, "control_dt": spec.control_dt}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: np.asarray(v, dtype=np.float64) for k, v in rec.items()},
             meta=json.dumps(meta))
    print(f"[traces] {len(rec['step'])} steps -> {out}", flush=True)
    env.close()   # Isaac's SimulationApp.close() ends the process: save before it


if __name__ == "__main__":
    main()
