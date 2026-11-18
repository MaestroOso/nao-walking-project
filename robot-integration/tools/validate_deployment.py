#!/usr/bin/env python3
"""
validate_deployment.py — check the deployment.yaml of one or more solutions.

Usage:
    python robot-integration/tools/validate_deployment.py              # every walking-solutions/*/*/deployment.yaml
    python robot-integration/tools/validate_deployment.py path/to/deployment.yaml
    python robot-integration/tools/validate_deployment.py --require-artifacts   # also require the trained model

Exit code 0 if every file is valid, 1 otherwise.
"""
import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROBOT_DIR = HERE.parent
REPO_ROOT = ROBOT_DIR.parent
sys.path.insert(0, str(ROBOT_DIR / "nao_rl_bridge"))

from nao_rl_bridge.core.deployment_spec import (  # noqa: E402
    DeploymentSpecError, load_deployment)


def summarize(spec) -> str:
    lines = [f"OK   {spec.path}",
             f"     {spec.name}  [{spec.algorithm}, {spec.policy.format}]  "
             f"{spec.control_hz:g} Hz, warm-up {spec.warmup_steps} steps, "
             f"max {spec.max_episode_steps} steps"]
    q, t = spec.load_gait_arrays()
    lines.append(f"     gait: {len(t)} frames, {t[-1]:.2f} s, after end: {spec.gait.after_end}, "
                 f"phase period {spec.gait.phase_period_s:.2f} s")
    if spec.observation:
        o = spec.observation
        comps = ", ".join(f"{c.name}({c.dim})" for c in o.components)
        lines.append(f"     obs dim {o.dim}, quaternion {o.quaternion_convention}, "
                     f"spawn yaw {o.spawn_yaw_deg:g} deg: {comps}")
        lines.append(f"     action: {spec.action.type}, scale {spec.action.scale_rad} rad "
                     f"({math.degrees(spec.action.scale_rad):.1f} deg)")
    if spec.pgrl:
        lines.append(f"     pgrl controller: {spec.pgrl.controller}, run: {spec.policy.run_dir}")
    for w in spec.warnings:
        lines.append(f"     WARN {w}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--require-artifacts", action="store_true",
                    help="treat a missing trained model / PGRL run as an error")
    args = ap.parse_args()

    paths = args.paths or sorted((REPO_ROOT / "walking-solutions").glob("*/*/deployment.yaml"))
    if not paths:
        print("No deployment.yaml found under walking-solutions/*/*/")
        return 1

    failed = 0
    for p in paths:
        try:
            print(summarize(load_deployment(p, require_artifacts=args.require_artifacts)))
        except DeploymentSpecError as e:
            failed += 1
            print(f"FAIL {e}")
    print(f"\n{len(paths) - failed}/{len(paths)} valid")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
