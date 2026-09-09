"""
stability_sweep.py — Find NAO's Stable Standing Pose
=====================================================

PURPOSE
-------
Systematically test combinations of hip-pitch, knee-pitch, and ankle-pitch
to find joint configurations where NAO can stand under free physics (no
control input) for a sustained period without falling.

HOW IT WORKS
------------
1. Loads NAO into Isaac Sim (headless — no GUI, fast).
2. For each (hip, knee, ankle) combination in a grid:
   a. Teleports NAO to spawn height with the candidate joint pose.
   b. Lets physics run for TRIAL_SECONDS with zero control.
   c. Samples torso height and orientation every few steps.
   d. Records whether the robot stayed upright the whole time.
3. Prints a ranked table of all poses sorted by stability.
4. Saves results to a CSV for later analysis.

KEY JOINTS (sagittal plane — forward/backward balance)
------------------------------------------------------
  HipPitch   — tilts the torso forward (negative = forward lean)
  KneePitch  — bends the knee (positive = bent)
  AnklePitch — tilts the foot/shin angle (negative = shin tilts forward)

The geometric constraint for static balance:
  hip_pitch + knee_pitch + ankle_pitch ≈ 0
  (so the torso stays vertical while the leg segments form a closed chain)

This is a simplification — the actual constraint depends on segment lengths —
but it's a good starting point for the sweep.

USAGE
-----
  cd <your_test_folder>
  python stability_sweep.py

  Optional flags:
    --urdf PATH       path to NAO URDF (default: ../../robots/nao_isaac_sim/nao.urdf)
    --seconds N       physics duration per trial (default: 5)
    --spawn-height H  initial torso Z (default: 0.335)
    --output FILE     CSV output path (default: sweep_results.csv)
    --fine            use a finer grid (slower but more precise)
"""

import math
import argparse
import csv
import itertools
import numpy as np
from datetime import datetime

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="NAO standing stability sweep")
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf",
                    help="Path to NAO URDF file")
parser.add_argument("--seconds", type=float, default=5.0,
                    help="How many seconds of free physics per trial")
parser.add_argument("--spawn-height", type=float, default=0.335,
                    help="Initial torso Z height (meters)")
parser.add_argument("--output", default="sweep_results.csv",
                    help="CSV output file")
parser.add_argument("--fine", action="store_true",
                    help="Use finer grid (5-degree steps instead of 10)")
args = parser.parse_args()

# ── Sweep grid ────────────────────────────────────────────────────────────────
# Coarse grid: 10-degree steps.  Fine grid: 5-degree steps.
# Ranges chosen based on NAO's joint limits and physical plausibility.
#   HipPitch:   -40 to 0   (negative = forward lean)
#   KneePitch:    0 to 60   (positive = bent)
#   AnklePitch: -30 to 10   (negative = forward tilt)

step = 5 if args.fine else 10

hip_range   = list(range(-40, 1, step))     # degrees
knee_range  = list(range(0, 61, step))       # degrees
ankle_range = list(range(-30, 11, step))     # degrees

all_combos = list(itertools.product(hip_range, knee_range, ankle_range))
total = len(all_combos)

print("=" * 80)
print("NAO STABILITY SWEEP")
print("=" * 80)
print(f"  Grid step:    {step}°")
print(f"  Hip range:    {hip_range[0]}° to {hip_range[-1]}°")
print(f"  Knee range:   {knee_range[0]}° to {knee_range[-1]}°")
print(f"  Ankle range:  {ankle_range[0]}° to {ankle_range[-1]}°")
print(f"  Combinations: {total}")
print(f"  Trial time:   {args.seconds}s each")
print(f"  Spawn height: {args.spawn_height}m")
print(f"  Output:       {args.output}")
print("=" * 80)
print()

# ── Isaac Sim startup ─────────────────────────────────────────────────────────
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "width": 640, "height": 480})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
import omni.kit.commands
import isaacsim.asset.importer.urdf._urdf as urdf_module

world = World(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
world.scene.add_default_ground_plane()

# Import NAO
config = urdf_module.ImportConfig()
config.merge_fixed_joints = False
config.fix_base = False
config.import_inertia_tensor = True
config.distance_scale = 1.0

result = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=args.urdf,
    import_config=config,
)

nao = world.scene.add(Articulation(result[1], name="nao"))
world.reset()

dof_names = nao.dof_names
num_dof = nao.num_dof

print(f"NAO loaded: {num_dof} DOFs")
print(f"DOF names: {dof_names}")
print()

# ── Arm pose (constant across all trials) ─────────────────────────────────────
# Arms lowered to sides so they don't affect balance.
ARM_JOINTS = {
    "LShoulderPitch": math.radians(84),
    "RShoulderPitch": math.radians(84),
    "LShoulderRoll":  math.radians(10),
    "RShoulderRoll":  math.radians(-10),
    "LElbowYaw":      math.radians(-70),
    "RElbowYaw":      math.radians(70),
    "LElbowRoll":     math.radians(-25),
    "RElbowRoll":     math.radians(25),
}

# Physics timing
PHYSICS_DT = 1.0 / 120.0
STEPS_PER_TRIAL = int(args.seconds / PHYSICS_DT)
SAMPLE_INTERVAL = int(0.1 / PHYSICS_DT)  # sample every 0.1s

# ── Fall detection thresholds ─────────────────────────────────────────────────
# qw of a unit quaternion: 1.0 = perfectly upright, <0.7 ≈ >45° tilt
MIN_QW = 0.7
# Torso height: NAO standing is ~0.33m, fallen is <0.15m
MIN_HEIGHT = 0.15


def build_pose(hip_deg, knee_deg, ankle_deg):
    """Build a full DOF array from the three leg-pitch angles + arm pose."""
    pose = np.zeros(num_dof, dtype=np.float32)

    joint_map = {
        "LHipPitch":   math.radians(hip_deg),
        "RHipPitch":   math.radians(hip_deg),
        "LKneePitch":  math.radians(knee_deg),
        "RKneePitch":  math.radians(knee_deg),
        "LAnklePitch": math.radians(ankle_deg),
        "RAnklePitch": math.radians(ankle_deg),
    }
    joint_map.update(ARM_JOINTS)

    for name, angle in joint_map.items():
        if name in dof_names:
            pose[dof_names.index(name)] = float(angle)

    return pose


def run_trial(pose):
    """
    Reset NAO to the given pose, run physics for STEPS_PER_TRIAL steps,
    return (survived_seconds, final_height, final_qw, min_height, min_qw).
    """
    # Full physics reset to clear residual forces
    world.reset()

    # Teleport to spawn position and joint pose
    nao.set_world_pose(
        position=np.array([0.0, 0.0, args.spawn_height]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),  # identity quaternion (wxyz? xyzw?)
    )
    nao.set_joint_positions(pose)
    nao.set_joint_velocities(np.zeros(num_dof))

    min_height = 999.0
    min_qw = 1.0
    survived_steps = 0
    final_height = args.spawn_height
    final_qw = 1.0

    for step_i in range(STEPS_PER_TRIAL):
        world.step(render=False)

        if step_i % SAMPLE_INTERVAL == 0:
            try:
                pos, quat = nao.get_world_pose()
                height = float(pos[2])
                # Isaac Sim quaternion convention: [x, y, z, w]
                qw = float(quat[3]) if len(quat) == 4 else float(quat[0])
                # Use absolute value — qw and -qw represent the same rotation
                qw_abs = abs(qw)

                final_height = height
                final_qw = qw_abs
                min_height = min(min_height, height)
                min_qw = min(min_qw, qw_abs)

                if height < MIN_HEIGHT or qw_abs < MIN_QW:
                    survived_steps = step_i
                    break
            except Exception:
                survived_steps = step_i
                break
        survived_steps = step_i

    survived_seconds = survived_steps * PHYSICS_DT
    return survived_seconds, final_height, final_qw, min_height, min_qw


# ── Run sweep ─────────────────────────────────────────────────────────────────
results = []

for idx, (hip, knee, ankle) in enumerate(all_combos):
    pose = build_pose(hip, knee, ankle)
    survived, final_h, final_qw, min_h, min_qw = run_trial(pose)

    # Geometric sum as a quick sanity indicator
    geo_sum = hip + knee + ankle

    results.append({
        "hip": hip,
        "knee": knee,
        "ankle": ankle,
        "geo_sum": geo_sum,
        "survived_s": round(survived, 2),
        "final_h": round(final_h, 4),
        "final_qw": round(final_qw, 4),
        "min_h": round(min_h, 4),
        "min_qw": round(min_qw, 4),
        "stable": survived >= (args.seconds - 0.1),  # survived ≈ full duration
    })

    # Progress indicator
    tag = "✅ STABLE" if results[-1]["stable"] else "❌"
    if (idx + 1) % 25 == 0 or results[-1]["stable"]:
        print(f"  [{idx+1}/{total}]  hip={hip:+3d}° knee={knee:+3d}° ankle={ankle:+3d}°  "
              f"sum={geo_sum:+3d}°  survived={survived:.1f}s  h={final_h:.3f}  qw={final_qw:.3f}  {tag}")

# ── Sort by stability, then by how upright the final pose is ──────────────────
results.sort(key=lambda r: (-r["survived_s"], -r["min_qw"], -r["min_h"]))

# ── Save CSV ──────────────────────────────────────────────────────────────────
with open(args.output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print()
print("=" * 80)
print("RESULTS — TOP 20 MOST STABLE POSES")
print("=" * 80)
print(f"{'Rank':>4}  {'Hip':>5}  {'Knee':>5}  {'Ankle':>6}  {'Sum':>5}  "
      f"{'Surv(s)':>7}  {'FinalH':>7}  {'FinalQw':>7}  {'MinH':>7}  {'MinQw':>7}  {'Status'}")
print("-" * 90)

stable_count = sum(1 for r in results if r["stable"])

for i, r in enumerate(results[:20]):
    tag = "✅ STABLE" if r["stable"] else "❌ fell"
    print(f"{i+1:>4}  {r['hip']:>+4}°  {r['knee']:>+4}°  {r['ankle']:>+5}°  {r['geo_sum']:>+4}°  "
          f"{r['survived_s']:>7.2f}  {r['final_h']:>7.4f}  {r['final_qw']:>7.4f}  "
          f"{r['min_h']:>7.4f}  {r['min_qw']:>7.4f}  {tag}")

print()
print(f"Total stable poses: {stable_count} / {total}")
print(f"Full results saved to: {args.output}")

# ── Recommend best pose ───────────────────────────────────────────────────────
if stable_count > 0:
    best = results[0]
    print()
    print("=" * 80)
    print("🏆 RECOMMENDED POSE")
    print("=" * 80)
    print(f"  HipPitch:   {best['hip']:+d}°")
    print(f"  KneePitch:  {best['knee']:+d}°")
    print(f"  AnklePitch: {best['ankle']:+d}°")
    print(f"  Geometric sum: {best['geo_sum']:+d}°")
    print(f"  Survived: {best['survived_s']:.2f}s / {args.seconds:.1f}s")
    print(f"  Final height: {best['final_h']:.4f}m")
    print(f"  Final qw: {best['final_qw']:.4f}")
    print()
    print("  Next step: run  confirm_standing.py  with these values to watch visually.")
    print("=" * 80)
else:
    print()
    print("⚠️  No stable pose found! Possible issues:")
    print("    - Spawn height may need adjustment (try --spawn-height 0.33 or 0.34)")
    print("    - Grid may be too coarse (try --fine)")
    print("    - URDF friction/inertia may need checking")
    print("=" * 80)

# Cleanup
try:
    simulation_app.close()
except:
    pass
