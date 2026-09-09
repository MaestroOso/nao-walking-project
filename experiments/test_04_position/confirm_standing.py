"""
confirm_standing.py — Visually Verify NAO Standing Stability
==============================================================

PURPOSE
-------
After running stability_sweep.py to find candidate poses, use this script
to VISUALLY confirm that NAO remains standing under free physics.

The script:
  1. Loads NAO with the specified joint angles (GUI mode — you can watch).
  2. Activates physics with NO control input (joints are free).
  3. Runs for --seconds (default 10s) while printing live telemetry.
  4. Reports final verdict: STABLE or FELL.

USAGE
-----
  # Use default angles (modify DEFAULT values below, or pass via CLI):
  python confirm_standing.py

  # Specify exact angles from the sweep results:
  python confirm_standing.py --hip -20 --knee 40 --ankle -20

  # Run longer to be really sure:
  python confirm_standing.py --hip -20 --knee 40 --ankle -20 --seconds 15

  # Run headless (faster, no GUI):
  python confirm_standing.py --hip -20 --knee 40 --ankle -20 --headless
"""

import math
import argparse
import numpy as np
from datetime import datetime

# ── CLI arguments ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Visually confirm NAO standing pose")
parser.add_argument("--hip",   type=float, default=-25,  help="HipPitch in degrees (default: -25)")
parser.add_argument("--knee",  type=float, default=45,   help="KneePitch in degrees (default: 45)")
parser.add_argument("--ankle", type=float, default=-10,  help="AnklePitch in degrees (default: -10)")
parser.add_argument("--spawn-height", type=float, default=0.335, help="Spawn Z height in meters")
parser.add_argument("--seconds", type=float, default=10.0, help="Duration to run physics")
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf", help="Path to NAO URDF")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
args = parser.parse_args()

print("=" * 80)
print("NAO STANDING CONFIRMATION TEST")
print("=" * 80)
print(f"  HipPitch:    {args.hip:+.1f}°")
print(f"  KneePitch:   {args.knee:+.1f}°")
print(f"  AnklePitch:  {args.ankle:+.1f}°")
print(f"  Geo sum:     {args.hip + args.knee + args.ankle:+.1f}°")
print(f"  Spawn height: {args.spawn_height:.3f}m")
print(f"  Duration:    {args.seconds:.1f}s")
print(f"  Headless:    {args.headless}")
print("=" * 80)
print()

# ── Isaac Sim startup ─────────────────────────────────────────────────────────
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless,
    "width": 1280,
    "height": 720,
})

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

# ── Build the joint pose ──────────────────────────────────────────────────────
pose = np.zeros(num_dof, dtype=np.float32)

JOINT_ANGLES = {
    # Legs — the three angles being tested
    "LHipPitch":   math.radians(args.hip),
    "RHipPitch":   math.radians(args.hip),
    "LKneePitch":  math.radians(args.knee),
    "RKneePitch":  math.radians(args.knee),
    "LAnklePitch": math.radians(args.ankle),
    "RAnklePitch": math.radians(args.ankle),
    # Arms — lowered to sides (same as env)
    "LShoulderPitch": math.radians(84),
    "RShoulderPitch": math.radians(84),
    "LShoulderRoll":  math.radians(10),
    "RShoulderRoll":  math.radians(-10),
    "LElbowYaw":      math.radians(-70),
    "RElbowYaw":      math.radians(70),
    "LElbowRoll":     math.radians(-25),
    "RElbowRoll":     math.radians(25),
}

for name, angle in JOINT_ANGLES.items():
    if name in dof_names:
        pose[dof_names.index(name)] = float(angle)
    else:
        print(f"  ⚠️ Joint '{name}' not found in DOFs!")

# ── Set initial state ─────────────────────────────────────────────────────────
nao.set_world_pose(
    position=np.array([0.0, 0.0, args.spawn_height]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
)
nao.set_joint_positions(pose)
nao.set_joint_velocities(np.zeros(num_dof))

# ── Run physics ───────────────────────────────────────────────────────────────
PHYSICS_DT = 1.0 / 120.0
total_steps = int(args.seconds / PHYSICS_DT)
print_interval = int(0.5 / PHYSICS_DT)  # print every 0.5s

MIN_HEIGHT = 0.15
MIN_QW = 0.7

fell = False
fell_time = None

print()
print(f"{'Time':>6}  {'Height':>8}  {'qw':>8}  {'Pitch°':>8}  {'vx':>8}  {'vy':>8}  {'Status'}")
print("-" * 70)

for step_i in range(total_steps):
    world.step(render=not args.headless)

    if step_i % print_interval == 0:
        try:
            pos, quat = nao.get_world_pose()
            lv = nao.get_linear_velocity()

            height = float(pos[2])
            # quat layout from Isaac: [x, y, z, w]
            qx, qy, qz, qw = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            qw_abs = abs(qw)

            # Compute pitch angle (rotation around Y axis) from quaternion
            # pitch = asin(2*(qw*qy - qz*qx))
            sin_pitch = 2.0 * (qw * qy - qz * qx)
            sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
            pitch_deg = math.degrees(math.asin(sin_pitch))

            vx = float(lv[0]) if lv is not None else 0.0
            vy = float(lv[1]) if lv is not None else 0.0

            t = step_i * PHYSICS_DT

            if height < MIN_HEIGHT or qw_abs < MIN_QW:
                status = "⚠️  FALLING"
                if not fell:
                    fell = True
                    fell_time = t
            elif abs(pitch_deg) > 15:
                status = "⚠️  LEANING"
            else:
                status = "✅ OK"

            print(f"{t:>5.1f}s  {height:>8.4f}  {qw_abs:>8.4f}  {pitch_deg:>+7.1f}°  "
                  f"{vx:>+7.3f}  {vy:>+7.3f}  {status}")

        except Exception as e:
            t = step_i * PHYSICS_DT
            print(f"{t:>5.1f}s  ---- ERROR: {e} ----")

# ── Final verdict ─────────────────────────────────────────────────────────────
print()
print("=" * 80)
if fell:
    print(f"❌ VERDICT: NAO FELL at t={fell_time:.2f}s")
    print(f"   This pose is NOT stable enough for RL training.")
else:
    print(f"✅ VERDICT: NAO REMAINED STANDING for {args.seconds:.1f}s")
    print(f"   This pose is suitable as the RL episode reset pose!")
    print()
    print(f"   Recommended _HOME_POSE_JOINTS values:")
    print(f"     HipPitch:   {args.hip:+.1f}°  ({math.radians(args.hip):+.4f} rad)")
    print(f"     KneePitch:  {args.knee:+.1f}°  ({math.radians(args.knee):+.4f} rad)")
    print(f"     AnklePitch: {args.ankle:+.1f}°  ({math.radians(args.ankle):+.4f} rad)")
print("=" * 80)

# Cleanup
try:
    simulation_app.close()
except:
    pass
