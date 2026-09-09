"""
forensic_v2.py — Discover the actual API and fix joint drives
===============================================================

KEY FINDINGS SO FAR:
  - Articulation is 'SingleArticulation' — no set_joint_stiffnesses()
  - USD DriveAPI.Apply() was tried but didn't work
  - Hypothesis: world.reset() clears USD drive changes

PLAN:
  1. Discover what methods ARE available on the Articulation object
  2. Apply USD drives AFTER world.reset() (not before)
  3. Step physics once, THEN check if drives are still set
  4. If drives persist, test standing

USAGE:
  python forensic_v2.py --headless
"""

import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

print("=" * 80)
print("NAO FORENSIC v2 — API Discovery + Drive Fix")
print("=" * 80)
print()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
import omni.kit.commands
import isaacsim.asset.importer.urdf._urdf as urdf_module
from pxr import UsdPhysics, Usd, Gf

world = World(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
world.scene.add_default_ground_plane()

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

nao_prim_path = result[1]
nao = world.scene.add(Articulation(nao_prim_path, name="nao"))
world.reset()

dof_names = nao.dof_names
num_dof = nao.num_dof
stage = world.stage

print(f"NAO: {num_dof} DOFs at {nao_prim_path}")
print(f"Type of nao: {type(nao)}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Discover available methods on the Articulation object
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("PART 1: Available methods on Articulation object")
print("=" * 80)
print()

methods = [m for m in dir(nao) if not m.startswith('_')]

# Group by relevance
joint_methods = [m for m in methods if 'joint' in m.lower() or 'dof' in m.lower()]
drive_methods = [m for m in methods if 'drive' in m.lower() or 'stiff' in m.lower()
                 or 'damp' in m.lower() or 'gain' in m.lower()]
pos_methods = [m for m in methods if 'position' in m.lower() or 'target' in m.lower()]
effort_methods = [m for m in methods if 'effort' in m.lower() or 'force' in m.lower()
                  or 'torque' in m.lower()]

print("  JOINT-related methods:")
for m in joint_methods:
    print(f"    {m}")

print("\n  DRIVE/STIFFNESS/DAMPING-related methods:")
for m in drive_methods:
    print(f"    {m}")
if not drive_methods:
    print("    (none found)")

print("\n  POSITION/TARGET-related methods:")
for m in pos_methods:
    print(f"    {m}")

print("\n  EFFORT/FORCE/TORQUE-related methods:")
for m in effort_methods:
    print(f"    {m}")
if not effort_methods:
    print("    (none found)")

print("\n  ALL methods (full list):")
for m in sorted(methods):
    print(f"    {m}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Check what ArticulationView or DC interface gives us
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("PART 2: Try alternative interfaces for joint control")
print("=" * 80)
print()

# Try the dynamic control (DC) interface
print("  Attempting DC (Dynamic Control) interface...")
try:
    from omni.isaac.dynamic_control import _dynamic_control
    dc = _dynamic_control.acquire_dynamic_control_interface()
    art_handle = dc.get_articulation(nao_prim_path)
    print(f"  DC articulation handle: {art_handle}")

    if art_handle != 0 and art_handle is not None:
        num_joints_dc = dc.get_articulation_joint_count(art_handle)
        print(f"  DC joint count: {num_joints_dc}")

        # Check a joint's properties
        for i in range(min(5, num_joints_dc)):
            joint = dc.get_articulation_joint(art_handle, i)
            name = dc.get_joint_name(joint)
            joint_type = dc.get_joint_type(joint)

            # Try to get/set drive properties
            try:
                props = dc.get_joint_drive_properties(joint)
                print(f"    Joint {name}: drive_props = {props}")
            except Exception as e:
                print(f"    Joint {name}: get_drive_properties error: {e}")
    else:
        print("  DC: Could not get articulation handle")
except ImportError:
    print("  DC interface not available (omni.isaac.dynamic_control not found)")
except Exception as e:
    print(f"  DC interface error: {e}")

# Try ArticulationView
print()
print("  Attempting ArticulationView interface...")
try:
    from omni.isaac.core.articulations import ArticulationView
    print("  ArticulationView imported successfully")

    # Check if ArticulationView has stiffness methods
    av_methods = [m for m in dir(ArticulationView)
                  if 'stiff' in m.lower() or 'damp' in m.lower() or 'gain' in m.lower()
                  or 'drive' in m.lower()]
    print(f"  ArticulationView drive-related methods: {av_methods}")
except ImportError:
    print("  ArticulationView not available")
except Exception as e:
    print(f"  ArticulationView error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Apply USD drives AFTER world.reset(), verify, then test
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("PART 3: USD DriveAPI — apply AFTER reset, verify persistence")
print("=" * 80)
print()

home = build_home_pose() if 'build_home_pose' in dir() else np.zeros(num_dof)

def build_home_pose(hip=-25, knee=45, ankle=-10):
    pose = np.zeros(num_dof, dtype=np.float32)
    joints = {
        "LHipPitch": math.radians(hip), "RHipPitch": math.radians(hip),
        "LKneePitch": math.radians(knee), "RKneePitch": math.radians(knee),
        "LAnklePitch": math.radians(ankle), "RAnklePitch": math.radians(ankle),
        "LShoulderPitch": math.radians(84), "RShoulderPitch": math.radians(84),
        "LShoulderRoll": math.radians(10), "RShoulderRoll": math.radians(-10),
        "LElbowYaw": math.radians(-70), "RElbowYaw": math.radians(70),
        "LElbowRoll": math.radians(-25), "RElbowRoll": math.radians(25),
    }
    for name, angle in joints.items():
        if name in dof_names:
            pose[dof_names.index(name)] = float(angle)
    return pose

home = build_home_pose()

def find_revolute_joints():
    joints = []
    root = stage.GetPrimAtPath(nao_prim_path)
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joints.append(prim)
    return joints

joint_prims = find_revolute_joints()
print(f"  Found {len(joint_prims)} revolute joints\n")

# Step A: world.reset()
print("  Step A: Calling world.reset()...")
world.reset()

# Step B: Set pose FIRST
print("  Step B: Setting world pose and joint positions...")
nao.set_world_pose(
    position=np.array([0.0, 0.0, 0.335]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
)
nao.set_joint_positions(home)
nao.set_joint_velocities(np.zeros(num_dof))

# Step C: Apply drives AFTER reset
STIFFNESS = 500.0
DAMPING = 50.0

print(f"  Step C: Applying USD DriveAPI (stiffness={STIFFNESS}, damping={DAMPING})...")

for prim in joint_prims:
    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(STIFFNESS)
    drive.CreateDampingAttr(DAMPING)

    # Set target = home pose angle for this joint
    name = prim.GetName()
    if name in dof_names:
        idx = dof_names.index(name)
        target_deg = math.degrees(home[idx])
        drive.CreateTargetPositionAttr(target_deg)
    else:
        drive.CreateTargetPositionAttr(0.0)

# Step D: Verify drives are set (re-read from USD)
print("  Step D: Verifying drives...\n")
sample_joints = joint_prims[:5]
for prim in sample_joints:
    name = prim.GetName()
    try:
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        s = drive.GetStiffnessAttr().Get()
        d = drive.GetDampingAttr().Get()
        t = drive.GetTargetPositionAttr().Get()
        print(f"    {name:25s}  stiffness={s}  damping={d}  target={t:.1f}°")
    except Exception as e:
        print(f"    {name:25s}  ERROR: {e}")

# Step E: Do ONE physics step, then re-check drives
print("\n  Step E: One physics step, then re-check...")
world.step(render=False)

for prim in sample_joints:
    name = prim.GetName()
    try:
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        s = drive.GetStiffnessAttr().Get()
        d = drive.GetDampingAttr().Get()
        print(f"    {name:25s}  stiffness={s}  damping={d}  (after 1 step)")
    except Exception as e:
        print(f"    {name:25s}  ERROR: {e}")

# Step F: Check joint positions after a few steps — are they drifting?
print("\n  Step F: Monitoring joint positions over 2 seconds...")
print(f"  {'Time':>6}  {'Height':>8}  {'LKnee(°)':>10}  {'LAnkle(°)':>10}  {'LHip(°)':>10}")

knee_idx = dof_names.index("LKneePitch") if "LKneePitch" in dof_names else 0
ankle_idx = dof_names.index("LAnklePitch") if "LAnklePitch" in dof_names else 0
hip_idx = dof_names.index("LHipPitch") if "LHipPitch" in dof_names else 0

PHYSICS_DT = 1.0 / 120.0

for step_i in range(int(2.0 / PHYSICS_DT)):
    world.step(render=not args.headless)

    if step_i % int(0.25 / PHYSICS_DT) == 0:
        try:
            pos, quat = nao.get_world_pose()
            jp = nao.get_joint_positions()
            h = float(pos[2])
            t = (step_i + 1) * PHYSICS_DT  # +1 because we already did 1 step

            knee_deg = math.degrees(jp[knee_idx]) if jp is not None else 0
            ankle_deg = math.degrees(jp[ankle_idx]) if jp is not None else 0
            hip_deg = math.degrees(jp[hip_idx]) if jp is not None else 0

            print(f"  {t:>5.2f}s  {h:>8.4f}  {knee_deg:>+9.1f}°  {ankle_deg:>+9.1f}°  {hip_deg:>+9.1f}°")
        except Exception as e:
            print(f"  ERROR: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: Try set_joint_efforts (direct torque) as alternative
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("PART 4: Try set_joint_efforts() — direct torque holding")
print("=" * 80)
print()

# Check if set_joint_efforts exists
has_efforts = hasattr(nao, 'set_joint_efforts')
print(f"  nao.set_joint_efforts exists: {has_efforts}")

has_targets = hasattr(nao, 'set_joint_position_targets')
print(f"  nao.set_joint_position_targets exists: {has_targets}")

has_vel_targets = hasattr(nao, 'set_joint_velocity_targets')
print(f"  nao.set_joint_velocity_targets exists: {has_vel_targets}")

# If set_joint_position_targets exists, it REQUIRES drives to be configured
if has_targets:
    print("\n  set_joint_position_targets IS available!")
    print("  This sends position commands to PD drives (requires stiffness > 0).")
    print("  Testing with USD drives already applied from Part 3...")

    # Drives should still be set from Part 3
    try:
        nao.set_joint_position_targets(home)
        print("  ✅ set_joint_position_targets() succeeded!")
    except Exception as e:
        print(f"  ⚠️ Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: Extended standing test (drives should still be applied from Part 3)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("PART 5: Extended standing test (8 seconds)")
print("=" * 80)
print()

# Full reset and reapply everything cleanly
world.reset()

# Pose
nao.set_world_pose(
    position=np.array([0.0, 0.0, 0.335]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
)
nao.set_joint_positions(home)
nao.set_joint_velocities(np.zeros(num_dof))

# Apply drives AFTER reset
for prim in joint_prims:
    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(STIFFNESS)
    drive.CreateDampingAttr(DAMPING)
    name = prim.GetName()
    if name in dof_names:
        idx = dof_names.index(name)
        drive.CreateTargetPositionAttr(float(math.degrees(home[idx])))
    else:
        drive.CreateTargetPositionAttr(0.0)

# Also try set_joint_position_targets if available
if has_targets:
    try:
        nao.set_joint_position_targets(home)
    except:
        pass

print(f"  Stiffness={STIFFNESS}, Damping={DAMPING}")
print(f"  {'Time':>6}  {'Height':>8}  {'qw':>8}  {'LKnee':>10}  Status")

fell = False
for step_i in range(int(8.0 / PHYSICS_DT)):
    world.step(render=not args.headless)

    if step_i % int(0.5 / PHYSICS_DT) == 0:
        try:
            pos, quat = nao.get_world_pose()
            jp = nao.get_joint_positions()
            h = float(pos[2])
            qw = abs(float(quat[3]))
            t = step_i * PHYSICS_DT
            knee_deg = math.degrees(jp[knee_idx]) if jp is not None else 0

            if h < 0.15 or qw < 0.7:
                print(f"  {t:>5.1f}s  {h:>8.4f}  {qw:>8.4f}  {knee_deg:>+9.1f}°  ❌ FELL")
                fell = True
                break
            else:
                print(f"  {t:>5.1f}s  {h:>8.4f}  {qw:>8.4f}  {knee_deg:>+9.1f}°  ✅")
        except Exception as e:
            print(f"  ERROR: {e}")

if not fell:
    print(f"\n  🏆 NAO SURVIVED 8 SECONDS!")
else:
    print(f"\n  ❌ Still falling. The drives may not be having effect on PhysX.")
    print(f"     Next step: try the DC (Dynamic Control) interface to set drives,")
    print(f"     or check if 'effort' mode works instead of 'force' mode.")

print()
print("=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)

try:
    simulation_app.close()
except:
    pass
