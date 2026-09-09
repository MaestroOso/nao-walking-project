"""
forensic_standing.py — Deep Investigation of NAO Standing Failure
===================================================================

Even 1000 Nm/rad stiffness didn't help. Every config falls to ~0.067m.
This means drives aren't working, or there's no ground contact, or
something else fundamental is wrong.

This script checks EVERYTHING:
  1. Are drives actually applied? (re-read after setting)
  2. Do drives survive world.reset()? (reset clears USD changes?)
  3. What are joint positions during the fall? (ragdoll vs fighting?)
  4. Mass/inertia of each link
  5. Collision geometry of feet
  6. Is there ground contact?
  7. Does set_joint_position_targets() work instead of USD drives?

USAGE
-----
  python forensic_standing.py --headless
"""

import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

print("=" * 80)
print("NAO FORENSIC INVESTIGATION")
print("=" * 80)
print()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
import omni.kit.commands
import isaacsim.asset.importer.urdf._urdf as urdf_module
from pxr import UsdPhysics, Usd, UsdGeom, PhysxSchema

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
print(f"DOFs: {dof_names}\n")


# ── Home pose ─────────────────────────────────────────────────────────────────
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


def find_revolute_joints():
    joints = []
    root = stage.GetPrimAtPath(nao_prim_path)
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joints.append(prim)
    return joints


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Check what the Articulation API gives us for stiffness/damping
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("TEST 1: Articulation-level stiffness and damping")
print("=" * 80)
print()
print("  The Articulation class has its own getter/setter for gains.")
print("  This may be separate from the USD DriveAPI.\n")

try:
    stiffnesses = nao.get_joint_stiffnesses()
    dampings = nao.get_joint_dampings()
    print(f"  {'DOF':35s} {'Stiffness':>12} {'Damping':>12}")
    print("  " + "-" * 62)
    for i, name in enumerate(dof_names):
        s = stiffnesses[i] if stiffnesses is not None else "N/A"
        d = dampings[i] if dampings is not None else "N/A"
        print(f"  {name:35s} {str(s):>12} {str(d):>12}")
    print()
    all_zero_s = stiffnesses is not None and np.all(stiffnesses == 0)
    all_zero_d = dampings is not None and np.all(dampings == 0)
    if all_zero_s:
        print("  ⚠️  ALL STIFFNESSES ARE ZERO!")
    if all_zero_d:
        print("  ⚠️  ALL DAMPINGS ARE ZERO!")
except Exception as e:
    print(f"  Error reading gains: {e}")
    stiffnesses = None
    dampings = None

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Set gains via Articulation API and verify
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("TEST 2: Set stiffness via Articulation API (not USD)")
print("=" * 80)
print()
print("  Using nao.set_joint_stiffnesses() and nao.set_joint_dampings()")
print("  This is the Isaac Sim Articulation-level API, which may be what")
print("  actually controls PhysX joint drives.\n")

STIFFNESS = 200.0
DAMPING = 20.0

new_stiffnesses = np.full(num_dof, STIFFNESS, dtype=np.float32)
new_dampings = np.full(num_dof, DAMPING, dtype=np.float32)

try:
    nao.set_joint_stiffnesses(new_stiffnesses)
    nao.set_joint_dampings(new_dampings)
    print(f"  Set all joints: stiffness={STIFFNESS}, damping={DAMPING}")

    # Read back to verify
    readback_s = nao.get_joint_stiffnesses()
    readback_d = nao.get_joint_dampings()
    print(f"  Readback stiffness[0]: {readback_s[0] if readback_s is not None else 'N/A'}")
    print(f"  Readback damping[0]:   {readback_d[0] if readback_d is not None else 'N/A'}")

    if readback_s is not None and readback_s[0] == STIFFNESS:
        print("  ✅ Stiffness set successfully via Articulation API!")
    else:
        print("  ⚠️  Stiffness readback doesn't match!")
except Exception as e:
    print(f"  Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Run with Articulation API gains + position targets
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("TEST 3: Standing trial with Articulation API gains")
print("=" * 80)
print()

home = build_home_pose()

# Reset and apply
world.reset()

# Re-apply stiffness after world.reset() (reset may clear it)
nao.set_joint_stiffnesses(new_stiffnesses)
nao.set_joint_dampings(new_dampings)

nao.set_world_pose(
    position=np.array([0.0, 0.0, 0.335]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
)
nao.set_joint_positions(home)
nao.set_joint_velocities(np.zeros(num_dof))

# CRITICAL: Set position targets for the PD controller
# set_joint_positions() teleports joints but doesn't set the DRIVE TARGET.
# The PD controller needs a target to drive toward.
try:
    nao.set_joint_position_targets(home)
    print("  ✅ set_joint_position_targets() accepted")
except Exception as e:
    print(f"  ⚠️  set_joint_position_targets() failed: {e}")

# Verify gains survived the reset
check_s = nao.get_joint_stiffnesses()
print(f"  Post-reset stiffness[0]: {check_s[0] if check_s is not None else 'N/A'}")

print()
print(f"  {'Time':>6}  {'Height':>8}  {'qw':>8}  {'JP[LKnee]':>12}  {'JV[LKnee]':>12}  Status")

PHYSICS_DT = 1.0 / 120.0
knee_idx = dof_names.index("LKneePitch") if "LKneePitch" in dof_names else 0

fell = False
for step_i in range(int(8.0 / PHYSICS_DT)):
    world.step(render=not args.headless)

    if step_i % int(0.5 / PHYSICS_DT) == 0:
        try:
            pos, quat = nao.get_world_pose()
            jp = nao.get_joint_positions()
            jv = nao.get_joint_velocities()
            height = float(pos[2])
            qw_abs = abs(float(quat[3]))
            knee_pos = math.degrees(jp[knee_idx]) if jp is not None else 0
            knee_vel = math.degrees(jv[knee_idx]) if jv is not None else 0
            t = step_i * PHYSICS_DT

            if height < 0.15 or qw_abs < 0.7:
                print(f"  {t:>5.1f}s  {height:>8.4f}  {qw_abs:>8.4f}  {knee_pos:>+11.1f}°  {knee_vel:>+11.1f}°/s  ❌ FELL")
                fell = True
                break
            else:
                print(f"  {t:>5.1f}s  {height:>8.4f}  {qw_abs:>8.4f}  {knee_pos:>+11.1f}°  {knee_vel:>+11.1f}°/s  ✅")
        except Exception as e:
            print(f"  ERROR: {e}")

if not fell:
    print("  ✅ SURVIVED 8 SECONDS!")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Try multiple stiffness levels via Articulation API
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("TEST 4: Stiffness sweep via Articulation API")
print("=" * 80)
print()

test_configs = [
    (50.0,   5.0),
    (200.0,  20.0),
    (500.0,  50.0),
    (1000.0, 100.0),
    (2000.0, 200.0),
    (5000.0, 500.0),
]

for stiff, damp in test_configs:
    world.reset()

    s_arr = np.full(num_dof, stiff, dtype=np.float32)
    d_arr = np.full(num_dof, damp, dtype=np.float32)

    nao.set_joint_stiffnesses(s_arr)
    nao.set_joint_dampings(d_arr)

    nao.set_world_pose(
        position=np.array([0.0, 0.0, 0.335]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    nao.set_joint_positions(home)
    nao.set_joint_velocities(np.zeros(num_dof))
    nao.set_joint_position_targets(home)

    # Run 5 seconds
    survived = True
    final_h = 0.335
    final_knee = 0.0

    for step_i in range(int(5.0 / PHYSICS_DT)):
        world.step(render=False)

        if step_i % int(1.0 / PHYSICS_DT) == 0:
            try:
                p, q = nao.get_world_pose()
                jp = nao.get_joint_positions()
                h = float(p[2])
                qw = abs(float(q[3]))
                final_h = h
                final_knee = math.degrees(jp[knee_idx]) if jp is not None else 0
                if h < 0.15 or qw < 0.7:
                    survived = False
                    break
            except:
                survived = False
                break

    tag = "✅" if survived else "❌"
    print(f"  stiffness={stiff:>7.1f}  damping={damp:>6.1f}  {tag}  "
          f"h={final_h:.4f}  knee={final_knee:+.1f}°")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: Mass properties
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("TEST 5: Link mass properties")
print("=" * 80)
print()

total_mass = 0.0
root = stage.GetPrimAtPath(nao_prim_path)
for prim in Usd.PrimRange(root):
    mass_api = UsdPhysics.MassAPI(prim)
    mass_attr = mass_api.GetMassAttr()
    if mass_attr and mass_attr.HasValue():
        mass = mass_attr.Get()
        if mass > 0:
            total_mass += mass
            if mass > 0.05:  # only print significant links
                print(f"  {prim.GetName():35s}  mass={mass:.4f} kg")

print(f"\n  Total robot mass: {total_mass:.3f} kg")
print(f"  (Real NAO V4/V5 is about 5.4 kg)")

if total_mass < 1.0:
    print("  ⚠️  Mass seems too low!")
elif total_mass > 20.0:
    print("  ⚠️  Mass seems too high!")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: Collision geometry on feet
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("TEST 6: Foot collision geometry")
print("=" * 80)
print()

foot_links = ["l_ankle", "r_ankle", "l_sole", "r_sole", "LAnklePitch", "RAnklePitch",
              "LAnkleRoll", "RAnkleRoll", "l_foot", "r_foot"]

for prim in Usd.PrimRange(root):
    name = prim.GetName().lower()
    prim_type = prim.GetTypeName()

    # Check for collision-related types on foot prims
    is_foot = any(fn in name for fn in ["ankle", "foot", "sole"])
    is_collision = "collision" in name or "Collision" in prim_type

    if is_foot or (is_collision and "ankle" in str(prim.GetPath()).lower()):
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        print(f"  {str(prim.GetPath()):60s}  type={prim_type:20s}  collision={has_collision}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: Fixed base test (sanity check — does robot hold pose if pinned?)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("TEST 7: Quick ground-contact check")
print("=" * 80)
print()
print("  Spawning at height=0.50m and tracking the fall...")
print("  If robot passes through ground, collision geometry is broken.\n")

world.reset()
nao.set_joint_stiffnesses(np.full(num_dof, 1000.0, dtype=np.float32))
nao.set_joint_dampings(np.full(num_dof, 100.0, dtype=np.float32))
nao.set_world_pose(
    position=np.array([0.0, 0.0, 0.50]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
)
nao.set_joint_positions(home)
nao.set_joint_velocities(np.zeros(num_dof))
nao.set_joint_position_targets(home)

for step_i in range(int(3.0 / PHYSICS_DT)):
    world.step(render=False)
    if step_i % int(0.25 / PHYSICS_DT) == 0:
        try:
            p, q = nao.get_world_pose()
            h = float(p[2])
            t = step_i * PHYSICS_DT
            below = "⚠️ BELOW GROUND" if h < 0.01 else ""
            print(f"  t={t:.2f}s  height={h:.4f}m  {below}")
        except:
            pass


print()
print("=" * 80)
print("FORENSIC INVESTIGATION COMPLETE")
print("=" * 80)
print()
print("Key questions answered:")
print("  - Are Articulation API stiffnesses being applied?")
print("  - Do gains survive world.reset()?")
print("  - Are joint positions changing (fighting) or passive (ragdoll)?")
print("  - Is the robot's mass reasonable?")
print("  - Do feet have collision geometry?")
print("  - Does the robot contact the ground or fall through?")
print()

try:
    simulation_app.close()
except:
    pass
