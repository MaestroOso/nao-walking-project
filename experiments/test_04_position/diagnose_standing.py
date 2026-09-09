"""
diagnose_standing.py — Find WHY NAO Falls & Fix It (v2)
=========================================================

v2 fixes: Use prim.IsA() for typed schemas (RevoluteJoint),
          not prim.HasAPI() which is only for applied API schemas.

DIAGNOSIS
---------
The stability sweep showed ALL 175 joint-angle combinations fail
within 0.7 seconds. The problem is NOT the pose — it's the physics:
joints have no stiffness, so NAO is a ragdoll.

USAGE
-----
  python diagnose_standing.py --headless        # Run all diagnostics
  python diagnose_standing.py --test stiffness  # Stiffness only (fastest)
"""

import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--test", default="all",
                    choices=["all", "stiffness", "friction", "spawn", "inspect"],
                    help="Which diagnostic to run")
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--seconds", type=float, default=8.0)
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

print("=" * 80)
print("NAO STANDING DIAGNOSTICS v2")
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
from pxr import UsdPhysics, Usd, Sdf

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

print(f"NAO loaded: {num_dof} DOFs at {nao_prim_path}")
print(f"DOFs: {dof_names}")
print()


# ── Helper: Build home pose ──────────────────────────────────────────────────
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


# ── Helper: Run a standing trial ──────────────────────────────────────────────
def run_standing_trial(label, spawn_height=0.335, seconds=None):
    if seconds is None:
        seconds = args.seconds

    PHYSICS_DT = 1.0 / 120.0
    total_steps = int(seconds / PHYSICS_DT)
    sample_interval = int(0.5 / PHYSICS_DT)

    world.reset()

    pose = build_home_pose()
    nao.set_world_pose(
        position=np.array([0.0, 0.0, spawn_height]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    nao.set_joint_positions(pose)
    nao.set_joint_velocities(np.zeros(num_dof))

    print(f"\n--- {label} ---")
    print(f"  Spawn height: {spawn_height:.3f}m | Duration: {seconds:.1f}s")
    print(f"  {'Time':>6}  {'Height':>8}  {'qw':>8}  {'Status'}")

    fell = False
    last_height = spawn_height

    for step_i in range(total_steps):
        world.step(render=not args.headless)

        if step_i % sample_interval == 0:
            try:
                pos, quat = nao.get_world_pose()
                height = float(pos[2])
                qw_abs = abs(float(quat[3]))
                t = step_i * PHYSICS_DT
                last_height = height

                if height < 0.15 or qw_abs < 0.7:
                    status = "❌ FELL"
                    fell = True
                    print(f"  {t:>5.1f}s  {height:>8.4f}  {qw_abs:>8.4f}  {status}")
                    break
                elif height < 0.25:
                    status = "⚠️  LOW"
                else:
                    status = "✅ OK"
                print(f"  {t:>5.1f}s  {height:>8.4f}  {qw_abs:>8.4f}  {status}")
            except Exception as e:
                print(f"  ERROR: {e}")

    if not fell:
        print(f"  ✅ SURVIVED {seconds:.1f}s! (final height: {last_height:.4f}m)")
    return not fell, last_height


# ══════════════════════════════════════════════════════════════════════════════
# Find all revolute joint prims using IsA() (correct for typed schemas)
# ══════════════════════════════════════════════════════════════════════════════
def find_revolute_joints():
    """
    Find all revolute joints under the NAO prim.
    Uses prim.IsA(UsdPhysics.RevoluteJoint) — correct for typed schemas.
    """
    joints = []
    root = stage.GetPrimAtPath(nao_prim_path)
    if not root:
        print("  ⚠️ Could not find NAO root prim, searching entire stage...")
        root = stage.GetPseudoRoot()

    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joints.append(prim)

    return joints


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 1: Inspect current joint drive settings
# ══════════════════════════════════════════════════════════════════════════════
def inspect_joint_drives():
    print("=" * 80)
    print("DIAGNOSTIC: Current Joint Drive Properties")
    print("=" * 80)

    joint_prims = find_revolute_joints()
    print(f"\n  Found {len(joint_prims)} revolute joints\n")

    has_any_drive = False

    print(f"  {'Joint Name':35s} {'Stiffness':>12} {'Damping':>12} {'Target':>12}")
    print("  " + "-" * 75)

    for prim in joint_prims:
        name = prim.GetName()

        stiffness = None
        damping = None
        target = None

        # Check if angular drive exists
        try:
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive:
                s_attr = drive.GetStiffnessAttr()
                d_attr = drive.GetDampingAttr()
                t_attr = drive.GetTargetPositionAttr()
                if s_attr and s_attr.HasValue():
                    stiffness = s_attr.Get()
                if d_attr and d_attr.HasValue():
                    damping = d_attr.Get()
                if t_attr and t_attr.HasValue():
                    target = t_attr.Get()
                if stiffness is not None and stiffness > 0:
                    has_any_drive = True
        except Exception:
            pass

        s_str = f"{stiffness:.2f}" if stiffness is not None else "—"
        d_str = f"{damping:.2f}" if damping is not None else "—"
        t_str = f"{target:.2f}" if target is not None else "—"
        print(f"  {name:35s} {s_str:>12} {d_str:>12} {t_str:>12}")

    print()
    if not has_any_drive:
        print("  ⚠️  NO ACTIVE JOINT DRIVES!")
        print("  Joints have zero stiffness → robot is a passive ragdoll.")
        print("  This is why NAO collapses instantly regardless of pose.")
    else:
        print("  ✅ Some drives found.")

    return has_any_drive


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 2: Apply joint stiffness and test
# ══════════════════════════════════════════════════════════════════════════════
def test_joint_stiffness():
    print()
    print("=" * 80)
    print("DIAGNOSTIC: Adding Joint Stiffness (PD Position Drives)")
    print("=" * 80)
    print()
    print("  A real robot has motors holding each joint in place.")
    print("  Without stiffness, Isaac Sim treats the robot as a ragdoll.")
    print("  Testing angular drives with increasing stiffness + damping...")
    print()

    joint_prims = find_revolute_joints()
    print(f"  Found {len(joint_prims)} revolute joints to configure.\n")

    home = build_home_pose()

    # Map joint prim names to DOF indices for target angles
    joint_to_dof = {}
    for prim in joint_prims:
        name = prim.GetName()
        if name in dof_names:
            joint_to_dof[name] = dof_names.index(name)

    print(f"  Matched {len(joint_to_dof)}/{len(joint_prims)} joints to DOF indices.\n")

    # ── Test matrix ───────────────────────────────────────────────────────────
    test_configs = [
        # (stiffness, damping)
        (5.0,    0.5),
        (10.0,   1.0),
        (20.0,   2.0),
        (50.0,   5.0),
        (100.0,  10.0),
        (200.0,  10.0),
        (500.0,  20.0),
        (1000.0, 50.0),
    ]

    best_config = None
    results_summary = []

    for stiff, damp in test_configs:
        # Apply drives to all joints
        for prim in joint_prims:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(stiff)
            drive.CreateDampingAttr(damp)

            # Set drive target to home pose angle (in degrees)
            name = prim.GetName()
            if name in joint_to_dof:
                target_rad = home[joint_to_dof[name]]
                drive.CreateTargetPositionAttr(float(math.degrees(target_rad)))
            else:
                drive.CreateTargetPositionAttr(0.0)

        label = f"stiffness={stiff:>7.1f}, damping={damp:>5.1f}"
        survived, final_h = run_standing_trial(label, seconds=5.0)
        results_summary.append((stiff, damp, survived, final_h))

        if survived and best_config is None:
            best_config = (stiff, damp)
            print(f"\n  🏆 FIRST STABLE CONFIG: stiffness={stiff}, damping={damp}\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("  STIFFNESS SWEEP SUMMARY:")
    print(f"  {'Stiffness':>12} {'Damping':>10} {'Survived':>10} {'FinalH':>10}")
    print("  " + "-" * 50)
    for stiff, damp, survived, final_h in results_summary:
        tag = "✅" if survived else "❌"
        print(f"  {stiff:>10.1f}  {damp:>9.1f}  {tag:>8}  {final_h:>9.4f}")

    print()
    print("=" * 80)
    if best_config:
        stiff, damp = best_config
        print(f"✅ SOLUTION: Joint PD drives fix the problem!")
        print(f"   Minimum working: stiffness={stiff}, damping={damp}")
        print()
        print(f"   To integrate into nao_walking_env.py:")
        print(f"   1. After world.reset(), apply drives to all revolute joints")
        print(f"   2. Set drive targets = home pose angles (in degrees)")
        print(f"   3. During RL step(), update drive targets instead of set_joint_positions()")
    else:
        print(f"❌ Joint stiffness alone did not stabilize NAO up to 1000 Nm/rad.")
        print(f"   Next steps:")
        print(f"     - Run with --test inspect to dump the full prim tree")
        print(f"     - Check URDF mass/inertia values")
        print(f"     - Verify foot collision geometry contacts the ground")
    print("=" * 80)

    return best_config


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 3: Inspect friction
# ══════════════════════════════════════════════════════════════════════════════
def inspect_friction():
    print()
    print("=" * 80)
    print("DIAGNOSTIC: Friction Properties")
    print("=" * 80)

    foot_keywords = ["ankle", "foot", "sole", "toe"]
    found_any = False

    print("\n  Searching for physics materials...\n")

    for prim in stage.Traverse():
        name = prim.GetName().lower()
        path = str(prim.GetPath()).lower()

        is_foot = any(kw in name for kw in foot_keywords)
        is_ground = "ground" in name or "plane" in name

        if is_foot or is_ground:
            has_material = False
            try:
                mat_api = UsdPhysics.MaterialAPI(prim)
                sf = mat_api.GetStaticFrictionAttr()
                df = mat_api.GetDynamicFrictionAttr()
                if (sf and sf.HasValue()) or (df and df.HasValue()):
                    has_material = True
                    sf_val = sf.Get() if sf and sf.HasValue() else "unset"
                    df_val = df.Get() if df and df.HasValue() else "unset"
                    print(f"  {prim.GetPath()}")
                    print(f"    Static friction:  {sf_val}")
                    print(f"    Dynamic friction: {df_val}")
                    found_any = True
            except:
                pass

            if not has_material and is_foot:
                print(f"  {prim.GetPath()} — no physics material")

    if not found_any:
        print("  No explicit friction materials found.")
        print("  Isaac Sim uses default friction (~0.5), should be OK for standing.")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 4: Dump prim tree (debugging aid)
# ══════════════════════════════════════════════════════════════════════════════
def inspect_prim_tree():
    print()
    print("=" * 80)
    print("DIAGNOSTIC: USD Prim Tree (NAO subtree)")
    print("=" * 80)

    root = stage.GetPrimAtPath(nao_prim_path)
    count = 0
    for prim in Usd.PrimRange(root):
        depth = str(prim.GetPath()).count("/") - str(root.GetPath()).count("/")
        indent = "  " * depth
        type_name = prim.GetTypeName()

        flags = []
        if prim.IsA(UsdPhysics.RevoluteJoint):
            flags.append("REVOLUTE_JOINT")

        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  {indent}{prim.GetName()} ({type_name}){flag_str}")
        count += 1
        if count > 250:
            print("  ... (truncated)")
            break


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if args.test == "all":
    inspect_joint_drives()
    run_standing_trial("Baseline — no drives", seconds=3.0)
    inspect_friction()
    best = test_joint_stiffness()

elif args.test == "stiffness":
    inspect_joint_drives()
    best = test_joint_stiffness()

elif args.test == "friction":
    inspect_friction()

elif args.test == "spawn":
    for h in [0.30, 0.32, 0.335, 0.34, 0.35, 0.38]:
        run_standing_trial(f"Spawn height = {h:.3f}m", spawn_height=h, seconds=3.0)

elif args.test == "inspect":
    inspect_joint_drives()
    inspect_friction()
    inspect_prim_tree()

print()
print("=" * 80)
print("DIAGNOSTICS COMPLETE")
print("=" * 80)

try:
    simulation_app.close()
except:
    pass
