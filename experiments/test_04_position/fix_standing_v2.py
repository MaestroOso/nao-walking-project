"""
fix_standing_v2.py — Fix DOF Properties via USD API
======================================================

dof_properties is READ-ONLY on SingleArticulation.
We must set stiffness/damping via the USD DriveAPI on each joint prim.

The key insight: the URDF importer already CREATED angular drives on
most joints (driveMode=2, stiffness=35810). We just need to:
  1. Add damping to each drive
  2. Fix broken joints (RHipYawPitch, all fingers)
  3. Set drive targets to the home pose
  4. Use apply_action(joint_positions=home) for control

USAGE:
  python fix_standing_v2.py --headless
"""

import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--output", default="fix_v2_results.txt")
parser.add_argument("--seconds", type=float, default=10.0)
args = parser.parse_args()


class Logger:
    def __init__(self, path):
        self.lines = []
        self.f = open(path, 'w')

    def log(self, msg=""):
        self.lines.append(msg)
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        self.f.close()

    def dump(self):
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        for line in self.lines:
            print(line)


L = Logger(args.output)
L.log("=" * 80)
L.log("NAO STANDING FIX v2 — USD DriveAPI")
L.log("=" * 80)
L.log("")

# ── Isaac Sim ─────────────────────────────────────────────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
import omni.kit.commands
import isaacsim.asset.importer.urdf._urdf as urdf_module
from pxr import UsdPhysics, Usd

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
PHYSICS_DT = 1.0 / 120.0

L.log(f"NAO: {num_dof} DOFs at {nao_prim_path}")
L.log("")

# ── Index lookups ─────────────────────────────────────────────────────────────
knee_idx = dof_names.index("LKneePitch")
ankle_idx = dof_names.index("LAnklePitch")
hip_idx = dof_names.index("LHipPitch")

# ── Home pose ─────────────────────────────────────────────────────────────────
home = np.zeros(num_dof, dtype=np.float32)
joint_angles_deg = {
    "LHipPitch": -25, "RHipPitch": -25,
    "LKneePitch": 45, "RKneePitch": 45,
    "LAnklePitch": -10, "RAnklePitch": -10,
    "LShoulderPitch": 84, "RShoulderPitch": 84,
    "LShoulderRoll": 10, "RShoulderRoll": -10,
    "LElbowYaw": -70, "RElbowYaw": 70,
    "LElbowRoll": -25, "RElbowRoll": 25,
}
for name, deg in joint_angles_deg.items():
    if name in dof_names:
        home[dof_names.index(name)] = math.radians(deg)


# ══════════════════════════════════════════════════════════════════════════════
# Find all revolute joints and apply fixes via USD
# ══════════════════════════════════════════════════════════════════════════════
def find_revolute_joints():
    joints = []
    root = stage.GetPrimAtPath(nao_prim_path)
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joints.append(prim)
    return joints


def apply_drive_fixes(damping_value):
    """
    Apply damping to all joints, fix broken joints, set targets.
    Must be called BEFORE world.reset() or after — we'll test both.
    """
    joint_prims = find_revolute_joints()

    for prim in joint_prims:
        name = prim.GetName()

        # Get or create angular drive
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")

        # Ensure drive type is "force" (position control)
        type_attr = drive.GetTypeAttr()
        if not type_attr or not type_attr.HasValue():
            drive.CreateTypeAttr("force")
        else:
            type_attr.Set("force")

        # Set stiffness (keep existing 35810 or set it for broken joints)
        stiff_attr = drive.GetStiffnessAttr()
        if stiff_attr and stiff_attr.HasValue():
            current_stiff = stiff_attr.Get()
            if current_stiff == 0.0:
                stiff_attr.Set(35810.0)
        else:
            drive.CreateStiffnessAttr(35810.0)

        # Set damping — THE KEY FIX
        damp_attr = drive.GetDampingAttr()
        if damp_attr and damp_attr.HasValue():
            damp_attr.Set(damping_value)
        else:
            drive.CreateDampingAttr(damping_value)

        # Set target position (in degrees for angular drives)
        target_attr = drive.GetTargetPositionAttr()
        if name in dof_names:
            idx = dof_names.index(name)
            target_deg = math.degrees(home[idx])
        else:
            target_deg = 0.0

        if target_attr and target_attr.HasValue():
            target_attr.Set(target_deg)
        else:
            drive.CreateTargetPositionAttr(target_deg)

        # Set max force (for broken joints that had maxEffort=0)
        max_force_attr = drive.GetMaxForceAttr()
        if max_force_attr and max_force_attr.HasValue():
            if max_force_attr.Get() == 0.0:
                max_force_attr.Set(3.0)
        else:
            drive.CreateMaxForceAttr(3.0)

    return len(joint_prims)


# ══════════════════════════════════════════════════════════════════════════════
# Standing test function
# ══════════════════════════════════════════════════════════════════════════════
def run_test(damping, seconds, label=""):
    """Apply fixes, reset, run physics, report."""

    # Apply drive fixes BEFORE world.reset
    n_fixed = apply_drive_fixes(damping)

    # Reset world
    world.reset()

    # Re-apply fixes AFTER reset (in case reset clears them)
    apply_drive_fixes(damping)

    # Set initial pose
    nao.set_world_pose(
        position=np.array([0.0, 0.0, 0.335]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    nao.set_joint_positions(home)
    nao.set_joint_velocities(np.zeros(num_dof))

    # Verify dof_properties after our changes
    props = nao.dof_properties
    actual_damp = props[0]['damping']
    actual_stiff = props[0]['stiffness']

    L.log(f"  --- {label}damping={damping} (actual: stiff={actual_stiff:.1f}, damp={actual_damp:.1f}) ---")
    L.log(f"  {'Step':>6}  {'Time':>7}  {'Height':>8}  {'qw':>8}  {'Knee':>8}  "
          f"{'Ankle':>8}  {'Hip':>8}  Note")
    L.log("  " + "-" * 78)

    fell = False
    nan_seen = False
    final_h = 0.335

    total_steps = int(seconds / PHYSICS_DT)
    print_interval = int(0.5 / PHYSICS_DT)

    for step_i in range(total_steps):
        # Position control via apply_action
        action = ArticulationAction(joint_positions=home)
        nao.apply_action(action)
        world.step(render=not args.headless)

        # Log at intervals
        log_this = (step_i < 120 and step_i % 20 == 0) or \
                   (step_i >= 120 and step_i % print_interval == 0)

        if log_this:
            try:
                pos, quat = nao.get_world_pose()
                jp = nao.get_joint_positions()

                h = float(pos[2]) if pos is not None else float('nan')
                qw = float(quat[3]) if quat is not None else float('nan')
                k = math.degrees(float(jp[knee_idx])) if jp is not None else float('nan')
                a = math.degrees(float(jp[ankle_idx])) if jp is not None else float('nan')
                hp = math.degrees(float(jp[hip_idx])) if jp is not None else float('nan')
                t = (step_i + 1) * PHYSICS_DT

                if not np.isnan(h):
                    final_h = h

                note = ""
                if np.isnan(h):
                    note = "NaN!"
                    nan_seen = True
                elif h < 0.10:
                    note = "GROUND"
                    fell = True
                elif h < 0.20:
                    note = "LOW"
                elif h > 0.28:
                    note = "OK"

                L.log(f"  {step_i+1:>6}  {t:>6.3f}s  {h:>8.4f}  {qw:>+8.4f}  "
                      f"{k:>+7.1f}°  {a:>+7.1f}°  {hp:>+7.1f}°  {note}")
            except Exception as e:
                L.log(f"  {step_i+1:>6}  ERROR: {e}")

    if nan_seen:
        verdict = "NaN"
    elif fell:
        verdict = "FELL"
    elif final_h > 0.25:
        verdict = "STANDING"
    else:
        verdict = "CROUCHING"

    L.log(f"  Result: {verdict} (final_h={final_h:.4f}m)")
    L.log("")
    return verdict, final_h


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: Test damping values
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("DAMPING SWEEP")
L.log("=" * 80)
L.log("")

damping_values = [100.0, 500.0, 1000.0, 3581.0, 5000.0, 10000.0]
results = []

for damp in damping_values:
    verdict, h = run_test(damp, seconds=5.0)
    results.append((damp, verdict, h))

L.log("SWEEP SUMMARY:")
L.log(f"  {'Damping':>10}  {'Verdict':>10}  {'FinalH':>10}")
L.log("  " + "-" * 35)
for damp, verdict, h in results:
    L.log(f"  {damp:>10.1f}  {verdict:>10}  {h:>10.4f}")
L.log("")

# Find best standing config
standing = [(d, v, h) for d, v, h in results if v == "STANDING"]
if standing:
    best_damp = standing[0][0]
    L.log(f"BEST: damping={best_damp}")
    L.log("")
    L.log("=" * 80)
    L.log(f"EXTENDED CONFIRMATION ({args.seconds}s)")
    L.log("=" * 80)
    L.log("")
    run_test(best_damp, seconds=args.seconds, label="CONFIRM: ")
else:
    L.log("No STANDING config found.")
    L.log("")

    # Try with reduced stiffness — maybe 35810 is too high
    L.log("Testing with reduced stiffness + damping via USD...")
    L.log("")

    joint_prims = find_revolute_joints()
    stiff_damp_tests = [
        (1000.0, 200.0),
        (500.0,  100.0),
        (200.0,  40.0),
        (100.0,  20.0),
    ]

    for stiff, damp in stiff_damp_tests:
        # Set custom stiffness
        for prim in joint_prims:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            stiff_attr = drive.GetStiffnessAttr()
            if stiff_attr:
                stiff_attr.Set(stiff)
            else:
                drive.CreateStiffnessAttr(stiff)

        verdict, h = run_test(damp, seconds=5.0, label=f"stiff={stiff}: ")
        L.log(f"  >> stiffness={stiff}, damping={damp} → {verdict} (h={h:.4f})")

        if verdict == "STANDING":
            L.log(f"\nFOUND: stiffness={stiff}, damping={damp}")
            L.log("")
            # Confirm
            for prim in joint_prims:
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.GetStiffnessAttr().Set(stiff)
            run_test(damp, seconds=args.seconds, label="CONFIRM: ")
            break


# ── Final DOF state ───────────────────────────────────────────────────────────
L.log("")
L.log("=" * 80)
L.log("FINAL DOF PROPERTIES (after all tests)")
L.log("=" * 80)
L.log("")

props = nao.dof_properties
L.log(f"  {'DOF':>3}  {'Name':30s}  {'driveMode':>10}  {'stiffness':>12}  {'damping':>10}  {'maxEffort':>10}")
L.log("  " + "-" * 80)
for i in range(num_dof):
    L.log(f"  {i:>3}  {dof_names[i]:30s}  {props[i]['driveMode']:>10}  "
          f"{props[i]['stiffness']:>12.1f}  {props[i]['damping']:>10.1f}  "
          f"{props[i]['maxEffort']:>10.3f}")

L.log("")
L.log("=" * 80)
L.log("DONE")
L.log("=" * 80)
L.close()

try:
    simulation_app.close()
except:
    pass

L.dump()
