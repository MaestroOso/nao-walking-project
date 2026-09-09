"""
calibrate_standing.py — Precise Spawn Height + Pose Calibration
=================================================================

ROOT CAUSE (finally clear):
  1. USD DriveAPI changes don't survive world.reset() — damping stays 0.0
  2. BUT: the existing stiffness=35810 DOES hold joints rigid
  3. The robot falls as a RIGID BLOCK — joints hold perfectly
  4. At t=0.008s height=0.3397, at t=0.092s height=0.3511 (BOUNCED UP)
     → feet spawn INSIDE the ground, initial impulse tips the robot

STRATEGY:
  A) Try STRAIGHT LEG pose (0,0,0) — simplest geometry, flat feet
  B) Calibrate spawn height precisely — find where feet just touch ground
  C) Try setting damping on the USD BEFORE creating the Articulation
     (before world.reset() freezes the properties)
  D) Try the get_articulation_controller() API for setting gains

USAGE:
  python calibrate_standing.py --headless
"""

import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--output", default="calibrate_results.txt")
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
L.log("NAO SPAWN CALIBRATION")
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

# ── Import URDF ───────────────────────────────────────────────────────────────
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
stage = world.stage

# ══════════════════════════════════════════════════════════════════════════════
# CRITICAL: Set damping BEFORE world.reset() / Articulation creation
# The hypothesis is that world.reset() reads USD state and bakes it into PhysX.
# If we set damping before that, it should persist.
# ══════════════════════════════════════════════════════════════════════════════
L.log("Setting damping on USD joints BEFORE Articulation.initialize()...")

DAMPING = 1000.0  # Nm·s/rad

joint_prims = []
root = stage.GetPrimAtPath(nao_prim_path)
for prim in Usd.PrimRange(root):
    if prim.IsA(UsdPhysics.RevoluteJoint):
        joint_prims.append(prim)

L.log(f"  Found {len(joint_prims)} revolute joints")

for prim in joint_prims:
    name = prim.GetName()
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")

    if drive:
        # Modify existing drive — set damping
        damp_attr = drive.GetDampingAttr()
        if damp_attr:
            damp_attr.Set(DAMPING)
        else:
            drive.CreateDampingAttr(DAMPING)
    else:
        # Create drive for broken joints
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(35810.0)
        drive.CreateDampingAttr(DAMPING)
        drive.CreateMaxForceAttr(3.0)

L.log(f"  Applied damping={DAMPING} to all joints")
L.log("")

# NOW create the Articulation and reset
nao = world.scene.add(Articulation(nao_prim_path, name="nao"))
world.reset()

dof_names = nao.dof_names
num_dof = nao.num_dof
PHYSICS_DT = 1.0 / 120.0

L.log(f"NAO: {num_dof} DOFs")
L.log("")

# Check if damping persisted
props = nao.dof_properties
L.log("DOF properties AFTER world.reset():")
L.log(f"  DOF 0 ({dof_names[0]}): stiffness={props[0]['stiffness']:.1f}, damping={props[0]['damping']:.1f}, driveMode={props[0]['driveMode']}")
L.log(f"  DOF 3 ({dof_names[3]}): stiffness={props[3]['stiffness']:.1f}, damping={props[3]['damping']:.1f}, driveMode={props[3]['driveMode']}")
L.log(f"  DOF 14 ({dof_names[14]}): stiffness={props[14]['stiffness']:.1f}, damping={props[14]['damping']:.1f}, driveMode={props[14]['driveMode']}")
L.log("")

damping_works = props[0]['damping'] > 0
L.log(f"  Damping persisted through world.reset(): {'YES' if damping_works else 'NO'}")
L.log("")

# ── Index lookups ─────────────────────────────────────────────────────────────
knee_idx = dof_names.index("LKneePitch")
ankle_idx = dof_names.index("LAnklePitch")
hip_idx = dof_names.index("LHipPitch")

# ── Pose builders ─────────────────────────────────────────────────────────────
def build_pose(hip=0, knee=0, ankle=0):
    pose = np.zeros(num_dof, dtype=np.float32)
    joints = {
        "LHipPitch": hip, "RHipPitch": hip,
        "LKneePitch": knee, "RKneePitch": knee,
        "LAnklePitch": ankle, "RAnklePitch": ankle,
        "LShoulderPitch": 84, "RShoulderPitch": 84,
        "LShoulderRoll": 10, "RShoulderRoll": -10,
        "LElbowYaw": -70, "RElbowYaw": 70,
        "LElbowRoll": -25, "RElbowRoll": 25,
    }
    for name, deg in joints.items():
        if name in dof_names:
            pose[dof_names.index(name)] = math.radians(deg)
    return pose


# ── Standing test ─────────────────────────────────────────────────────────────
def run_test(pose, spawn_height, seconds, label=""):
    """Test standing with given pose and spawn height. No world.reset()."""

    nao.set_world_pose(
        position=np.array([0.0, 0.0, spawn_height]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    nao.set_joint_positions(pose)
    nao.set_joint_velocities(np.zeros(num_dof))

    L.log(f"  --- {label} (spawn_h={spawn_height:.4f}) ---")
    L.log(f"  {'Step':>6}  {'Time':>7}  {'Height':>8}  {'qw':>8}  {'Knee':>8}  "
          f"{'Ankle':>8}  {'Hip':>8}  Note")
    L.log("  " + "-" * 78)

    total_steps = int(seconds / PHYSICS_DT)
    fell = False
    nan_seen = False
    final_h = spawn_height
    min_h = spawn_height
    max_h = spawn_height

    for step_i in range(total_steps):
        action = ArticulationAction(joint_positions=pose)
        nao.apply_action(action)
        world.step(render=not args.headless)

        log_this = (step_i < 120 and step_i % 10 == 0) or \
                   (step_i >= 120 and step_i % int(0.5 / PHYSICS_DT) == 0)

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
                    min_h = min(min_h, h)
                    max_h = max(max_h, h)

                note = ""
                if np.isnan(h):
                    note = "NaN!"
                    nan_seen = True
                elif h < 0.10:
                    note = "GROUND"
                    fell = True
                elif h > spawn_height + 0.02:
                    note = "BOUNCE!"
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
        verdict = "LOW"

    L.log(f"  Result: {verdict} (final={final_h:.4f}, min={min_h:.4f}, max={max_h:.4f})")
    L.log("")
    return verdict, final_h


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Straight-leg pose at various spawn heights
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("TEST 1: STRAIGHT-LEG POSE (all leg joints = 0°)")
L.log("  Testing spawn heights from 0.30 to 0.38")
L.log("=" * 80)
L.log("")

straight = build_pose(hip=0, knee=0, ankle=0)

# Need world.reset() before first test to clear state
world.reset()

for h in [0.30, 0.31, 0.32, 0.33, 0.335, 0.34, 0.345, 0.35, 0.36, 0.38]:
    run_test(straight, h, seconds=3.0, label=f"Straight legs, h={h}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Bent-leg home pose at various spawn heights
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("TEST 2: BENT-LEG HOME POSE (hip=-25, knee=+45, ankle=-10)")
L.log("  Testing spawn heights from 0.28 to 0.36")
L.log("=" * 80)
L.log("")

bent = build_pose(hip=-25, knee=45, ankle=-10)

world.reset()

for h in [0.28, 0.29, 0.30, 0.31, 0.315, 0.32, 0.325, 0.33, 0.335, 0.34, 0.35, 0.36]:
    run_test(bent, h, seconds=3.0, label=f"Bent legs, h={h}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Try ArticulationController for setting gains
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("TEST 3: ArticulationController API")
L.log("=" * 80)
L.log("")

try:
    controller = nao.get_articulation_controller()
    L.log(f"  Controller type: {type(controller)}")
    L.log(f"  Controller methods: {[m for m in dir(controller) if not m.startswith('_')]}")

    # Check for gain-setting methods
    has_set_gains = hasattr(controller, 'set_gains')
    has_switch = hasattr(controller, 'switch_control_mode')
    L.log(f"  has set_gains: {has_set_gains}")
    L.log(f"  has switch_control_mode: {has_switch}")

    if has_set_gains:
        L.log("  Attempting to set gains via controller...")
        kps = np.full(num_dof, 35810.0, dtype=np.float32)
        kds = np.full(num_dof, 1000.0, dtype=np.float32)
        try:
            controller.set_gains(kps, kds)
            L.log("  set_gains() succeeded!")

            # Verify
            props_after = nao.dof_properties
            L.log(f"  After set_gains: stiff={props_after[0]['stiffness']:.1f}, damp={props_after[0]['damping']:.1f}")
        except Exception as e:
            L.log(f"  set_gains() error: {e}")

    if has_switch:
        L.log("  Checking control modes...")
        try:
            controller.switch_control_mode("position")
            L.log("  Switched to position mode")
        except Exception as e:
            L.log(f"  switch_control_mode error: {e}")

except Exception as e:
    L.log(f"  get_articulation_controller() error: {e}")

L.log("")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: If controller.set_gains worked, re-run standing tests
# ══════════════════════════════════════════════════════════════════════════════
props_check = nao.dof_properties
if props_check[0]['damping'] > 0:
    L.log("=" * 80)
    L.log("TEST 4: Standing with DAMPING (gains were set!)")
    L.log("=" * 80)
    L.log("")

    world.reset()

    # Re-apply gains after reset
    try:
        controller = nao.get_articulation_controller()
        kps = np.full(num_dof, 35810.0, dtype=np.float32)
        kds = np.full(num_dof, 1000.0, dtype=np.float32)
        controller.set_gains(kps, kds)
        L.log(f"  Gains re-applied after reset")
        p = nao.dof_properties
        L.log(f"  Verify: stiff={p[0]['stiffness']:.1f}, damp={p[0]['damping']:.1f}")
    except:
        pass

    # Test straight pose
    for h in [0.32, 0.335, 0.34, 0.35]:
        run_test(straight, h, seconds=5.0, label=f"With damping, straight, h={h}")

    # Test bent pose
    for h in [0.30, 0.315, 0.33]:
        run_test(bent, h, seconds=5.0, label=f"With damping, bent, h={h}")


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
