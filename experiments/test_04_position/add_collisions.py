"""
add_collisions.py — Add Collision Geometry to NAO's Feet
==========================================================

ROOT CAUSE FOUND:
  ALL collision prims in NAO are empty Xform nodes — NO actual geometry.
  The URDF importer created collision Xform containers but didn't populate
  them with shapes (mesh, box, sphere, etc.).
  
  NAO is a GHOST — it passes through the ground. The 0.059m "resting"
  height is just the torso lying flat with no collision resistance.

FIX:
  Add box collision shapes to the foot links (l_ankle, r_ankle).
  NAO foot is approximately 16cm x 9cm x 2cm.

USAGE:
  python add_collisions.py --headless
  python add_collisions.py            # GUI to watch
"""

import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--output", default="collision_results.txt")
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
L.log("NAO COLLISION FIX — Adding Foot Geometry")
L.log("=" * 80)
L.log("")

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
import omni.kit.commands
import isaacsim.asset.importer.urdf._urdf as urdf_module
from pxr import UsdPhysics, UsdGeom, Usd, Gf, Sdf

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

L.log(f"NAO imported at {nao_prim_path}")
L.log("")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Investigate existing collision prims — look INSIDE the Xforms
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("STEP 1: Inspect existing collision structure")
L.log("=" * 80)
L.log("")

# Look at what's inside the collision Xforms for foot links
foot_links = ["l_ankle", "r_ankle", "LAnklePitch", "RAnklePitch",
              "l_sole", "r_sole"]

root = stage.GetPrimAtPath(nao_prim_path)
for prim in Usd.PrimRange(root):
    name = prim.GetName()
    path = str(prim.GetPath())

    # Check foot-related links and their children
    if any(fl in name for fl in foot_links):
        L.log(f"  {path}  type={prim.GetTypeName()}")
        # List all children
        for child in prim.GetAllChildren():
            child_type = child.GetTypeName()
            has_col = child.HasAPI(UsdPhysics.CollisionAPI)
            L.log(f"    └─ {child.GetName()}  type={child_type}  collision={has_col}")
            # Go one more level deep
            for grandchild in child.GetAllChildren():
                gc_type = grandchild.GetTypeName()
                gc_col = grandchild.HasAPI(UsdPhysics.CollisionAPI)
                L.log(f"        └─ {grandchild.GetName()}  type={gc_type}  collision={gc_col}")
                for ggc in grandchild.GetAllChildren():
                    L.log(f"            └─ {ggc.GetName()}  type={ggc.GetTypeName()}  "
                          f"collision={ggc.HasAPI(UsdPhysics.CollisionAPI)}")

L.log("")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Add box collision shapes to both feet
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("STEP 2: Add box collision geometry to feet")
L.log("=" * 80)
L.log("")

# NAO V4/V5 foot dimensions (from documentation):
# Length: ~16cm, Width: ~9cm, Height: ~2cm
# The foot link (l_ankle/r_ankle) is at the ankle joint.
# The sole is about 4.5cm below the ankle and extends forward.
#
# From Test 6 data:
#   l_ankle position: z = -0.2627 (relative to base at z=0.335 → world z=0.0723)
#   l_sole position:  z = -0.3079 (relative to base → world z=0.0271)
#   FootBumper:       z = -0.2984, x = +0.0863
#
# So the sole is about 4.5cm below the ankle.
# The foot extends about 8.6cm forward (x) from the ankle.

FOOT_LENGTH = 0.16   # 16cm along X
FOOT_WIDTH  = 0.09   # 9cm along Y
FOOT_HEIGHT = 0.02   # 2cm along Z (thin)

# Offset from ankle joint to foot center
# Ankle is at top of foot, foot extends forward and slightly back
FOOT_OFFSET_X = 0.04   # center of foot is ~4cm forward of ankle
FOOT_OFFSET_Z = -0.045  # sole is ~4.5cm below ankle

foot_configs = [
    {
        "link": "l_ankle",
        "box_name": "foot_collision_box",
        "offset_y": 0.0,
    },
    {
        "link": "r_ankle",
        "box_name": "foot_collision_box",
        "offset_y": 0.0,
    },
]

for fc in foot_configs:
    # Find the collision Xform under this link
    collision_path = f"{nao_prim_path}/{fc['link']}/collisions"
    collision_prim = stage.GetPrimAtPath(collision_path)

    if not collision_prim:
        L.log(f"  ⚠️ Could not find {collision_path}")
        # Try to create it
        collision_path_alt = f"{nao_prim_path}/{fc['link']}"
        collision_prim = stage.GetPrimAtPath(collision_path_alt)
        if collision_prim:
            collision_path = f"{collision_path_alt}/foot_box"
        else:
            L.log(f"  ⚠️ Could not find {collision_path_alt} either!")
            continue

    # Create a box under the collisions Xform
    box_path = f"{collision_path}/{fc['box_name']}"
    L.log(f"  Creating box at: {box_path}")

    box_prim = UsdGeom.Cube.Define(stage, box_path)

    # Set the size (Cube.size is the full edge length, but we scale with xformOp)
    box_prim.CreateSizeAttr(1.0)

    # Set scale to match foot dimensions
    xform = UsdGeom.Xformable(box_prim.GetPrim())
    xform.ClearXformOpOrder()

    # Translate to correct position relative to ankle
    translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(FOOT_OFFSET_X, fc['offset_y'], FOOT_OFFSET_Z))

    # Scale to foot dimensions (Cube default size=1, so scale = desired size)
    scale_op = xform.AddScaleOp()
    scale_op.Set(Gf.Vec3d(FOOT_LENGTH, FOOT_WIDTH, FOOT_HEIGHT))

    # Add collision API
    UsdPhysics.CollisionAPI.Apply(box_prim.GetPrim())

    # Add physics material for friction
    mat_path = f"{box_path}/physics_material"
    mat_prim = stage.DefinePrim(mat_path, "Material")
    mat_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
    mat_api.CreateStaticFrictionAttr(1.0)    # high friction
    mat_api.CreateDynamicFrictionAttr(0.8)
    mat_api.CreateRestitutionAttr(0.1)       # low bounce

    # Bind material to the box
    # Use the material binding API
    try:
        from pxr import UsdShade
        bind_api = UsdShade.MaterialBindingAPI.Apply(box_prim.GetPrim())
        bind_api.Bind(UsdShade.Material(mat_prim), bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        L.log(f"  ✅ Box created: {FOOT_LENGTH}x{FOOT_WIDTH}x{FOOT_HEIGHT}m, "
              f"offset=({FOOT_OFFSET_X}, {fc['offset_y']}, {FOOT_OFFSET_Z})")
        L.log(f"     Friction: static=1.0, dynamic=0.8, restitution=0.1")
    except Exception as e:
        L.log(f"  Material binding error: {e}")
        L.log(f"  ✅ Box created (without material binding)")

L.log("")

# Also add collision to other body parts that need it (torso, shins)
# For now, focus on feet — they're the critical contact surface.

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Create articulation and test standing
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("STEP 3: Test standing with collision geometry")
L.log("=" * 80)
L.log("")

nao = world.scene.add(Articulation(nao_prim_path, name="nao"))
world.reset()

dof_names = nao.dof_names
num_dof = nao.num_dof

knee_idx = dof_names.index("LKneePitch")
ankle_idx = dof_names.index("LAnklePitch")
hip_idx = dof_names.index("LHipPitch")

# Configure gains
controller = nao.get_articulation_controller()
kps = np.full(num_dof, 35810.0, dtype=np.float32)
kds = np.full(num_dof, 1000.0, dtype=np.float32)
controller.set_gains(kps, kds)
controller.set_max_efforts(np.full(num_dof, 100.0, dtype=np.float32))

# Verify collision geometry survived
L.log("  Verifying collision prims after world.reset():")
for fc in foot_configs:
    box_path = f"{nao_prim_path}/{fc['link']}/collisions/{fc['box_name']}"
    bp = stage.GetPrimAtPath(box_path)
    if bp and bp.IsValid():
        has_col = bp.HasAPI(UsdPhysics.CollisionAPI)
        L.log(f"    {box_path}: exists={True}, collision={has_col}")
    else:
        L.log(f"    {box_path}: NOT FOUND after reset!")

L.log("")

# Home poses
home = np.zeros(num_dof, dtype=np.float32)
for name, deg in {"LHipPitch": -25, "RHipPitch": -25,
                   "LKneePitch": 45, "RKneePitch": 45,
                   "LAnklePitch": -10, "RAnklePitch": -10,
                   "LShoulderPitch": 84, "RShoulderPitch": 84,
                   "LShoulderRoll": 10, "RShoulderRoll": -10,
                   "LElbowYaw": -70, "RElbowYaw": 70,
                   "LElbowRoll": -25, "RElbowRoll": 25}.items():
    if name in dof_names:
        home[dof_names.index(name)] = math.radians(deg)

straight = np.zeros(num_dof, dtype=np.float32)
for name, deg in {"LShoulderPitch": 84, "RShoulderPitch": 84,
                   "LShoulderRoll": 10, "RShoulderRoll": -10,
                   "LElbowYaw": -70, "RElbowYaw": 70,
                   "LElbowRoll": -25, "RElbowRoll": 25}.items():
    if name in dof_names:
        straight[dof_names.index(name)] = math.radians(deg)


def run_test(pose, spawn_height, seconds, label=""):
    nao.set_world_pose(
        position=np.array([0.0, 0.0, spawn_height]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    nao.set_joint_positions(pose)
    nao.set_joint_velocities(np.zeros(num_dof))

    L.log(f"  --- {label} (h={spawn_height}) ---")
    L.log(f"  {'Step':>6}  {'Time':>7}  {'Height':>8}  {'qw':>8}  {'Hip':>8}  "
          f"{'Knee':>8}  {'Ankle':>8}  Note")
    L.log("  " + "-" * 78)

    total_steps = int(seconds / PHYSICS_DT)
    fell = False
    final_h = spawn_height

    PHYSICS_DT = 1.0 / 120.0

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
                hp = math.degrees(float(jp[hip_idx])) if jp is not None else float('nan')
                k = math.degrees(float(jp[knee_idx])) if jp is not None else float('nan')
                a = math.degrees(float(jp[ankle_idx])) if jp is not None else float('nan')
                t = (step_i + 1) * PHYSICS_DT

                if not np.isnan(h):
                    final_h = h

                note = ""
                if np.isnan(h):
                    note = "NaN!"
                elif h < 0.10:
                    note = "GROUND"
                    fell = True
                elif h > 0.28:
                    note = "OK"
                elif h > 0.20:
                    note = "low"

                L.log(f"  {step_i+1:>6}  {t:>6.3f}s  {h:>8.4f}  {qw:>+8.4f}  "
                      f"{hp:>+7.1f}°  {k:>+7.1f}°  {a:>+7.1f}°  {note}")
            except Exception as e:
                L.log(f"  ERROR: {e}")

    if final_h > 0.25:
        verdict = "STANDING"
    elif fell:
        verdict = "FELL"
    else:
        verdict = "LOW"

    L.log(f"  >> {verdict} (h={final_h:.4f}m)")
    L.log("")
    return verdict, final_h


# Test different spawn heights with straight legs
L.log("--- Straight leg tests ---")
for h in [0.335, 0.34, 0.35, 0.38]:
    run_test(straight, h, seconds=5.0, label=f"Straight, h={h}")

# Test bent legs
L.log("--- Bent leg tests ---")
for h in [0.335, 0.34, 0.35]:
    run_test(home, h, seconds=5.0, label=f"Bent, h={h}")


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
