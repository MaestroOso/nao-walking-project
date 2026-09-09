"""
add_collisions_v2.py — Add Collision Geometry (handles USD instancing)
========================================================================

The collision Xforms under l_ankle/r_ankle are instance proxies — we
can't author geometry under them. Instead, we add collision boxes
directly to the link prims, or we find the actual prototype prim and
modify that.

Strategy: Add collision cubes as NEW children of the link prims
(l_ankle, r_ankle) rather than under their /collisions/ child.

USAGE:
  python add_collisions_v2.py --headless
  python add_collisions_v2.py            # GUI
"""

import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--output", default="collision_v2_results.txt")
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
L.log("NAO COLLISION FIX v2")
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
# STEP 1: Check instancing and find writable prims
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("STEP 1: Check USD instancing")
L.log("=" * 80)
L.log("")

root = stage.GetPrimAtPath(nao_prim_path)

# Check which prims are instanceable or instance proxies
for prim in Usd.PrimRange(root):
    name = prim.GetName()
    path = str(prim.GetPath())
    if any(kw in name.lower() for kw in ["ankle", "sole", "foot"]):
        is_instance = prim.IsInstance()
        is_proxy = prim.IsInstanceProxy()
        is_instanceable = prim.IsInstanceable()
        L.log(f"  {path}")
        L.log(f"    type={prim.GetTypeName()}  instance={is_instance}  "
              f"proxy={is_proxy}  instanceable={is_instanceable}")

L.log("")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Disable instancing on the NAO root so we can author freely
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("STEP 2: Disable instancing and add collision boxes")
L.log("=" * 80)
L.log("")

# Disable instanceable on the root and all children
for prim in Usd.PrimRange(root):
    if prim.IsInstanceable():
        prim.SetInstanceable(False)
        L.log(f"  Disabled instancing on: {prim.GetPath()}")

# Also check if any prims are instances and clear them
for prim in Usd.PrimRange(root):
    if prim.IsInstance():
        prim.SetInstanceable(False)
        L.log(f"  Cleared instance flag on: {prim.GetPath()}")

L.log("")

# NAO foot dimensions
FOOT_LENGTH = 0.16   # 16cm
FOOT_WIDTH  = 0.09   # 9cm
FOOT_HEIGHT = 0.02   # 2cm

# Offset from ankle to foot center
FOOT_OFFSET_X = 0.04    # foot extends forward
FOOT_OFFSET_Z = -0.045  # sole is below ankle

feet = [
    ("l_ankle", "left_foot_box"),
    ("r_ankle", "right_foot_box"),
]

boxes_created = 0

for link_name, box_name in feet:
    # Try multiple paths
    paths_to_try = [
        f"{nao_prim_path}/{link_name}/collisions/{box_name}",
        f"{nao_prim_path}/{link_name}/{box_name}",
    ]

    created = False
    for box_path in paths_to_try:
        try:
            L.log(f"  Trying: {box_path}")

            # Check if parent exists and is writable
            parent_path = "/".join(box_path.split("/")[:-1])
            parent = stage.GetPrimAtPath(parent_path)
            if not parent or not parent.IsValid():
                L.log(f"    Parent not found: {parent_path}")
                continue

            if parent.IsInstanceProxy():
                L.log(f"    Parent is instance proxy, skipping")
                continue

            box_prim = UsdGeom.Cube.Define(stage, box_path)
            box_prim.CreateSizeAttr(1.0)

            xform = UsdGeom.Xformable(box_prim.GetPrim())
            xform.ClearXformOpOrder()
            translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(FOOT_OFFSET_X, 0.0, FOOT_OFFSET_Z))
            scale_op = xform.AddScaleOp()
            scale_op.Set(Gf.Vec3d(FOOT_LENGTH, FOOT_WIDTH, FOOT_HEIGHT))

            UsdPhysics.CollisionAPI.Apply(box_prim.GetPrim())

            L.log(f"    ✅ Created collision box at {box_path}")
            L.log(f"       Size: {FOOT_LENGTH}x{FOOT_WIDTH}x{FOOT_HEIGHT}m")
            L.log(f"       Offset: ({FOOT_OFFSET_X}, 0, {FOOT_OFFSET_Z})")
            boxes_created += 1
            created = True
            break

        except Exception as e:
            L.log(f"    ❌ Failed: {e}")
            continue

    if not created:
        # Last resort: create at the root level as a separate prim
        box_path = f"{nao_prim_path}/{box_name}"
        try:
            L.log(f"  Last resort: creating at {box_path}")
            box_prim = UsdGeom.Cube.Define(stage, box_path)
            box_prim.CreateSizeAttr(1.0)

            xform = UsdGeom.Xformable(box_prim.GetPrim())
            xform.ClearXformOpOrder()
            translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(FOOT_OFFSET_X, 0.0, FOOT_OFFSET_Z))
            scale_op = xform.AddScaleOp()
            scale_op.Set(Gf.Vec3d(FOOT_LENGTH, FOOT_WIDTH, FOOT_HEIGHT))

            UsdPhysics.CollisionAPI.Apply(box_prim.GetPrim())

            L.log(f"    ✅ Created at root level: {box_path}")
            boxes_created += 1
        except Exception as e:
            L.log(f"    ❌ Root-level also failed: {e}")

L.log(f"\n  Total boxes created: {boxes_created}")
L.log("")

# Verify the boxes exist
L.log("  Verification — scanning for collision cubes:")
for prim in Usd.PrimRange(root):
    if prim.GetTypeName() == "Cube" or "foot" in prim.GetName().lower() or "box" in prim.GetName().lower():
        has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
        L.log(f"    {prim.GetPath()}  type={prim.GetTypeName()}  collision={has_col}")

L.log("")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Create articulation and test
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("STEP 3: Standing tests")
L.log("=" * 80)
L.log("")

nao = world.scene.add(Articulation(nao_prim_path, name="nao"))
world.reset()

dof_names = nao.dof_names
num_dof = nao.num_dof
PHYSICS_DT = 1.0 / 120.0

knee_idx = dof_names.index("LKneePitch")
ankle_idx = dof_names.index("LAnklePitch")
hip_idx = dof_names.index("LHipPitch")

# Configure gains
controller = nao.get_articulation_controller()
kps = np.full(num_dof, 35810.0, dtype=np.float32)
kds = np.full(num_dof, 1000.0, dtype=np.float32)
controller.set_gains(kps, kds)
controller.set_max_efforts(np.full(num_dof, 100.0, dtype=np.float32))

L.log(f"  NAO: {num_dof} DOFs, gains configured")
L.log("")

# Poses
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

    L.log(f"  --- {label} ---")
    L.log(f"  {'Step':>6}  {'Time':>7}  {'Height':>8}  {'qw':>8}  {'Hip':>8}  "
          f"{'Knee':>8}  {'Ankle':>8}  Note")
    L.log("  " + "-" * 78)

    fell = False
    final_h = spawn_height

    total_steps = int(seconds / PHYSICS_DT)

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


# Run tests
for h in [0.35, 0.38, 0.40]:
    run_test(straight, h, seconds=5.0, label=f"Straight, h={h}")

for h in [0.35, 0.38]:
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
