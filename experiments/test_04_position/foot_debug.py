"""
foot_debug.py — Investigate foot collision and contact
========================================================

The joints now work perfectly (gains + maxEffort confirmed).
But NAO tips over as a rigid block in 0.3s.

The issue must be foot collision geometry or ground contact:
  - Feet too small?
  - Feet not touching ground? (phasing through?)
  - Friction too low? (sliding?)
  - Contact geometry is a point instead of a surface?

This script:
  1. Dumps ALL collision geometry under each foot link
  2. Checks foot bounding box dimensions
  3. Tests with increased solver iterations (more accurate contact)
  4. Tests with a thicker ground collision box
  5. Tries disabling gravity to verify joints work in isolation

USAGE:
  python foot_debug.py --headless
"""

import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--output", default="foot_debug_results.txt")
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
L.log("NAO FOOT COLLISION DEBUG")
L.log("=" * 80)
L.log("")

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
import omni.kit.commands
import isaacsim.asset.importer.urdf._urdf as urdf_module
from pxr import UsdPhysics, UsdGeom, Usd, Gf, PhysxSchema

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

# ── Configure gains (known-working) ──────────────────────────────────────────
controller = nao.get_articulation_controller()
kps = np.full(num_dof, 35810.0, dtype=np.float32)
kds = np.full(num_dof, 1000.0, dtype=np.float32)
controller.set_gains(kps, kds)
controller.set_max_efforts(np.full(num_dof, 100.0, dtype=np.float32))

# ── Home poses ────────────────────────────────────────────────────────────────
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

knee_idx = dof_names.index("LKneePitch")
hip_idx = dof_names.index("LHipPitch")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Dump ALL collision geometry in NAO
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("TEST 1: All collision geometry in NAO")
L.log("=" * 80)
L.log("")

root = stage.GetPrimAtPath(nao_prim_path)
collision_count = 0

for prim in Usd.PrimRange(root):
    has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
    prim_type = prim.GetTypeName()
    path = str(prim.GetPath())
    name = prim.GetName()

    # Check for geometry types that indicate collision shapes
    is_geom = prim_type in ["Mesh", "Cube", "Sphere", "Cylinder", "Capsule",
                            "Cone", "UsdGeomMesh", "UsdGeomCube", "UsdGeomSphere",
                            "UsdGeomCylinder", "UsdGeomCapsule"]

    if has_collision or ("collision" in name.lower()) or ("collider" in name.lower()):
        collision_count += 1

        # Get geometry extent/size
        extent_str = "?"
        try:
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get()
                if points:
                    pts = np.array(points)
                    bbox_min = pts.min(axis=0)
                    bbox_max = pts.max(axis=0)
                    size = bbox_max - bbox_min
                    extent_str = f"mesh {len(points)}pts, size=({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f})m"
            elif prim.IsA(UsdGeom.Sphere):
                sphere = UsdGeom.Sphere(prim)
                r = sphere.GetRadiusAttr().Get()
                extent_str = f"sphere r={r}"
            elif prim.IsA(UsdGeom.Cube):
                cube = UsdGeom.Cube(prim)
                s = cube.GetSizeAttr().Get()
                extent_str = f"cube size={s}"
            elif prim.IsA(UsdGeom.Cylinder):
                cyl = UsdGeom.Cylinder(prim)
                r = cyl.GetRadiusAttr().Get()
                h = cyl.GetHeightAttr().Get()
                extent_str = f"cylinder r={r}, h={h}"
            elif prim.IsA(UsdGeom.Capsule):
                cap = UsdGeom.Capsule(prim)
                r = cap.GetRadiusAttr().Get()
                h = cap.GetHeightAttr().Get()
                extent_str = f"capsule r={r}, h={h}"
        except:
            pass

        # Check if this is in the foot area
        is_foot = any(kw in path.lower() for kw in
                      ["ankle", "foot", "sole", "toe", "l_sole", "r_sole"])
        foot_tag = " <<< FOOT" if is_foot else ""

        L.log(f"  [{collision_count}] {path}")
        L.log(f"       type={prim_type}  collision={has_collision}  {extent_str}{foot_tag}")

L.log(f"\n  Total collision prims: {collision_count}")
L.log("")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Check NAO's total height from feet to head in initial pose
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("TEST 2: NAO body dimensions")
L.log("=" * 80)
L.log("")

# Get all link positions
world.reset()
controller.set_gains(kps, kds)
controller.set_max_efforts(np.full(num_dof, 100.0, dtype=np.float32))

nao.set_world_pose(
    position=np.array([0.0, 0.0, 0.5]),  # spawn high
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
)
nao.set_joint_positions(home)
nao.set_joint_velocities(np.zeros(num_dof))

# Step once to settle
world.step(render=False)

# Enumerate body links and their positions
L.log("  Link positions (spawned at z=0.5):")
body_count = nao.num_bodies
L.log(f"  Number of bodies: {body_count}")

# Get the base position
base_pos, _ = nao.get_world_pose()
L.log(f"  Base (torso) position: [{base_pos[0]:.4f}, {base_pos[1]:.4f}, {base_pos[2]:.4f}]")
L.log("")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Gravity disabled — do joints hold?
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("TEST 3: Gravity disabled — joint test")
L.log("=" * 80)
L.log("")

nao.disable_gravity()
L.log("  Gravity DISABLED")

world.reset()
controller.set_gains(kps, kds)
controller.set_max_efforts(np.full(num_dof, 100.0, dtype=np.float32))

nao.disable_gravity()

nao.set_world_pose(
    position=np.array([0.0, 0.0, 0.335]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
)
nao.set_joint_positions(home)
nao.set_joint_velocities(np.zeros(num_dof))

L.log(f"  {'Time':>7}  {'Height':>8}  {'qw':>8}  {'Hip':>8}  {'Knee':>8}  Note")
L.log("  " + "-" * 55)

for step_i in range(int(3.0 / PHYSICS_DT)):
    action = ArticulationAction(joint_positions=home)
    nao.apply_action(action)
    world.step(render=False)

    if step_i % int(0.5 / PHYSICS_DT) == 0:
        try:
            pos, quat = nao.get_world_pose()
            jp = nao.get_joint_positions()
            h = float(pos[2])
            qw = float(quat[3])
            hip = math.degrees(float(jp[hip_idx]))
            knee = math.degrees(float(jp[knee_idx]))
            t = (step_i + 1) * PHYSICS_DT
            L.log(f"  {t:>6.2f}s  {h:>8.4f}  {qw:>+8.4f}  {hip:>+7.1f}°  {knee:>+7.1f}°  "
                  f"{'OK' if h > 0.3 else 'BAD'}")
        except Exception as e:
            L.log(f"  ERROR: {e}")

L.log("")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Increased solver iterations for better contact
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("TEST 4: Increased solver iterations")
L.log("=" * 80)
L.log("")

nao.enable_gravity()

# Increase solver iterations on the articulation
try:
    current_pos_iters = nao.get_solver_position_iteration_count()
    current_vel_iters = nao.get_solver_velocity_iteration_count()
    L.log(f"  Current solver iterations: pos={current_pos_iters}, vel={current_vel_iters}")

    nao.set_solver_position_iteration_count(64)
    nao.set_solver_velocity_iteration_count(64)
    L.log(f"  Set solver iterations: pos=64, vel=64")
except Exception as e:
    L.log(f"  Error setting solver iterations: {e}")

world.reset()
controller.set_gains(kps, kds)
controller.set_max_efforts(np.full(num_dof, 100.0, dtype=np.float32))
nao.set_solver_position_iteration_count(64)
nao.set_solver_velocity_iteration_count(64)

nao.set_world_pose(
    position=np.array([0.0, 0.0, 0.335]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
)
nao.set_joint_positions(home)
nao.set_joint_velocities(np.zeros(num_dof))

L.log(f"  {'Time':>7}  {'Height':>8}  {'qw':>8}  {'Hip':>8}  {'Knee':>8}  Note")
L.log("  " + "-" * 55)

for step_i in range(int(5.0 / PHYSICS_DT)):
    action = ArticulationAction(joint_positions=home)
    nao.apply_action(action)
    world.step(render=False)

    log_this = (step_i < 60 and step_i % 10 == 0) or \
               (step_i >= 60 and step_i % int(0.5 / PHYSICS_DT) == 0)
    if log_this:
        try:
            pos, quat = nao.get_world_pose()
            jp = nao.get_joint_positions()
            h = float(pos[2])
            qw = float(quat[3])
            hip = math.degrees(float(jp[hip_idx]))
            knee = math.degrees(float(jp[knee_idx]))
            t = (step_i + 1) * PHYSICS_DT
            note = "OK" if h > 0.28 else ("GROUND" if h < 0.10 else "low")
            L.log(f"  {t:>6.2f}s  {h:>8.4f}  {qw:>+8.4f}  {hip:>+7.1f}°  {knee:>+7.1f}°  {note}")
        except:
            pass

L.log("")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: Check friction on ground and feet via physics scene
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("TEST 5: Physics scene and material properties")
L.log("=" * 80)
L.log("")

# Search for physics materials
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MaterialAPI):
        mat = UsdPhysics.MaterialAPI(prim)
        sf_attr = mat.GetStaticFrictionAttr()
        df_attr = mat.GetDynamicFrictionAttr()
        rest_attr = mat.GetRestitutionAttr()

        sf = sf_attr.Get() if sf_attr and sf_attr.HasValue() else "unset"
        df = df_attr.Get() if df_attr and df_attr.HasValue() else "unset"
        rest = rest_attr.Get() if rest_attr and rest_attr.HasValue() else "unset"

        L.log(f"  Material: {prim.GetPath()}")
        L.log(f"    static_friction={sf}, dynamic_friction={df}, restitution={rest}")

# Check physics scene settings
for prim in stage.Traverse():
    if prim.GetTypeName() == "PhysicsScene":
        L.log(f"\n  PhysicsScene: {prim.GetPath()}")
        for attr in prim.GetAttributes():
            if attr.HasValue():
                L.log(f"    {attr.GetName()}: {attr.Get()}")

L.log("")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: Measure actual foot-ground distance at spawn
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("TEST 6: Foot position at spawn (where are the feet?)")
L.log("=" * 80)
L.log("")

world.reset()
controller.set_gains(kps, kds)
controller.set_max_efforts(np.full(num_dof, 100.0, dtype=np.float32))

# Try different poses and check base height
for pose_name, pose, spawn_h in [("straight", np.zeros(num_dof, dtype=np.float32), 0.335),
                                   ("bent", home, 0.335)]:
    nao.set_world_pose(
        position=np.array([0.0, 0.0, spawn_h]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    nao.set_joint_positions(pose)

    # Don't step physics — just check initial positions
    base_pos, _ = nao.get_world_pose()
    L.log(f"  {pose_name} pose, spawn_h={spawn_h}:")
    L.log(f"    Base position: z={float(base_pos[2]):.4f}m")

    # Walk the kinematic tree to find foot link positions
    for prim in Usd.PrimRange(root):
        name = prim.GetName().lower()
        if any(kw in name for kw in ["sole", "foot", "ankle"]):
            if prim.IsA(UsdGeom.Xformable):
                xform = UsdGeom.Xformable(prim)
                world_xform = xform.ComputeLocalToWorldTransform(0)
                translation = world_xform.ExtractTranslation()
                L.log(f"    {prim.GetName()}: pos=({translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f})")

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
