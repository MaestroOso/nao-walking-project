"""
add_collisions_v3.py — Enable Existing + Add New Collisions BEFORE Articulation
==================================================================================

Key finding from v2:
  - Our foot boxes were created but robot still fell to 0.059m
  - FSR frames already have Cube geometry but collision=False!
  - Collision shapes must exist BEFORE Articulation creation

Strategy:
  1. Import URDF
  2. Disable all instancing
  3. Enable CollisionAPI on existing FSR cubes
  4. Add foot box collisions under l_ankle/r_ankle
  5. THEN create Articulation and world.reset()

USAGE:
  python add_collisions_v3.py --headless
"""

import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--output", default="collision_v3_results.txt")
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
L.log("NAO COLLISION FIX v3 — Before Articulation Init")
L.log("=" * 80)
L.log("")

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
import omni.kit.commands
import isaacsim.asset.importer.urdf._urdf as urdf_module
from pxr import UsdPhysics, UsdGeom, Usd, Gf

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
root = stage.GetPrimAtPath(nao_prim_path)

L.log(f"NAO imported at {nao_prim_path}")
L.log("")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Disable ALL instancing (must be done before modifying prims)
# ══════════════════════════════════════════════════════════════════════════════
L.log("STEP 1: Disable instancing...")
count = 0
for prim in Usd.PrimRange(root):
    if prim.IsInstanceable():
        prim.SetInstanceable(False)
        count += 1
L.log(f"  Disabled instancing on {count} prims")
L.log("")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Find and catalog ALL existing collision geometry
# ══════════════════════════════════════════════════════════════════════════════
L.log("STEP 2: Catalog ALL geometry prims (Cube, Mesh, Sphere, etc.)")
L.log("")

geom_types = ["Cube", "Mesh", "Sphere", "Cylinder", "Capsule"]
all_geom = []

for prim in Usd.PrimRange(root):
    ptype = prim.GetTypeName()
    if ptype in geom_types:
        has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
        path = str(prim.GetPath())

        # Get size info
        size_info = ""
        try:
            if prim.IsA(UsdGeom.Cube):
                cube = UsdGeom.Cube(prim)
                s = cube.GetSizeAttr().Get()
                # Check for scale
                xf = UsdGeom.Xformable(prim)
                ops = xf.GetOrderedXformOps()
                scale = None
                for op in ops:
                    if "scale" in str(op.GetOpType()).lower():
                        scale = op.Get()
                if scale:
                    size_info = f"cube size={s}, scale=({scale[0]:.4f}, {scale[1]:.4f}, {scale[2]:.4f})"
                else:
                    size_info = f"cube size={s}"
            elif prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                pts = mesh.GetPointsAttr().Get()
                if pts:
                    arr = np.array(pts)
                    mn, mx = arr.min(axis=0), arr.max(axis=0)
                    sz = mx - mn
                    size_info = f"mesh {len(pts)}pts bbox=({sz[0]:.4f}, {sz[1]:.4f}, {sz[2]:.4f})"
        except:
            pass

        all_geom.append((path, ptype, has_col, size_info))

        is_foot = any(kw in path.lower() for kw in ["ankle", "foot", "sole", "fsr"])
        tag = " <<< FOOT" if is_foot else ""
        col_tag = "COL" if has_col else "---"
        L.log(f"  [{col_tag}] {path}")
        L.log(f"         {ptype}  {size_info}{tag}")

L.log(f"\n  Total geometry prims: {len(all_geom)}")
L.log(f"  With CollisionAPI: {sum(1 for _, _, c, _ in all_geom if c)}")
L.log(f"  Without CollisionAPI: {sum(1 for _, _, c, _ in all_geom if not c)}")
L.log("")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Enable CollisionAPI on ALL geometry prims that don't have it
# ══════════════════════════════════════════════════════════════════════════════
L.log("STEP 3: Enable CollisionAPI on all geometry prims")
L.log("")

enabled_count = 0
for path, ptype, has_col, _ in all_geom:
    if not has_col:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            try:
                UsdPhysics.CollisionAPI.Apply(prim)
                enabled_count += 1
                is_foot = any(kw in path.lower() for kw in ["ankle", "foot", "sole", "fsr"])
                if is_foot:
                    L.log(f"  ✅ Enabled collision: {path}")
            except Exception as e:
                L.log(f"  ❌ Failed to enable: {path} — {e}")

L.log(f"\n  Enabled CollisionAPI on {enabled_count} additional prims")
L.log("")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Add foot box collisions (in case existing geometry is too small)
# ══════════════════════════════════════════════════════════════════════════════
L.log("STEP 4: Add foot box collisions")
L.log("")

FOOT_LENGTH = 0.16
FOOT_WIDTH  = 0.09
FOOT_HEIGHT = 0.02
FOOT_OFFSET_X = 0.04
FOOT_OFFSET_Z = -0.045

for link_name, box_name in [("l_ankle", "left_foot_col"),
                              ("r_ankle", "right_foot_col")]:
    box_path = f"{nao_prim_path}/{link_name}/collisions/{box_name}"
    try:
        box_prim = UsdGeom.Cube.Define(stage, box_path)
        box_prim.CreateSizeAttr(1.0)

        xform = UsdGeom.Xformable(box_prim.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(FOOT_OFFSET_X, 0.0, FOOT_OFFSET_Z))
        xform.AddScaleOp().Set(Gf.Vec3d(FOOT_LENGTH, FOOT_WIDTH, FOOT_HEIGHT))

        UsdPhysics.CollisionAPI.Apply(box_prim.GetPrim())
        L.log(f"  ✅ Created: {box_path}")
    except Exception as e:
        L.log(f"  ❌ Failed: {box_path} — {e}")

L.log("")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Also reduce ground restitution (0.8 is way too bouncy)
# ══════════════════════════════════════════════════════════════════════════════
L.log("STEP 5: Fix ground material (reduce bounce)")
L.log("")

for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MaterialAPI):
        mat = UsdPhysics.MaterialAPI(prim)
        rest = mat.GetRestitutionAttr()
        if rest and rest.HasValue() and rest.Get() > 0.3:
            old_val = rest.Get()
            rest.Set(0.1)
            L.log(f"  Fixed restitution: {prim.GetPath()} {old_val} → 0.1")

        # Also boost friction
        sf = mat.GetStaticFrictionAttr()
        df = mat.GetDynamicFrictionAttr()
        if sf and sf.HasValue():
            sf.Set(1.0)
        if df and df.HasValue():
            df.Set(0.8)
        L.log(f"  Set friction: static=1.0, dynamic=0.8")

L.log("")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: NOW create Articulation and world.reset()
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("STEP 6: Create Articulation (collision shapes already in place)")
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

L.log(f"  NAO: {num_dof} DOFs")

# Configure gains
controller = nao.get_articulation_controller()
kps = np.full(num_dof, 35810.0, dtype=np.float32)
kds = np.full(num_dof, 1000.0, dtype=np.float32)
controller.set_gains(kps, kds)
controller.set_max_efforts(np.full(num_dof, 100.0, dtype=np.float32))

# Verify gains
p = nao.dof_properties
L.log(f"  Gains: stiff={p[0]['stiffness']:.1f}, damp={p[0]['damping']:.1f}, maxEff={p[0]['maxEffort']:.1f}")
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

    for step_i in range(int(seconds / PHYSICS_DT)):
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
                if np.isnan(h): note = "NaN!"
                elif h < 0.10: note = "GROUND"; fell = True
                elif h > 0.28: note = "OK"
                elif h > 0.20: note = "low"

                L.log(f"  {step_i+1:>6}  {t:>6.3f}s  {h:>8.4f}  {qw:>+8.4f}  "
                      f"{hp:>+7.1f}°  {k:>+7.1f}°  {a:>+7.1f}°  {note}")
            except Exception as e:
                L.log(f"  ERROR: {e}")

    verdict = "STANDING" if final_h > 0.25 else ("FELL" if fell else "LOW")
    L.log(f"  >> {verdict} (h={final_h:.4f}m)")
    L.log("")
    return verdict, final_h


# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("STANDING TESTS")
L.log("=" * 80)
L.log("")

# Straight legs at multiple heights
for h in [0.35, 0.38, 0.40, 0.45]:
    run_test(straight, h, seconds=5.0, label=f"Straight, h={h}")

# Bent legs
for h in [0.35, 0.38, 0.40]:
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
