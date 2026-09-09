"""
standing_final.py — Complete Standing Fix
==========================================

Based on full URDF analysis:
  - Total mass: 5.195 kg
  - Foot: 10.1cm long (7cm front, 3cm rear of ankle), 4.5cm below ankle
  - FSR boxes in URDF: 1cm x 1cm x 0.5cm (tiny)
  - Isaac Sim imported FSR cubes as size=1.0 (wrong! should be 0.01)
  - Existing mesh collisions were never enabled (CollisionAPI missing)
  - Robot IS statically stable with straight legs (CoM within support polygon)

FIX (all done BEFORE Articulation creation):
  1. Disable instancing
  2. Enable CollisionAPI on foot mesh collisions
  3. Add proper foot sole boxes (full foot surface, not tiny FSR points)
  4. Fix ground restitution
  5. Set damping via USD before world.reset()
  6. Configure gains via ArticulationController after reset

USAGE:
  python standing_final.py              # GUI — watch NAO stand!
  python standing_final.py --headless   # Fast test
"""

import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--output", default="standing_final_results.txt")
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
L.log("NAO STANDING — FINAL FIX")
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
# FIX 1: Disable instancing
# ══════════════════════════════════════════════════════════════════════════════
L.log("FIX 1: Disable instancing")
for prim in Usd.PrimRange(root):
    if prim.IsInstanceable():
        prim.SetInstanceable(False)
L.log("  Done")

# ══════════════════════════════════════════════════════════════════════════════
# FIX 2: Enable CollisionAPI on foot collision meshes ONLY
#         (not on visual meshes — those would cause double collisions)
# ══════════════════════════════════════════════════════════════════════════════
L.log("FIX 2: Enable CollisionAPI on foot collision meshes")

foot_collision_paths = [
    # The actual foot meshes (under /collisions/, not /visuals/)
    f"{nao_prim_path}/l_ankle/collisions",
    f"{nao_prim_path}/r_ankle/collisions",
    f"{nao_prim_path}/LAnklePitch/collisions",
    f"{nao_prim_path}/RAnklePitch/collisions",
]

enabled = 0
for col_path in foot_collision_paths:
    col_prim = stage.GetPrimAtPath(col_path)
    if col_prim:
        for child in Usd.PrimRange(col_prim):
            if child.GetTypeName() == "Mesh":
                if not child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(child)
                    enabled += 1
                    L.log(f"  ✅ {child.GetPath()}")

L.log(f"  Enabled on {enabled} meshes")
L.log("")

# ══════════════════════════════════════════════════════════════════════════════
# FIX 3: Add proper foot sole collision boxes
#         From URDF: foot is 10cm long, 5cm wide
#         FSR corners: front=+7cm, rear=-3cm from ankle, 4.5cm below
#         We add a flat box covering the full sole area
# ══════════════════════════════════════════════════════════════════════════════
L.log("FIX 3: Add foot sole collision boxes")

# Foot sole box dimensions (from URDF FSR analysis)
SOLE_LENGTH = 0.10    # 10cm (front to rear)
SOLE_WIDTH  = 0.06    # 6cm (side to side, conservative)
SOLE_HEIGHT = 0.005   # 0.5cm thin

# Position relative to ankle joint:
# Center X = (0.07025 + (-0.03025)) / 2 = 0.02 (2cm forward of ankle)
# Z = -0.04511 (sole level)
SOLE_CENTER_X = 0.02
SOLE_CENTER_Z = -0.04511

for link_name, box_name in [("l_ankle", "l_sole_box"), ("r_ankle", "r_sole_box")]:
    box_path = f"{nao_prim_path}/{link_name}/collisions/{box_name}"
    try:
        box_prim = UsdGeom.Cube.Define(stage, box_path)
        box_prim.CreateSizeAttr(1.0)

        xform = UsdGeom.Xformable(box_prim.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(SOLE_CENTER_X, 0.0, SOLE_CENTER_Z))
        xform.AddScaleOp().Set(Gf.Vec3d(SOLE_LENGTH, SOLE_WIDTH, SOLE_HEIGHT))

        UsdPhysics.CollisionAPI.Apply(box_prim.GetPrim())
        L.log(f"  ✅ {box_path}: {SOLE_LENGTH*100}x{SOLE_WIDTH*100}x{SOLE_HEIGHT*100}cm "
              f"at ({SOLE_CENTER_X*100:.0f}, 0, {SOLE_CENTER_Z*100:.1f})cm from ankle")
    except Exception as e:
        L.log(f"  ❌ {box_path}: {e}")

L.log("")

# ══════════════════════════════════════════════════════════════════════════════
# FIX 4: Fix ground material
# ══════════════════════════════════════════════════════════════════════════════
L.log("FIX 4: Fix ground material (reduce bounce, increase friction)")
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MaterialAPI):
        mat = UsdPhysics.MaterialAPI(prim)
        mat.GetRestitutionAttr().Set(0.0)       # zero bounce
        mat.GetStaticFrictionAttr().Set(1.0)
        mat.GetDynamicFrictionAttr().Set(0.8)
        L.log(f"  ✅ {prim.GetPath()}: restitution=0, friction=1.0/0.8")
L.log("")

# ══════════════════════════════════════════════════════════════════════════════
# FIX 5: Set damping on USD joints BEFORE Articulation creation
# ══════════════════════════════════════════════════════════════════════════════
L.log("FIX 5: Set USD joint damping before Articulation init")
DAMPING = 500.0
for prim in Usd.PrimRange(root):
    if prim.IsA(UsdPhysics.RevoluteJoint):
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            damp_attr = drive.GetDampingAttr()
            if damp_attr:
                damp_attr.Set(DAMPING)
            else:
                drive.CreateDampingAttr(DAMPING)
L.log(f"  Set damping={DAMPING} on all joints")
L.log("")

# ══════════════════════════════════════════════════════════════════════════════
# NOW create Articulation
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("Creating Articulation and world.reset()...")
L.log("=" * 80)

nao = world.scene.add(Articulation(nao_prim_path, name="nao"))
world.reset()

dof_names = nao.dof_names
num_dof = nao.num_dof
PHYSICS_DT = 1.0 / 120.0

knee_idx = dof_names.index("LKneePitch")
ankle_idx = dof_names.index("LAnklePitch")
hip_idx = dof_names.index("LHipPitch")

# ══════════════════════════════════════════════════════════════════════════════
# FIX 6: Configure gains via ArticulationController (survives reset)
# ══════════════════════════════════════════════════════════════════════════════
controller = nao.get_articulation_controller()
kps = np.full(num_dof, 35810.0, dtype=np.float32)
kds = np.full(num_dof, 500.0, dtype=np.float32)
controller.set_gains(kps, kds)
controller.set_max_efforts(np.full(num_dof, 100.0, dtype=np.float32))

p = nao.dof_properties
L.log(f"  NAO: {num_dof} DOFs")
L.log(f"  Gains: stiff={p[0]['stiffness']:.1f}, damp={p[0]['damping']:.1f}, maxEff={p[0]['maxEffort']:.1f}")
L.log("")

# ── Poses ─────────────────────────────────────────────────────────────────────
# Straight legs with arms down
straight = np.zeros(num_dof, dtype=np.float32)
for name, deg in {"LShoulderPitch": 84, "RShoulderPitch": 84,
                   "LShoulderRoll": 10, "RShoulderRoll": -10,
                   "LElbowYaw": -70, "RElbowYaw": 70,
                   "LElbowRoll": -25, "RElbowRoll": 25}.items():
    if name in dof_names:
        straight[dof_names.index(name)] = math.radians(deg)

# Bent legs (original home pose)
bent = np.zeros(num_dof, dtype=np.float32)
for name, deg in {"LHipPitch": -25, "RHipPitch": -25,
                   "LKneePitch": 45, "RKneePitch": 45,
                   "LAnklePitch": -10, "RAnklePitch": -10,
                   "LShoulderPitch": 84, "RShoulderPitch": 84,
                   "LShoulderRoll": 10, "RShoulderRoll": -10,
                   "LElbowYaw": -70, "RElbowYaw": 70,
                   "LElbowRoll": -25, "RElbowRoll": 25}.items():
    if name in dof_names:
        bent[dof_names.index(name)] = math.radians(deg)


# ── Standing test ─────────────────────────────────────────────────────────────
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
# TESTS — Straight legs at spawn heights near the expected standing height
# NAO standing height: ankle at ~0.26m above ground + 0.045m sole = ~0.31m
# Plus torso offset. Total CoM height ≈ 0.33m.
# Spawn slightly higher to avoid ground penetration.
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("STANDING TESTS")
L.log("=" * 80)
L.log("")

# Straight legs
for h in [0.335, 0.34, 0.35, 0.38]:
    run_test(straight, h, seconds=args.seconds, label=f"Straight, h={h}")

# Bent legs
for h in [0.32, 0.335, 0.35]:
    run_test(bent, h, seconds=args.seconds, label=f"Bent, h={h}")


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
