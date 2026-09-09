"""
gain_calibration.py — test_05 Round 4: Realistic PD gain sweep
================================================================

Rounds 1-3 findings:
  - Knee limit fix works (knee holds at +2°, no more backward buckling)
  - Hip lean sweep has minimal effect (robot falls at all lean values)
  - Root cause: hip drifts 15-20° from target in every trial regardless
    of Kd or hip lean. The PD controller can't hold position.

Diagnosis: TORQUE-TO-WEIGHT MISMATCH
  - URDF importer set Kp=35,810 — this is unrealistically high
  - URDF max effort: 3.023 Nm for hip/knee/ankle
  - Gravitational torque at hip: ~5.2kg × 9.81 × 0.15m ≈ 7.6 Nm
  - The drives generate unrealistic torques that cause instability
  - Isaac Sim's PhysX solver may be clamping or fighting these values

Solution: Calibrate gains to realistic NAO actuator specs.
  NAO V40 uses MX-28 servos in the legs:
    - Stall torque: ~2.5 Nm (with reduction)
    - Max effort in URDF: 3.023 Nm
    - Reasonable Kp: 10-100 (proportional to torque/radian)
    - Reasonable Kd: 0.1-5 (light damping for small servos)

Method: Sweep Kp and Kd together at fixed ratios.
  Fix hip lean at -2° (best from round 3), knee limit at +2°.
  Test Kp = [10, 25, 50, 100, 200, 500] with Kd = Kp/10.
  Also test max_effort at realistic (3.0 Nm) vs current (100 Nm).

USAGE:
  python gain_calibration.py --headless
  python gain_calibration.py             # GUI mode
"""

import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--output", default="gain_calibration_results.txt")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--spawn-height", type=float, default=0.34)
args = parser.parse_args()

# ── Trial configurations ──────────────────────────────────────────────────────
# Each tuple: (Kp, Kd, max_effort, label)
# We test realistic gains at two max_effort levels:
#   - 3.0 Nm (URDF-realistic for NAO leg servos)
#   - 10.0 Nm (relaxed, allows more correction authority)
#   - 100.0 Nm (previous setting, for comparison)
TRIALS = [
    # Realistic gains, realistic torque limit
    (10,    1.0,   3.0,  "Kp=10,  Kd=1,   eff=3"),
    (25,    2.5,   3.0,  "Kp=25,  Kd=2.5, eff=3"),
    (50,    5.0,   3.0,  "Kp=50,  Kd=5,   eff=3"),
    (100,  10.0,   3.0,  "Kp=100, Kd=10,  eff=3"),

    # Realistic gains, relaxed torque limit
    (10,    1.0,  10.0,  "Kp=10,  Kd=1,   eff=10"),
    (25,    2.5,  10.0,  "Kp=25,  Kd=2.5, eff=10"),
    (50,    5.0,  10.0,  "Kp=50,  Kd=5,   eff=10"),
    (100,  10.0,  10.0,  "Kp=100, Kd=10,  eff=10"),

    # Moderate gains, high torque (previous ballpark)
    (500,   50.0, 100.0, "Kp=500, Kd=50,  eff=100"),
    (1000, 100.0, 100.0, "Kp=1k,  Kd=100, eff=100"),

    # Original setting for baseline comparison
    (35810, 500.0, 100.0, "Kp=35k, Kd=500, eff=100 [ORIGINAL]"),
]

# Fixed pose: hip lean = -2° with ankle compensation, knees straight
HIP_LEAN_DEG = -2.0


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
L.log("test_05 Round 4: PD GAIN CALIBRATION")
L.log(f"  Hip lean = {HIP_LEAN_DEG}° (fixed)")
L.log(f"  Spawn height = {args.spawn_height}m")
L.log(f"  Trial duration = {args.seconds}s")
L.log(f"  {len(TRIALS)} configurations to test")
L.log("=" * 80)
L.log("")

# ══════════════════════════════════════════════════════════════════════════════
# Initialize Isaac Sim
# ══════════════════════════════════════════════════════════════════════════════
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
import omni.kit.commands
import isaacsim.asset.importer.urdf._urdf as urdf_module
from pxr import UsdPhysics, UsdGeom, Usd, Gf

PHYSICS_DT = 1.0 / 120.0

world = World(physics_dt=PHYSICS_DT, rendering_dt=1.0 / 60.0)
world.scene.add_default_ground_plane()

# ══════════════════════════════════════════════════════════════════════════════
# Import URDF
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# FIX 1: Disable instancing
# ══════════════════════════════════════════════════════════════════════════════
for prim in Usd.PrimRange(root):
    if prim.IsInstanceable():
        prim.SetInstanceable(False)
L.log("FIX 1: Instancing disabled")

# ══════════════════════════════════════════════════════════════════════════════
# FIX 2: Enable CollisionAPI on foot collision meshes
# ══════════════════════════════════════════════════════════════════════════════
foot_collision_paths = [
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
L.log(f"FIX 2: CollisionAPI on {enabled} foot meshes")

# ══════════════════════════════════════════════════════════════════════════════
# FIX 3: Add foot sole collision boxes
# ══════════════════════════════════════════════════════════════════════════════
SOLE_LENGTH = 0.10
SOLE_WIDTH = 0.06
SOLE_HEIGHT = 0.005
SOLE_CENTER_X = 0.02
SOLE_CENTER_Z = -0.04511

for link_name, box_name in [("l_ankle", "l_sole_box"), ("r_ankle", "r_sole_box")]:
    box_path = f"{nao_prim_path}/{link_name}/collisions/{box_name}"
    box_prim = UsdGeom.Cube.Define(stage, box_path)
    box_prim.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(box_prim.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(SOLE_CENTER_X, 0.0, SOLE_CENTER_Z))
    xform.AddScaleOp().Set(Gf.Vec3d(SOLE_LENGTH, SOLE_WIDTH, SOLE_HEIGHT))
    UsdPhysics.CollisionAPI.Apply(box_prim.GetPrim())
L.log("FIX 3: Sole boxes added")

# ══════════════════════════════════════════════════════════════════════════════
# FIX 4: Ground material
# ══════════════════════════════════════════════════════════════════════════════
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MaterialAPI):
        mat = UsdPhysics.MaterialAPI(prim)
        mat.GetRestitutionAttr().Set(0.0)
        mat.GetStaticFrictionAttr().Set(1.0)
        mat.GetDynamicFrictionAttr().Set(0.8)
L.log("FIX 4: Ground material fixed")

# ══════════════════════════════════════════════════════════════════════════════
# FIX 5: Set initial damping via USD DriveAPI (before Articulation)
#         Also set stiffness via USD to a sane initial value
# ══════════════════════════════════════════════════════════════════════════════
INITIAL_KP = TRIALS[0][0]
INITIAL_KD = TRIALS[0][1]
for prim in Usd.PrimRange(root):
    if prim.IsA(UsdPhysics.RevoluteJoint):
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            # Set stiffness
            stiff_attr = drive.GetStiffnessAttr()
            if stiff_attr:
                stiff_attr.Set(float(INITIAL_KP))
            else:
                drive.CreateStiffnessAttr(float(INITIAL_KP))
            # Set damping
            damp_attr = drive.GetDampingAttr()
            if damp_attr:
                damp_attr.Set(float(INITIAL_KD))
            else:
                drive.CreateDampingAttr(float(INITIAL_KD))
L.log(f"FIX 5: USD gains set to Kp={INITIAL_KP}, Kd={INITIAL_KD} (initial)")

# ══════════════════════════════════════════════════════════════════════════════
# FIX 6: Override knee joint lower limits to +2°
# ══════════════════════════════════════════════════════════════════════════════
KNEE_LOWER_RAD = math.radians(2.0)
for prim in Usd.PrimRange(root):
    if prim.IsA(UsdPhysics.RevoluteJoint):
        name = prim.GetName()
        if "KneePitch" in name:
            joint = UsdPhysics.RevoluteJoint(prim)
            joint.GetLowerLimitAttr().Set(math.degrees(KNEE_LOWER_RAD))
            L.log(f"FIX 6: {name} lower limit → {math.degrees(KNEE_LOWER_RAD):.1f}°")
L.log("")

# ══════════════════════════════════════════════════════════════════════════════
# Create Articulation
# ══════════════════════════════════════════════════════════════════════════════
nao = world.scene.add(Articulation(nao_prim_path, name="nao"))
world.reset()

dof_names = nao.dof_names
num_dof = nao.num_dof
controller = nao.get_articulation_controller()

L.log(f"NAO: {num_dof} DOFs")
L.log("")

# Joint indices for logging
hip_idx = dof_names.index("LHipPitch")
knee_idx = dof_names.index("LKneePitch")
ankle_idx = dof_names.index("LAnklePitch")

# ══════════════════════════════════════════════════════════════════════════════
# Build the fixed pose: hip lean = -2°, ankle compensated
# ══════════════════════════════════════════════════════════════════════════════
def make_pose(joint_degrees):
    pose = np.zeros(num_dof, dtype=np.float32)
    for name, deg in joint_degrees.items():
        if name in dof_names:
            pose[dof_names.index(name)] = math.radians(deg)
    return pose


target_pose = make_pose({
    # Arms down
    "LShoulderPitch": 84, "RShoulderPitch": 84,
    "LShoulderRoll": 10, "RShoulderRoll": -10,
    "LElbowYaw": -70, "RElbowYaw": 70,
    "LElbowRoll": -25, "RElbowRoll": 25,
    # Leg lean
    "LHipPitch": HIP_LEAN_DEG, "RHipPitch": HIP_LEAN_DEG,
    "LAnklePitch": -HIP_LEAN_DEG, "RAnklePitch": -HIP_LEAN_DEG,
})


# ══════════════════════════════════════════════════════════════════════════════
# Run one trial
# ══════════════════════════════════════════════════════════════════════════════
def run_trial(kp, kd, max_eff, pose, spawn_h, seconds, label):
    """Run a single standing trial with the given PD gains and torque limit."""

    # Set gains for this trial
    kps = np.full(num_dof, float(kp), dtype=np.float32)
    kds = np.full(num_dof, float(kd), dtype=np.float32)
    controller.set_gains(kps, kds)
    controller.set_max_efforts(np.full(num_dof, float(max_eff), dtype=np.float32))

    # Reset robot state
    nao.set_world_pose(
        position=np.array([0.0, 0.0, spawn_h]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    nao.set_joint_positions(pose)
    nao.set_joint_velocities(np.zeros(num_dof))

    # Let physics settle after teleport
    for _ in range(5):
        world.step(render=not args.headless)

    L.log(f"  --- {label} ---")
    L.log(f"  Kp={kp}, Kd={kd}, maxEffort={max_eff}")
    L.log(f"  {'Step':>6}  {'Time':>7}  {'Height':>8}  {'qw':>8}  {'Hip':>8}  "
          f"{'Knee':>8}  {'Ankle':>8}  Note")
    L.log("  " + "-" * 80)

    total_steps = int(seconds / PHYSICS_DT)
    fell = False
    fell_time = None
    final_h = spawn_h
    min_h = spawn_h
    max_hip_error = 0.0
    max_knee_error = 0.0

    hip_target_deg = math.degrees(pose[hip_idx])
    knee_target_deg = math.degrees(pose[knee_idx])

    for step_i in range(total_steps):
        action = ArticulationAction(joint_positions=pose)
        nao.apply_action(action)
        world.step(render=not args.headless)

        log_this = (step_i < 120 and step_i % 10 == 0) or \
                   (step_i >= 120 and step_i % int(0.5 / PHYSICS_DT) == 0)

        if log_this or step_i == total_steps - 1:
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
                    min_h = min(min_h, h)

                if not np.isnan(hp):
                    max_hip_error = max(max_hip_error, abs(hp - hip_target_deg))
                if not np.isnan(k):
                    max_knee_error = max(max_knee_error, abs(k - knee_target_deg))

                note = ""
                if np.isnan(h):
                    note = "NaN!"
                elif h < 0.10:
                    note = "GROUND"
                    if not fell:
                        fell = True
                        fell_time = t
                elif h > 0.28:
                    note = "OK"
                elif h > 0.20:
                    note = "low"

                if log_this:
                    L.log(f"  {step_i+1:>6}  {t:>6.3f}s  {h:>8.4f}  {qw:>+8.4f}  "
                          f"{hp:>+7.1f}°  {k:>+7.1f}°  {a:>+7.1f}°  {note}")
            except Exception as e:
                L.log(f"  ERROR at step {step_i}: {e}")

        if fell:
            break

    standing = final_h > 0.25 and not fell
    verdict = "STANDING" if standing else ("FELL" if fell else "LOW")
    survive_time = fell_time if fell else seconds

    L.log(f"  >> {verdict} | final_h={final_h:.4f}m | min_h={min_h:.4f}m | "
          f"survive={survive_time:.2f}s")
    L.log(f"     max_hip_err={max_hip_error:.1f}° | max_knee_err={max_knee_error:.1f}°")
    L.log("")

    return {
        "label": label,
        "kp": kp, "kd": kd, "max_eff": max_eff,
        "verdict": verdict,
        "final_h": final_h,
        "min_h": min_h,
        "survive_time": survive_time,
        "max_hip_error": max_hip_error,
        "max_knee_error": max_knee_error,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL TRIALS
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("GAIN CALIBRATION SWEEP")
L.log("=" * 80)
L.log("")

results = []
for i, (kp, kd, eff, label) in enumerate(TRIALS, 1):
    full_label = f"[{i}/{len(TRIALS)}] {label}"
    r = run_trial(kp, kd, eff, target_pose, args.spawn_height, args.seconds,
                  full_label)
    results.append(r)

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
L.log("")
L.log("=" * 80)
L.log("COMPARISON TABLE (sorted by survive time)")
L.log("=" * 80)
L.log("")
L.log(f"  {'Kp':>7}  {'Kd':>6}  {'Eff':>5}  {'Verdict':>10}  {'Survive':>8}  "
      f"{'Final H':>8}  {'Min H':>8}  {'HipErr':>8}  {'KneeErr':>8}")
L.log("  " + "-" * 82)

sorted_results = sorted(results, key=lambda r: (r['survive_time'], r['min_h']),
                         reverse=True)
for r in sorted_results:
    marker = " ✅" if r['verdict'] == 'STANDING' else ""
    L.log(f"  {r['kp']:>7}  {r['kd']:>6.1f}  {r['max_eff']:>5.0f}  "
          f"{r['verdict']:>10}  {r['survive_time']:>7.2f}s  "
          f"{r['final_h']:>8.4f}  {r['min_h']:>8.4f}  "
          f"{r['max_hip_error']:>7.1f}°  {r['max_knee_error']:>7.1f}°{marker}")

L.log("")

standing_results = [r for r in results if r['verdict'] == 'STANDING']
if standing_results:
    best = max(standing_results, key=lambda r: r['min_h'])
    L.log(f"✅ BEST: {best['label']} — stood for full {args.seconds}s, "
          f"min height={best['min_h']:.4f}m")
else:
    best = max(results, key=lambda r: r['survive_time'])
    L.log(f"⚠️  No trial survived full duration.")
    L.log(f"    Best: {best['label']} — survived {best['survive_time']:.2f}s")
    L.log("")
    L.log("ANALYSIS NOTES:")
    L.log("  If ALL trials fail, the problem is likely not gains alone.")
    L.log("  Consider: fix_base=True test, ankle roll stiffness,")
    L.log("  or reducing physics_dt for more stable integration.")

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
