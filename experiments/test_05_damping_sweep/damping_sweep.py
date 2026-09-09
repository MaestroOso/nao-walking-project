"""
damping_sweep.py — test_05: Kd + hip lean + knee limit fix
============================================================

Round 1: Pure Kd sweep, no knee fix. All failed. Knee buckled to -5.3°.
Round 2: Added knee limit fix (+2°). Knee holds, but hip drifts -15° to -30°
  forward under gravity. CoM is at rear edge of support polygon.
  No Kd value survived >1.5s.

Round 3 (this run): Added hip lean sweep.
  The foot extends 7cm forward / 3cm rear of ankle.
  With straight legs, CoM projects near the rear edge of support.
  A small forward hip lean should center it.

  Convention: HipPitch negative = forward lean (hip flexion).
  We sweep hip lean from -8° to +2° (mostly forward lean).
  Also sweep ankle pitch to compensate (keep torso upright).
  For each hip lean, ankle pitch ≈ -hip_lean to keep torso vertical.

  Kd fixed at 100 (best from round 2).

USAGE:
  python damping_sweep.py --headless
  python damping_sweep.py             # GUI mode
"""

import math
import argparse
import numpy as np
import time as pytime

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--output", default="damping_sweep_r3_results.txt")
parser.add_argument("--seconds", type=float, default=10.0,
                    help="Duration of each trial in seconds")
parser.add_argument("--kd-values", type=float, nargs="+",
                    default=[100, 500],
                    help="Kd values to sweep")
parser.add_argument("--hip-leans", type=float, nargs="+",
                    default=[-8, -5, -3, -2, -1, 0, 2],
                    help="Hip lean in degrees (negative = forward lean)")
parser.add_argument("--spawn-height", type=float, default=0.34,
                    help="Spawn height for all trials")
args = parser.parse_args()

KP = 35810.0  # Held constant across all trials


# ══════════════════════════════════════════════════════════════════════════════
# Logger
# ══════════════════════════════════════════════════════════════════════════════
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
L.log("test_05 Round 3: Kd × HIP LEAN SWEEP (+ knee limit fix)")
L.log(f"  Kp = {KP} (constant)")
L.log(f"  Kd values = {args.kd_values}")
L.log(f"  Hip leans = {args.hip_leans}° (neg = forward)")
L.log(f"  Spawn height = {args.spawn_height}m")
L.log(f"  Trial duration = {args.seconds}s each")
L.log(f"  Total trials = {len(args.kd_values) * len(args.hip_leans)}")
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
# Import URDF (identical to standing_final.py)
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
# FIX 4: Ground material (restitution=0, high friction)
# ══════════════════════════════════════════════════════════════════════════════
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MaterialAPI):
        mat = UsdPhysics.MaterialAPI(prim)
        mat.GetRestitutionAttr().Set(0.0)
        mat.GetStaticFrictionAttr().Set(1.0)
        mat.GetDynamicFrictionAttr().Set(0.8)
L.log("FIX 4: Ground material fixed")

# ══════════════════════════════════════════════════════════════════════════════
# FIX 5: Set INITIAL damping via USD DriveAPI (before Articulation)
#         This gets overridden per-trial, but we need a valid starting value
# ══════════════════════════════════════════════════════════════════════════════
INITIAL_KD = args.kd_values[0]
for prim in Usd.PrimRange(root):
    if prim.IsA(UsdPhysics.RevoluteJoint):
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            damp_attr = drive.GetDampingAttr()
            if damp_attr:
                damp_attr.Set(INITIAL_KD)
            else:
                drive.CreateDampingAttr(INITIAL_KD)
L.log(f"FIX 5: USD damping set to {INITIAL_KD} (initial)")

# ══════════════════════════════════════════════════════════════════════════════
# FIX 6: Override knee joint lower limits
#         URDF defines LKneePitch/RKneePitch lower = -5.3° (backward bend).
#         This allows the knee to hyperextend, buckling the leg under gravity.
#         Real NAO knees don't hyperextend. Set lower limit to +2° (slight flex)
#         to keep the leg structurally sound.
# ══════════════════════════════════════════════════════════════════════════════
KNEE_LOWER_RAD = math.radians(2.0)  # minimum 2° forward flex
knee_joints_fixed = 0
for prim in Usd.PrimRange(root):
    if prim.IsA(UsdPhysics.RevoluteJoint):
        name = prim.GetName()
        if "KneePitch" in name:
            joint = UsdPhysics.RevoluteJoint(prim)
            old_lower = joint.GetLowerLimitAttr().Get()
            # USD revolute limits are in degrees
            joint.GetLowerLimitAttr().Set(math.degrees(KNEE_LOWER_RAD))
            new_lower = joint.GetLowerLimitAttr().Get()
            L.log(f"FIX 6: {name} lower limit: {old_lower:.1f}° → {new_lower:.1f}°")
            knee_joints_fixed += 1
L.log(f"FIX 6: Fixed {knee_joints_fixed} knee joints")
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
L.log(f"DOF names: {dof_names}")
L.log("")

# Joint indices for logging
knee_idx = dof_names.index("LKneePitch")
ankle_idx = dof_names.index("LAnklePitch")
hip_idx = dof_names.index("LHipPitch")

# ══════════════════════════════════════════════════════════════════════════════
# Build poses
# ══════════════════════════════════════════════════════════════════════════════
ARM_POSE = {
    "LShoulderPitch": 84, "RShoulderPitch": 84,
    "LShoulderRoll": 10, "RShoulderRoll": -10,
    "LElbowYaw": -70, "RElbowYaw": 70,
    "LElbowRoll": -25, "RElbowRoll": 25,
}


def make_pose(joint_degrees):
    """Create a full joint position array from a dict of {name: degrees}."""
    pose = np.zeros(num_dof, dtype=np.float32)
    for name, deg in joint_degrees.items():
        if name in dof_names:
            pose[dof_names.index(name)] = math.radians(deg)
    return pose


def make_lean_pose(hip_lean_deg):
    """Create a straight-leg pose with the given hip lean.

    hip_lean_deg: HipPitch angle in degrees (negative = forward lean).
    Ankle pitch is set to compensate so the torso stays roughly vertical:
      ankle_pitch ≈ -hip_lean  (opposite sign to cancel the lean at ground level)

    Example: hip_lean=-3° → ankles=+3° → robot leans forward at hip,
    feet tilt to keep soles flat, torso stays upright.
    """
    leg_pose = {
        "LHipPitch": hip_lean_deg,
        "RHipPitch": hip_lean_deg,
        "LAnklePitch": -hip_lean_deg,  # compensate to keep torso upright
        "RAnklePitch": -hip_lean_deg,
    }
    return make_pose({**ARM_POSE, **leg_pose})


# ══════════════════════════════════════════════════════════════════════════════
# Run one trial
# ══════════════════════════════════════════════════════════════════════════════
def run_trial(kd_value, pose, spawn_h, seconds, label, hip_lean_deg=0.0):
    """Run a single standing trial with the given Kd value and pose.

    Returns dict with summary metrics for the comparison table.
    """
    # --- Set gains for this trial ---
    kps = np.full(num_dof, KP, dtype=np.float32)
    kds = np.full(num_dof, kd_value, dtype=np.float32)
    controller.set_gains(kps, kds)
    controller.set_max_efforts(np.full(num_dof, 100.0, dtype=np.float32))

    # --- Reset robot state ---
    nao.set_world_pose(
        position=np.array([0.0, 0.0, spawn_h]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    nao.set_joint_positions(pose)
    nao.set_joint_velocities(np.zeros(num_dof))

    # Let physics settle for a few frames after teleport
    for _ in range(5):
        world.step(render=not args.headless)

    L.log(f"  --- {label} ---")
    L.log(f"  Kp={KP}, Kd={kd_value}")
    L.log(f"  {'Step':>6}  {'Time':>7}  {'Height':>8}  {'qw':>8}  {'Hip':>8}  "
          f"{'Knee':>8}  {'Ankle':>8}  Note")
    L.log("  " + "-" * 80)

    total_steps = int(seconds / PHYSICS_DT)
    fell = False
    fell_time = None
    final_h = spawn_h
    min_h = spawn_h
    max_hip_error = 0.0     # Track worst joint tracking error
    max_knee_error = 0.0

    # Target angles for error calculation
    hip_target_deg = math.degrees(pose[hip_idx])
    knee_target_deg = math.degrees(pose[knee_idx])

    for step_i in range(total_steps):
        action = ArticulationAction(joint_positions=pose)
        nao.apply_action(action)
        world.step(render=not args.headless)

        # Log at key moments: dense at start, sparse later
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

                # Track joint errors
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

        # Early termination: if fallen, no need to continue
        if fell:
            break

    # --- Summary ---
    standing = final_h > 0.25 and not fell
    verdict = "STANDING" if standing else ("FELL" if fell else "LOW")
    survive_time = fell_time if fell else seconds

    L.log(f"  >> {verdict} | final_h={final_h:.4f}m | min_h={min_h:.4f}m | "
          f"survive={survive_time:.2f}s")
    L.log(f"     max_hip_err={max_hip_error:.1f}° | max_knee_err={max_knee_error:.1f}°")
    L.log("")

    return {
        "kd": kd_value,
        "hip_lean": hip_lean_deg,
        "verdict": verdict,
        "final_h": final_h,
        "min_h": min_h,
        "survive_time": survive_time,
        "max_hip_error": max_hip_error,
        "max_knee_error": max_knee_error,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2D SWEEP: Hip Lean × Kd
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("HIP LEAN × Kd SWEEP")
L.log("=" * 80)
L.log("")

results = []
trial_num = 0
total_trials = len(args.hip_leans) * len(args.kd_values)

for hip_lean in args.hip_leans:
    pose = make_lean_pose(hip_lean)
    for kd in args.kd_values:
        trial_num += 1
        label = (f"[{trial_num}/{total_trials}] "
                 f"hip={hip_lean:+.0f}°, Kd={kd}, h={args.spawn_height}")
        r = run_trial(kd, pose, args.spawn_height, args.seconds, label,
                      hip_lean_deg=hip_lean)
        results.append(r)

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
L.log("")
L.log("=" * 80)
L.log("COMPARISON TABLE (sorted by survive time)")
L.log("=" * 80)
L.log("")
L.log(f"  {'HipLean':>8}  {'Kd':>6}  {'Verdict':>10}  {'Survive':>8}  "
      f"{'Final H':>8}  {'Min H':>8}  {'HipErr':>8}  {'KneeErr':>8}")
L.log("  " + "-" * 78)

# Sort by survive time (descending), then by min_h
sorted_results = sorted(results, key=lambda r: (r['survive_time'], r['min_h']),
                         reverse=True)
for r in sorted_results:
    marker = " ✅" if r['verdict'] == 'STANDING' else ""
    L.log(f"  {r['hip_lean']:>+7.0f}°  {r['kd']:>6.0f}  {r['verdict']:>10}  "
          f"{r['survive_time']:>7.2f}s  {r['final_h']:>8.4f}  {r['min_h']:>8.4f}  "
          f"{r['max_hip_error']:>7.1f}°  {r['max_knee_error']:>7.1f}°{marker}")

L.log("")

# Identify best
standing_results = [r for r in results if r['verdict'] == 'STANDING']
if standing_results:
    best = max(standing_results, key=lambda r: r['min_h'])
    L.log(f"✅ BEST: hip={best['hip_lean']:+.0f}°, Kd={best['kd']} "
          f"— stood for full {args.seconds}s, min height={best['min_h']:.4f}m")
else:
    best = max(results, key=lambda r: r['survive_time'])
    L.log(f"⚠️  No trial survived full duration. Best: "
          f"hip={best['hip_lean']:+.0f}°, Kd={best['kd']} "
          f"— survived {best['survive_time']:.2f}s")

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
