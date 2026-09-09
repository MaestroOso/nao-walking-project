"""
single_trial.py — Run ONE standing trial with specified PD gains
=================================================================

Runs a single trial and exits. This avoids the NaN contamination issue
where PhysX corruption in one trial poisons all subsequent trials.

Use run_sweep.sh to run multiple configurations sequentially.

USAGE:
  python single_trial.py --kp 50 --kd 5 --eff 3 --headless
  python single_trial.py --kp 100 --kd 10 --eff 10 --headless --seconds 15
"""

import math
import argparse
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--spawn-height", type=float, default=0.34)
parser.add_argument("--hip-lean", type=float, default=-2.0,
                    help="Hip lean in degrees (negative = forward)")
parser.add_argument("--kp", type=float, required=True)
parser.add_argument("--kd", type=float, required=True)
parser.add_argument("--eff", type=float, required=True,
                    help="Max effort (torque limit) in Nm")
parser.add_argument("--output", type=str, default=None,
                    help="Output file (default: trial_kp{kp}_kd{kd}_eff{eff}.txt)")
args = parser.parse_args()

if args.output is None:
    args.output = f"trial_kp{args.kp:.0f}_kd{args.kd:.1f}_eff{args.eff:.0f}.txt"

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
        for line in self.lines:
            print(line)

L = Logger(args.output)
L.log(f"TRIAL: Kp={args.kp}, Kd={args.kd}, maxEff={args.eff}, "
      f"hipLean={args.hip_lean}°, h={args.spawn_height}, dur={args.seconds}s")
L.log("")

# ══════════════════════════════════════════════════════════════════════════════
# Initialize Isaac Sim (fresh process = clean PhysX state)
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

# ── Import URDF ───────────────────────────────────────────────────────────────
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

# ── FIX 1: Disable instancing ────────────────────────────────────────────────
for prim in Usd.PrimRange(root):
    if prim.IsInstanceable():
        prim.SetInstanceable(False)

# ── FIX 2: CollisionAPI on foot meshes ────────────────────────────────────────
for col_path in [f"{nao_prim_path}/l_ankle/collisions",
                 f"{nao_prim_path}/r_ankle/collisions",
                 f"{nao_prim_path}/LAnklePitch/collisions",
                 f"{nao_prim_path}/RAnklePitch/collisions"]:
    col_prim = stage.GetPrimAtPath(col_path)
    if col_prim:
        for child in Usd.PrimRange(col_prim):
            if child.GetTypeName() == "Mesh":
                if not child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(child)

# ── FIX 3: Foot sole boxes ───────────────────────────────────────────────────
for link_name, box_name in [("l_ankle", "l_sole_box"), ("r_ankle", "r_sole_box")]:
    box_path = f"{nao_prim_path}/{link_name}/collisions/{box_name}"
    box_prim = UsdGeom.Cube.Define(stage, box_path)
    box_prim.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(box_prim.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.02, 0.0, -0.04511))
    xform.AddScaleOp().Set(Gf.Vec3d(0.10, 0.06, 0.005))
    UsdPhysics.CollisionAPI.Apply(box_prim.GetPrim())

# ── FIX 4: Ground material ───────────────────────────────────────────────────
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MaterialAPI):
        mat = UsdPhysics.MaterialAPI(prim)
        mat.GetRestitutionAttr().Set(0.0)
        mat.GetStaticFrictionAttr().Set(1.0)
        mat.GetDynamicFrictionAttr().Set(0.8)

# ── FIX 5: Set gains via USD DriveAPI BEFORE Articulation ─────────────────────
for prim in Usd.PrimRange(root):
    if prim.IsA(UsdPhysics.RevoluteJoint):
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            stiff = drive.GetStiffnessAttr()
            if stiff:
                stiff.Set(float(args.kp))
            else:
                drive.CreateStiffnessAttr(float(args.kp))
            damp = drive.GetDampingAttr()
            if damp:
                damp.Set(float(args.kd))
            else:
                drive.CreateDampingAttr(float(args.kd))

# ── FIX 6: Knee lower limit → +2° ───────────────────────────────────────────
for prim in Usd.PrimRange(root):
    if prim.IsA(UsdPhysics.RevoluteJoint):
        if "KneePitch" in prim.GetName():
            UsdPhysics.RevoluteJoint(prim).GetLowerLimitAttr().Set(2.0)

L.log("All fixes applied (instancing, collision, soles, ground, gains, knee limits)")

# ══════════════════════════════════════════════════════════════════════════════
# Create Articulation
# ══════════════════════════════════════════════════════════════════════════════
nao = world.scene.add(Articulation(nao_prim_path, name="nao"))
world.reset()

dof_names = nao.dof_names
num_dof = nao.num_dof
controller = nao.get_articulation_controller()

# Set gains via controller (post-reset)
kps = np.full(num_dof, float(args.kp), dtype=np.float32)
kds = np.full(num_dof, float(args.kd), dtype=np.float32)
controller.set_gains(kps, kds)
controller.set_max_efforts(np.full(num_dof, float(args.eff), dtype=np.float32))

# Verify gains took effect
p = nao.dof_properties
L.log(f"NAO: {num_dof} DOFs")
L.log(f"Gains readback: Kp={p[0]['stiffness']:.1f}, Kd={p[0]['damping']:.1f}, "
      f"maxEff={p[0]['maxEffort']:.1f}")
L.log("")

# Joint indices
hip_idx = dof_names.index("LHipPitch")
knee_idx = dof_names.index("LKneePitch")
ankle_idx = dof_names.index("LAnklePitch")

# ══════════════════════════════════════════════════════════════════════════════
# Build pose
# ══════════════════════════════════════════════════════════════════════════════
def make_pose(joint_degrees):
    pose = np.zeros(num_dof, dtype=np.float32)
    for name, deg in joint_degrees.items():
        if name in dof_names:
            pose[dof_names.index(name)] = math.radians(deg)
    return pose

target_pose = make_pose({
    "LShoulderPitch": 84, "RShoulderPitch": 84,
    "LShoulderRoll": 10, "RShoulderRoll": -10,
    "LElbowYaw": -70, "RElbowYaw": 70,
    "LElbowRoll": -25, "RElbowRoll": 25,
    "LHipPitch": args.hip_lean, "RHipPitch": args.hip_lean,
    "LAnklePitch": -args.hip_lean, "RAnklePitch": -args.hip_lean,
})

# ══════════════════════════════════════════════════════════════════════════════
# Run trial
# ══════════════════════════════════════════════════════════════════════════════

# --- SETTLE PHASE ---
# After world.reset(), the physics needs time to initialize ground contacts.
# We teleport the robot, apply the pose target, and run for 120 frames (1s)
# to let contacts establish. Then re-teleport to clean starting state.
SETTLE_FRAMES = 120

nao.set_world_pose(
    position=np.array([0.0, 0.0, args.spawn_height]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
)
nao.set_joint_positions(target_pose)
nao.set_joint_velocities(np.zeros(num_dof))

for _ in range(SETTLE_FRAMES):
    nao.apply_action(ArticulationAction(joint_positions=target_pose))
    world.step(render=not args.headless)

# Check if settle phase already corrupted physics
settle_pos, _ = nao.get_world_pose()
settle_h = float(settle_pos[2]) if settle_pos is not None else float('nan')
L.log(f"After {SETTLE_FRAMES}-frame settle: h={settle_h:.4f}m")

# Re-teleport to clean starting position with zero velocities
nao.set_world_pose(
    position=np.array([0.0, 0.0, args.spawn_height]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
)
nao.set_joint_positions(target_pose)
nao.set_joint_velocities(np.zeros(num_dof))

# Brief re-settle (just a few frames to re-establish contact)
for _ in range(10):
    nao.apply_action(ArticulationAction(joint_positions=target_pose))
    world.step(render=not args.headless)

# Read actual starting height after settle
start_pos, _ = nao.get_world_pose()
start_h = float(start_pos[2]) if start_pos is not None else args.spawn_height
L.log(f"Trial start: h={start_h:.4f}m")
L.log("")

L.log(f"{'Step':>6}  {'Time':>7}  {'Height':>8}  {'qw':>8}  {'Hip':>8}  "
      f"{'Knee':>8}  {'Ankle':>8}  Note")
L.log("-" * 78)

total_steps = int(args.seconds / PHYSICS_DT)
fell = False
fell_time = None
final_h = start_h
min_h = start_h
max_hip_error = 0.0
max_knee_error = 0.0
nan_count = 0

hip_target_deg = math.degrees(target_pose[hip_idx])
knee_target_deg = math.degrees(target_pose[knee_idx])

for step_i in range(total_steps):
    nao.apply_action(ArticulationAction(joint_positions=target_pose))
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

            if np.isnan(h):
                nan_count += 1
                # If we get 3+ consecutive NaN readings, physics is corrupted
                if nan_count >= 3:
                    L.log(f"{step_i+1:>6}  {t:>6.3f}s  NaN — physics corrupted, aborting")
                    break
            else:
                nan_count = 0  # reset on valid reading
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
                L.log(f"{step_i+1:>6}  {t:>6.3f}s  {h:>8.4f}  {qw:>+8.4f}  "
                      f"{hp:>+7.1f}°  {k:>+7.1f}°  {a:>+7.1f}°  {note}")
        except Exception as e:
            L.log(f"ERROR at step {step_i}: {e}")

    if fell:
        break

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
survive_time = fell_time if fell else args.seconds
if nan_count >= 3:
    verdict = "NaN_CRASH"
elif fell:
    verdict = "FELL"
elif final_h > 0.25:
    verdict = "STANDING"
else:
    verdict = "LOW"

L.log("")
L.log("=" * 78)
L.log(f"RESULT: {verdict}")
L.log(f"  survive_time = {survive_time:.2f}s")
L.log(f"  final_h = {final_h:.4f}m")
L.log(f"  min_h = {min_h:.4f}m")
L.log(f"  max_hip_error = {max_hip_error:.1f}°")
L.log(f"  max_knee_error = {max_knee_error:.1f}°")
L.log("=" * 78)

# Print one-line summary to stdout for the sweep script to capture
summary = (f"Kp={args.kp:<7.0f} Kd={args.kd:<6.1f} Eff={args.eff:<5.0f} "
           f"→ {verdict:<10} survive={survive_time:.2f}s "
           f"h={final_h:.4f} hip_err={max_hip_error:.1f}° "
           f"knee_err={max_knee_error:.1f}°")
print(summary)

L.close()

try:
    simulation_app.close()
except:
    pass
