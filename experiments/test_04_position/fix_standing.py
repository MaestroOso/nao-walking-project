"""
fix_standing.py — Fix DOF Properties and Achieve Standing
============================================================

ROOT CAUSE FOUND (from minimal_results.txt):
  1. Joints have stiffness=35810 but damping=0.0
     → Undamped position drives oscillate violently
  2. RHipYawPitch (DOF 3) has driveMode=0, stiffness=0, maxEffort=0,
     maxVelocity=5.9e+36 — completely broken configuration
  3. set_joint_efforts() conflicts with existing position drives → NaN
  4. Ragdoll mode (Test A) and position mode (Test D) both run without
     NaN — proving the physics engine itself is fine

FIX:
  1. Set damping on all DOFs via dof_properties
  2. Fix RHipYawPitch to match the other joints
  3. Use apply_action(joint_positions=...) for control
  4. Optionally reduce stiffness from 35810 to something more realistic

USAGE:
  python fix_standing.py --headless
  python fix_standing.py                   # GUI — watch NAO stand
"""

import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--output", default="fix_results.txt")
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
L.log("NAO STANDING FIX")
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
PHYSICS_DT = 1.0 / 120.0

L.log(f"NAO: {num_dof} DOFs at {nao_prim_path}")
L.log("")

# ── Index lookups ─────────────────────────────────────────────────────────────
knee_idx = dof_names.index("LKneePitch")
ankle_idx = dof_names.index("LAnklePitch")
hip_idx = dof_names.index("LHipPitch")

# ── Home pose ─────────────────────────────────────────────────────────────────
home = np.zeros(num_dof, dtype=np.float32)
joint_angles = {
    "LHipPitch": -25, "RHipPitch": -25,
    "LKneePitch": 45, "RKneePitch": 45,
    "LAnklePitch": -10, "RAnklePitch": -10,
    "LShoulderPitch": 84, "RShoulderPitch": 84,
    "LShoulderRoll": 10, "RShoulderRoll": -10,
    "LElbowYaw": -70, "RElbowYaw": 70,
    "LElbowRoll": -25, "RElbowRoll": 25,
}
for name, deg in joint_angles.items():
    if name in dof_names:
        home[dof_names.index(name)] = math.radians(deg)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Inspect and fix DOF properties
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("STEP 1: Inspect and Fix DOF Properties")
L.log("=" * 80)
L.log("")

props = nao.dof_properties
L.log("BEFORE FIX:")
L.log(f"  {'DOF':>3}  {'Name':30s}  {'driveMode':>10}  {'stiffness':>12}  {'damping':>10}  "
      f"{'maxEffort':>10}  {'maxVelocity':>12}  Issues")
L.log("  " + "-" * 120)

for i in range(num_dof):
    issues = []
    if props[i]['damping'] == 0.0:
        issues.append("NO_DAMPING")
    if props[i]['stiffness'] == 0.0:
        issues.append("NO_STIFFNESS")
    if props[i]['driveMode'] == 0:
        issues.append("DRIVE_OFF")
    if props[i]['maxEffort'] == 0.0:
        issues.append("NO_EFFORT")
    if props[i]['maxVelocity'] > 1e10:
        issues.append("INF_VEL")

    issue_str = ", ".join(issues) if issues else "ok"
    L.log(f"  {i:>3}  {dof_names[i]:30s}  {props[i]['driveMode']:>10}  "
          f"{props[i]['stiffness']:>12.1f}  {props[i]['damping']:>10.1f}  "
          f"{props[i]['maxEffort']:>10.3f}  {props[i]['maxVelocity']:>12.3f}  {issue_str}")

# ── Apply fixes ───────────────────────────────────────────────────────────────
L.log("")
L.log("APPLYING FIXES:")
L.log("")

# Get a reference stiffness from a known-good joint (DOF 0)
ref_stiffness = float(props[0]['stiffness'])  # 35809.86
ref_drive_mode = int(props[0]['driveMode'])    # 2

# Damping: critical damping ratio ζ = Kd / (2 * sqrt(Kp * I))
# For small joints with I ~ 0.001 kg·m², Kp = 35810:
#   critical Kd = 2 * sqrt(35810 * 0.001) ≈ 12
# But we'll use a fraction of stiffness as a practical starting point.
# Typical ratio: damping = stiffness * 0.01 to 0.1
DAMPING_VALUES_TO_TEST = [100.0, 500.0, 1000.0, 3581.0]

for i in range(num_dof):
    # Fix broken joints (like RHipYawPitch)
    if props[i]['driveMode'] == 0 or props[i]['stiffness'] == 0:
        L.log(f"  Fixing DOF {i} ({dof_names[i]}): driveMode=0→{ref_drive_mode}, "
              f"stiffness=0→{ref_stiffness:.1f}")
        props[i]['driveMode'] = ref_drive_mode
        props[i]['stiffness'] = ref_stiffness

    if props[i]['maxEffort'] == 0.0:
        # Use a reasonable max effort for NAO joints
        props[i]['maxEffort'] = 3.0  # Nm
        L.log(f"  Fixing DOF {i} ({dof_names[i]}): maxEffort=0→3.0")

    if props[i]['maxVelocity'] > 1e10:
        props[i]['maxVelocity'] = 8.27  # rad/s — same as other joints
        L.log(f"  Fixing DOF {i} ({dof_names[i]}): maxVelocity=inf→8.27")

L.log("")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Test different damping values
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("STEP 2: Test Damping Values")
L.log("=" * 80)
L.log("")

def run_standing_test(damping_val, seconds, label=""):
    """Apply damping, reset, and test standing with position control."""

    # Set damping on all DOFs
    for i in range(num_dof):
        props[i]['damping'] = damping_val

    # Write properties back — this is the critical step
    # The dof_properties setter should update PhysX
    nao.dof_properties = props

    # Reset world and pose
    world.reset()

    # Re-apply properties after reset (reset may clear them)
    nao.dof_properties = props

    nao.set_world_pose(
        position=np.array([0.0, 0.0, 0.335]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    nao.set_joint_positions(home)
    nao.set_joint_velocities(np.zeros(num_dof))

    L.log(f"  --- {label}Damping={damping_val:.1f} (stiffness={props[0]['stiffness']:.1f}) ---")
    L.log(f"  {'Step':>6}  {'Time':>7}  {'Height':>8}  {'qw':>8}  {'Knee':>8}  "
          f"{'Ankle':>8}  {'Hip':>8}  Note")
    L.log("  " + "-" * 78)

    fell = False
    nan_seen = False
    final_h = 0.335

    total_steps = int(seconds / PHYSICS_DT)
    print_interval = int(0.5 / PHYSICS_DT)

    for step_i in range(total_steps):
        # Position control: tell drives to hold the home pose
        action = ArticulationAction(joint_positions=home)
        nao.apply_action(action)
        world.step(render=not args.headless)

        # Log at intervals (more frequent in first second)
        log_this = (step_i < 120 and step_i % 20 == 0) or (step_i >= 120 and step_i % print_interval == 0)
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

    # Verdict
    if nan_seen:
        verdict = "NaN_CORRUPTION"
    elif fell:
        verdict = "FELL"
    elif final_h > 0.25:
        verdict = "STANDING"
    else:
        verdict = "CROUCHING"

    L.log(f"  Result: {verdict} (final_h={final_h:.4f}m)")
    L.log("")
    return verdict, final_h


# Test each damping value
results = []
for damp in DAMPING_VALUES_TO_TEST:
    verdict, h = run_standing_test(damp, seconds=5.0)
    results.append((damp, verdict, h))

L.log("DAMPING SWEEP SUMMARY:")
L.log(f"  {'Damping':>10}  {'Verdict':>15}  {'FinalH':>10}")
L.log("  " + "-" * 40)
for damp, verdict, h in results:
    L.log(f"  {damp:>10.1f}  {verdict:>15}  {h:>10.4f}")
L.log("")

# Find best
standing = [(d, v, h) for d, v, h in results if v == "STANDING"]
if standing:
    best_damp = standing[0][0]
    L.log(f"BEST DAMPING: {best_damp}")
    L.log("")

    # Extended confirmation with best
    L.log("=" * 80)
    L.log(f"EXTENDED CONFIRMATION ({args.seconds}s with damping={best_damp})")
    L.log("=" * 80)
    L.log("")
    run_standing_test(best_damp, seconds=args.seconds, label="CONFIRM: ")
else:
    L.log("No standing config found. Trying with reduced stiffness + damping...")
    L.log("")

    # Maybe the stiffness is too high — try reducing it
    stiffness_tests = [
        (1000.0, 100.0),
        (500.0,  50.0),
        (200.0,  20.0),
        (100.0,  10.0),
    ]

    for stiff, damp in stiffness_tests:
        for i in range(num_dof):
            props[i]['stiffness'] = stiff
        verdict, h = run_standing_test(damp, seconds=5.0, label=f"Kp={stiff}: ")
        if verdict == "STANDING":
            L.log(f"\nFOUND: stiffness={stiff}, damping={damp}")
            L.log("Running extended confirmation...")
            run_standing_test(damp, seconds=args.seconds, label="CONFIRM: ")
            break


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Verify final DOF properties
# ══════════════════════════════════════════════════════════════════════════════
L.log("")
L.log("=" * 80)
L.log("FINAL DOF PROPERTIES")
L.log("=" * 80)
L.log("")

final_props = nao.dof_properties
L.log(f"  {'DOF':>3}  {'Name':30s}  {'driveMode':>10}  {'stiffness':>12}  {'damping':>10}  "
      f"{'maxEffort':>10}")
L.log("  " + "-" * 100)
for i in range(num_dof):
    L.log(f"  {i:>3}  {dof_names[i]:30s}  {final_props[i]['driveMode']:>10}  "
          f"{final_props[i]['stiffness']:>12.1f}  {final_props[i]['damping']:>10.1f}  "
          f"{final_props[i]['maxEffort']:>10.3f}")


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
