"""
final_standing.py — Fix MaxEffort + Use controller.set_gains()
================================================================

PROBLEM SOLVED:
  controller.set_gains() works! Damping persists.
  BUT: maxEffort=3.023 Nm on hip joints can't resist gravity.
  Torso (~2.5kg) at ~0.15m lever = ~3.7 Nm gravitational torque.
  Hip maxEffort=3.023 < 3.7 → hip collapses to -88° (limit).

FIX:
  Increase maxEffort via controller.set_max_efforts()
  (found in the controller methods list)

USAGE:
  python final_standing.py --headless
  python final_standing.py               # GUI to watch
"""

import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--output", default="final_results.txt")
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
L.log("NAO STANDING — Final Fix (MaxEffort + Gains)")
L.log("=" * 80)
L.log("")

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

L.log(f"NAO: {num_dof} DOFs")
L.log("")

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

straight = np.zeros(num_dof, dtype=np.float32)
for name, deg in {"LShoulderPitch": 84, "RShoulderPitch": 84,
                   "LShoulderRoll": 10, "RShoulderRoll": -10,
                   "LElbowYaw": -70, "RElbowYaw": 70,
                   "LElbowRoll": -25, "RElbowRoll": 25}.items():
    if name in dof_names:
        straight[dof_names.index(name)] = math.radians(deg)


# ── Show current max efforts ─────────────────────────────────────────────────
L.log("CURRENT DOF PROPERTIES:")
props = nao.dof_properties
L.log(f"  {'DOF':>3}  {'Name':25s}  {'stiff':>10}  {'damp':>8}  {'maxEff':>8}  {'maxVel':>8}")
L.log("  " + "-" * 70)

# Only show leg + key joints
key_joints = ["LHipPitch", "RHipPitch", "LKneePitch", "RKneePitch",
              "LAnklePitch", "RAnklePitch", "LAnkleRoll", "RAnkleRoll",
              "LHipRoll", "RHipRoll", "LHipYawPitch", "RHipYawPitch"]
for i in range(num_dof):
    if dof_names[i] in key_joints:
        L.log(f"  {i:>3}  {dof_names[i]:25s}  {props[i]['stiffness']:>10.1f}  "
              f"{props[i]['damping']:>8.1f}  {props[i]['maxEffort']:>8.3f}  "
              f"{props[i]['maxVelocity']:>8.3f}")
L.log("")


# ── Apply gains and max efforts ───────────────────────────────────────────────
def configure_robot(kp, kd, max_effort):
    """Set gains and max efforts using the ArticulationController."""
    controller = nao.get_articulation_controller()

    kps = np.full(num_dof, kp, dtype=np.float32)
    kds = np.full(num_dof, kd, dtype=np.float32)
    controller.set_gains(kps, kds)

    max_effs = np.full(num_dof, max_effort, dtype=np.float32)
    controller.set_max_efforts(max_effs)

    # Verify
    p = nao.dof_properties
    actual_s = p[hip_idx]['stiffness']
    actual_d = p[hip_idx]['damping']
    actual_e = p[hip_idx]['maxEffort']
    L.log(f"  Config: Kp={kp}, Kd={kd}, maxEffort={max_effort}")
    L.log(f"  Verify hip: stiff={actual_s:.1f}, damp={actual_d:.1f}, maxEff={actual_e:.3f}")
    return actual_e > 0


# ── Standing test ─────────────────────────────────────────────────────────────
def run_test(pose, spawn_height, seconds, label=""):
    nao.set_world_pose(
        position=np.array([0.0, 0.0, spawn_height]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    nao.set_joint_positions(pose)
    nao.set_joint_velocities(np.zeros(num_dof))

    L.log(f"  --- {label} ---")

    total_steps = int(seconds / PHYSICS_DT)
    fell = False
    final_h = spawn_height

    header_printed = False

    for step_i in range(total_steps):
        action = ArticulationAction(joint_positions=pose)
        nao.apply_action(action)
        world.step(render=not args.headless)

        log_this = (step_i < 120 and step_i % 20 == 0) or \
                   (step_i >= 120 and step_i % int(0.5 / PHYSICS_DT) == 0)

        if log_this:
            if not header_printed:
                L.log(f"  {'Step':>6}  {'Time':>7}  {'Height':>8}  {'qw':>8}  "
                      f"{'Knee':>8}  {'Ankle':>8}  {'Hip':>8}  Note")
                L.log("  " + "-" * 78)
                header_printed = True

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
                elif h < 0.10:
                    note = "GROUND"
                    fell = True
                elif h > 0.28:
                    note = "OK"
                elif h > 0.20:
                    note = "low"

                L.log(f"  {step_i+1:>6}  {t:>6.3f}s  {h:>8.4f}  {qw:>+8.4f}  "
                      f"{k:>+7.1f}°  {a:>+7.1f}°  {hp:>+7.1f}°  {note}")
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


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Increase max effort and test
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("TEST 1: MaxEffort sweep (straight legs, spawn=0.335)")
L.log("=" * 80)
L.log("")

results = []

for max_eff in [5.0, 10.0, 20.0, 50.0, 100.0]:
    world.reset()
    configure_robot(kp=35810.0, kd=1000.0, max_effort=max_eff)
    L.log("")
    verdict, h = run_test(straight, 0.335, seconds=5.0,
                          label=f"maxEffort={max_eff}, straight, h=0.335")
    results.append(("straight", max_eff, verdict, h))

L.log("SUMMARY (straight legs):")
for pose_name, me, v, h in results:
    L.log(f"  maxEffort={me:>6.1f}  {v:>10}  h={h:.4f}")
L.log("")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Same with bent legs
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("TEST 2: MaxEffort sweep (bent legs, spawn=0.335)")
L.log("=" * 80)
L.log("")

results2 = []

for max_eff in [5.0, 10.0, 20.0, 50.0, 100.0]:
    world.reset()
    configure_robot(kp=35810.0, kd=1000.0, max_effort=max_eff)
    L.log("")
    verdict, h = run_test(home, 0.335, seconds=5.0,
                          label=f"maxEffort={max_eff}, bent, h=0.335")
    results2.append(("bent", max_eff, verdict, h))

L.log("SUMMARY (bent legs):")
for pose_name, me, v, h in results2:
    L.log(f"  maxEffort={me:>6.1f}  {v:>10}  h={h:.4f}")
L.log("")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Best config — extended confirmation
# ══════════════════════════════════════════════════════════════════════════════
all_results = results + results2
standing = [(p, me, v, h) for p, me, v, h in all_results if v == "STANDING"]

if standing:
    best = standing[0]
    L.log("=" * 80)
    L.log(f"BEST: {best[0]} legs, maxEffort={best[1]}")
    L.log(f"EXTENDED CONFIRMATION ({args.seconds}s)")
    L.log("=" * 80)
    L.log("")

    world.reset()
    configure_robot(kp=35810.0, kd=1000.0, max_effort=best[1])

    pose = straight if best[0] == "straight" else home
    L.log("")
    run_test(pose, 0.335, seconds=args.seconds,
             label=f"CONFIRM: {best[0]}, maxEffort={best[1]}")

    # Also try different spawn heights with the best config
    L.log("Height sweep with best config:")
    for h in [0.32, 0.33, 0.335, 0.34, 0.35]:
        run_test(pose, h, seconds=5.0,
                 label=f"Best config, h={h}")
else:
    L.log("No STANDING config found.")
    L.log("Trying even higher max efforts...")

    for max_eff in [200.0, 500.0, 1000.0]:
        world.reset()
        configure_robot(kp=35810.0, kd=1000.0, max_effort=max_eff)
        L.log("")
        v1, h1 = run_test(straight, 0.335, seconds=5.0,
                          label=f"maxEffort={max_eff}, straight")
        v2, h2 = run_test(home, 0.335, seconds=5.0,
                          label=f"maxEffort={max_eff}, bent")
        L.log(f"  >> straight: {v1} h={h1:.4f} | bent: {v2} h={h2:.4f}")
        if v1 == "STANDING" or v2 == "STANDING":
            break


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
