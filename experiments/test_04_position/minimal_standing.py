"""
minimal_standing.py — Minimal NAO Standing Test
=================================================

The sweep showed ALL configs at identical height (0.3397) — meaning
physics wasn't actually running or state was stale. Then the extended
test showed NaN from t=0.5s onward.

This script takes a different approach:
  1. ONE setup, NO world.reset() after initial setup
  2. Check state EVERY 10 steps (not every 60) to catch NaN onset
  3. Try apply_action() as alternative to set_joint_efforts()
  4. Try stepping without render to rule out render-related issues
  5. Log EVERYTHING to file

USAGE:
  python minimal_standing.py --headless
"""

import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--output", default="minimal_results.txt")
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
L.log("MINIMAL NAO STANDING TEST")
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

# Only ONE world.reset() — at startup
world.reset()

dof_names = nao.dof_names
num_dof = nao.num_dof
PHYSICS_DT = 1.0 / 120.0

L.log(f"NAO: {num_dof} DOFs at {nao_prim_path}")
L.log(f"DOFs: {dof_names}")
L.log("")

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

knee_idx = dof_names.index("LKneePitch")
ankle_idx = dof_names.index("LAnklePitch")
hip_idx = dof_names.index("LHipPitch")


# ══════════════════════════════════════════════════════════════════════════════
# TEST A: No control, no reset — just step physics and observe
# ══════════════════════════════════════════════════════════════════════════════
L.log("=" * 80)
L.log("TEST A: Step physics with NO control (ragdoll baseline)")
L.log("  Goal: See if physics actually runs and when NaN appears")
L.log("=" * 80)
L.log("")

# Set initial pose (no world.reset — already done above)
nao.set_world_pose(
    position=np.array([0.0, 0.0, 0.335]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
)
nao.set_joint_positions(home)
nao.set_joint_velocities(np.zeros(num_dof))

L.log(f"  {'Step':>6}  {'Time':>7}  {'Height':>8}  {'qw':>8}  {'Knee':>8}  {'KneeVel':>10}  Note")
L.log("  " + "-" * 75)

# Step very frequently for the first 2 seconds
for step_i in range(int(2.0 / PHYSICS_DT)):
    world.step(render=False)  # headless stepping only

    # Log every 10 steps (~0.083s) for first 0.5s, then every 60 steps
    log_this = (step_i < 60 and step_i % 10 == 0) or (step_i >= 60 and step_i % 60 == 0)

    if log_this:
        t = (step_i + 1) * PHYSICS_DT
        try:
            pos, quat = nao.get_world_pose()
            jp = nao.get_joint_positions()
            jv = nao.get_joint_velocities()

            h = float(pos[2]) if pos is not None else float('nan')
            qw = float(quat[3]) if quat is not None else float('nan')
            k = math.degrees(float(jp[knee_idx])) if jp is not None else float('nan')
            kv = math.degrees(float(jv[knee_idx])) if jv is not None else float('nan')

            note = ""
            if np.isnan(h):
                note = "<-- NaN!"
            elif h < 0.15:
                note = "<-- GROUND"

            L.log(f"  {step_i+1:>6}  {t:>6.3f}s  {h:>8.4f}  {qw:>+8.4f}  "
                  f"{k:>+7.1f}°  {kv:>+9.1f}°/s  {note}")
        except Exception as e:
            L.log(f"  {step_i+1:>6}  ERROR: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST B: PD control via set_joint_efforts — single clean run
# ══════════════════════════════════════════════════════════════════════════════
L.log("")
L.log("=" * 80)
L.log("TEST B: PD control via set_joint_efforts()")
L.log("  Kp=10, Kd=1.0, MaxTorque=3Nm")
L.log("  NO world.reset() — using post_reset() instead")
L.log("=" * 80)
L.log("")

# Try post_reset instead of world.reset
try:
    nao.post_reset()
    L.log("  nao.post_reset() succeeded")
except Exception as e:
    L.log(f"  nao.post_reset() failed: {e}")

nao.set_world_pose(
    position=np.array([0.0, 0.0, 0.335]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
)
nao.set_joint_positions(home)
nao.set_joint_velocities(np.zeros(num_dof))

KP = 10.0
KD = 1.0
MAX_T = 3.0

L.log(f"  {'Step':>6}  {'Time':>7}  {'Height':>8}  {'qw':>8}  {'Knee':>8}  "
      f"{'TqKnee':>8}  {'TqAnkle':>8}  Note")
L.log("  " + "-" * 80)

for step_i in range(int(5.0 / PHYSICS_DT)):
    jp = nao.get_joint_positions()
    jv = nao.get_joint_velocities()

    # Compute PD torques
    if jp is not None and jv is not None:
        err = home - jp
        torques = KP * err - KD * jv
        torques = np.clip(torques, -MAX_T, MAX_T)
        torques = np.nan_to_num(torques, nan=0.0)
    else:
        torques = np.zeros(num_dof, dtype=np.float32)

    nao.set_joint_efforts(torques)
    world.step(render=False)

    log_this = (step_i < 60 and step_i % 10 == 0) or (step_i >= 60 and step_i % 60 == 0)
    if log_this:
        t = (step_i + 1) * PHYSICS_DT
        try:
            pos, quat = nao.get_world_pose()
            h = float(pos[2]) if pos is not None else float('nan')
            qw = float(quat[3]) if quat is not None else float('nan')
            k = math.degrees(float(jp[knee_idx])) if jp is not None else float('nan')
            tq_k = torques[knee_idx]
            tq_a = torques[ankle_idx]

            note = ""
            if np.isnan(h):
                note = "<-- NaN!"
            elif h < 0.15:
                note = "<-- GROUND"
            elif h > 0.30:
                note = "OK"

            L.log(f"  {step_i+1:>6}  {t:>6.3f}s  {h:>8.4f}  {qw:>+8.4f}  "
                  f"{k:>+7.1f}°  {tq_k:>+7.2f}  {tq_a:>+7.2f}  {note}")
        except Exception as e:
            L.log(f"  {step_i+1:>6}  ERROR: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST C: apply_action() — the ArticulationAction interface
# ══════════════════════════════════════════════════════════════════════════════
L.log("")
L.log("=" * 80)
L.log("TEST C: apply_action() with joint efforts")
L.log("  Uses ArticulationAction(joint_efforts=...) instead of set_joint_efforts()")
L.log("=" * 80)
L.log("")

# Fresh reset via world
world.reset()

nao.set_world_pose(
    position=np.array([0.0, 0.0, 0.335]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
)
nao.set_joint_positions(home)
nao.set_joint_velocities(np.zeros(num_dof))

L.log(f"  {'Step':>6}  {'Time':>7}  {'Height':>8}  {'qw':>8}  {'Knee':>8}  "
      f"{'TqKnee':>8}  Note")
L.log("  " + "-" * 80)

for step_i in range(int(5.0 / PHYSICS_DT)):
    jp = nao.get_joint_positions()
    jv = nao.get_joint_velocities()

    if jp is not None and jv is not None:
        err = home - jp
        torques = KP * err - KD * jv
        torques = np.clip(torques, -MAX_T, MAX_T)
        torques = np.nan_to_num(torques, nan=0.0)
    else:
        torques = np.zeros(num_dof, dtype=np.float32)

    # Use apply_action instead
    action = ArticulationAction(joint_efforts=torques)
    nao.apply_action(action)
    world.step(render=False)

    log_this = (step_i < 60 and step_i % 10 == 0) or (step_i >= 60 and step_i % 60 == 0)
    if log_this:
        t = (step_i + 1) * PHYSICS_DT
        try:
            pos, quat = nao.get_world_pose()
            h = float(pos[2]) if pos is not None else float('nan')
            qw = float(quat[3]) if quat is not None else float('nan')
            k = math.degrees(float(jp[knee_idx])) if jp is not None else float('nan')
            tq_k = torques[knee_idx]

            note = ""
            if np.isnan(h):
                note = "<-- NaN!"
            elif h < 0.15:
                note = "<-- GROUND"
            elif h > 0.30:
                note = "OK"

            L.log(f"  {step_i+1:>6}  {t:>6.3f}s  {h:>8.4f}  {qw:>+8.4f}  "
                  f"{k:>+7.1f}°  {tq_k:>+7.2f}  {note}")
        except Exception as e:
            L.log(f"  {step_i+1:>6}  ERROR: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST D: apply_action with joint_positions (position control mode)
# ══════════════════════════════════════════════════════════════════════════════
L.log("")
L.log("=" * 80)
L.log("TEST D: apply_action() with joint_positions (position mode)")
L.log("  This tells the articulation controller to drive to positions.")
L.log("  May require internal drives to be configured — testing anyway.")
L.log("=" * 80)
L.log("")

world.reset()

nao.set_world_pose(
    position=np.array([0.0, 0.0, 0.335]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
)
nao.set_joint_positions(home)
nao.set_joint_velocities(np.zeros(num_dof))

L.log(f"  {'Step':>6}  {'Time':>7}  {'Height':>8}  {'qw':>8}  {'Knee':>8}  Note")
L.log("  " + "-" * 80)

for step_i in range(int(5.0 / PHYSICS_DT)):
    action = ArticulationAction(joint_positions=home)
    nao.apply_action(action)
    world.step(render=False)

    log_this = (step_i < 60 and step_i % 10 == 0) or (step_i >= 60 and step_i % 60 == 0)
    if log_this:
        t = (step_i + 1) * PHYSICS_DT
        try:
            pos, quat = nao.get_world_pose()
            jp = nao.get_joint_positions()
            h = float(pos[2]) if pos is not None else float('nan')
            qw = float(quat[3]) if quat is not None else float('nan')
            k = math.degrees(float(jp[knee_idx])) if jp is not None else float('nan')

            note = ""
            if np.isnan(h):
                note = "<-- NaN!"
            elif h < 0.15:
                note = "<-- GROUND"
            elif h > 0.30:
                note = "OK"

            L.log(f"  {step_i+1:>6}  {t:>6.3f}s  {h:>8.4f}  {qw:>+8.4f}  "
                  f"{k:>+7.1f}°  {note}")
        except Exception as e:
            L.log(f"  {step_i+1:>6}  ERROR: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST E: Check dof_properties — might reveal drive config
# ══════════════════════════════════════════════════════════════════════════════
L.log("")
L.log("=" * 80)
L.log("TEST E: DOF Properties")
L.log("=" * 80)
L.log("")

try:
    props = nao.dof_properties
    L.log(f"  Type: {type(props)}")
    if props is not None:
        if hasattr(props, 'dtype'):
            L.log(f"  Dtype: {props.dtype}")
            L.log(f"  Shape: {props.shape}")
            if hasattr(props.dtype, 'names') and props.dtype.names:
                L.log(f"  Fields: {props.dtype.names}")
                for i in range(min(5, len(props))):
                    L.log(f"  DOF {i} ({dof_names[i]}):")
                    for field in props.dtype.names:
                        L.log(f"    {field}: {props[i][field]}")
            else:
                L.log(f"  First 3 rows: {props[:3]}")
        else:
            L.log(f"  Content: {props}")
except Exception as e:
    L.log(f"  Error: {e}")


L.log("")
L.log("=" * 80)
L.log("ALL TESTS COMPLETE")
L.log("=" * 80)
L.close()

try:
    simulation_app.close()
except:
    pass

L.dump()
