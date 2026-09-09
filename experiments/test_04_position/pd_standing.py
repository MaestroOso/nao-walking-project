"""
pd_standing.py — Manual PD Controller for NAO Standing (v2 — file output)
===========================================================================

All results are written to a clean text file (default: pd_results.txt).
After Isaac Sim shuts down, results are also printed to console.

USAGE:
  python pd_standing.py --sweep --headless
  python pd_standing.py --sweep --headless --output results.txt
"""

import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--sweep", action="store_true")
parser.add_argument("--kp", type=float, default=10.0)
parser.add_argument("--kd", type=float, default=1.0)
parser.add_argument("--max-torque", type=float, default=5.0)
parser.add_argument("--output", default="pd_results.txt")
args = parser.parse_args()


class ResultLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.lines = []
        self.f = open(filepath, 'w')

    def log(self, msg=""):
        self.lines.append(msg)
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        self.f.close()

    def dump_to_console(self):
        print("\n" + "=" * 80)
        print(f"RESULTS FROM: {self.filepath}")
        print("=" * 80)
        for line in self.lines:
            print(line)


log = ResultLogger(args.output)
log.log("=" * 80)
log.log("NAO PD STANDING TEST RESULTS")
log.log("=" * 80)
log.log("")

# ── Isaac Sim startup ─────────────────────────────────────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
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

log.log(f"NAO: {num_dof} DOFs at {nao_prim_path}")
log.log("")


# ── Home pose ─────────────────────────────────────────────────────────────────
def build_home_pose(hip=-25, knee=45, ankle=-10):
    pose = np.zeros(num_dof, dtype=np.float32)
    joints = {
        "LHipPitch": hip, "RHipPitch": hip,
        "LKneePitch": knee, "RKneePitch": knee,
        "LAnklePitch": ankle, "RAnklePitch": ankle,
        "LShoulderPitch": 84, "RShoulderPitch": 84,
        "LShoulderRoll": 10, "RShoulderRoll": -10,
        "LElbowYaw": -70, "RElbowYaw": 70,
        "LElbowRoll": -25, "RElbowRoll": 25,
    }
    for name, deg in joints.items():
        if name in dof_names:
            pose[dof_names.index(name)] = math.radians(deg)
    return pose


# ── PD Controller ─────────────────────────────────────────────────────────────
class PDController:
    def __init__(self, num_dof, kp=10.0, kd=1.0, max_torque=5.0):
        self.num_dof = num_dof
        self.kp = np.full(num_dof, kp, dtype=np.float32)
        self.kd = np.full(num_dof, kd, dtype=np.float32)
        self.max_torque = max_torque
        self.targets = np.zeros(num_dof, dtype=np.float32)

    def set_targets(self, targets):
        self.targets = targets.copy()

    def compute_torques(self, positions, velocities):
        if positions is None or velocities is None:
            return np.zeros(self.num_dof, dtype=np.float32)
        pos_error = self.targets - positions
        torques = self.kp * pos_error - self.kd * velocities
        torques = np.clip(torques, -self.max_torque, self.max_torque)
        torques = np.nan_to_num(torques, nan=0.0, posinf=0.0, neginf=0.0)
        return torques


# ── Index lookups ─────────────────────────────────────────────────────────────
knee_idx = dof_names.index("LKneePitch") if "LKneePitch" in dof_names else 0
ankle_idx = dof_names.index("LAnklePitch") if "LAnklePitch" in dof_names else 0
hip_idx = dof_names.index("LHipPitch") if "LHipPitch" in dof_names else 0
PHYSICS_DT = 1.0 / 120.0


# ── Run a standing trial ──────────────────────────────────────────────────────
def run_pd_trial(kp, kd, max_torque, seconds, spawn_height=0.335, verbose=True):
    total_steps = int(seconds / PHYSICS_DT)
    print_interval = int(0.5 / PHYSICS_DT)

    pd = PDController(num_dof, kp=kp, kd=kd, max_torque=max_torque)
    home = build_home_pose()
    pd.set_targets(home)

    world.reset()
    nao.set_world_pose(
        position=np.array([0.0, 0.0, spawn_height]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    nao.set_joint_positions(home)
    nao.set_joint_velocities(np.zeros(num_dof))

    if verbose:
        log.log(f"  PD: Kp={kp}, Kd={kd}, MaxTorque={max_torque}Nm, duration={seconds}s")
        log.log(f"  {'Time':>6}  {'Height':>8}  {'qw':>8}  {'Knee°':>8}  "
                f"{'Ankle°':>8}  {'Hip°':>8}  {'TqKnee':>8}  Status")
        log.log("  " + "-" * 80)

    fell = False
    final_h = spawn_height

    for step_i in range(total_steps):
        jp = nao.get_joint_positions()
        jv = nao.get_joint_velocities()
        torques = pd.compute_torques(jp, jv)
        nao.set_joint_efforts(torques)
        world.step(render=not args.headless)

        if step_i % print_interval == 0:
            try:
                pos, quat = nao.get_world_pose()
                h = float(pos[2]) if pos is not None else float('nan')
                qw = abs(float(quat[3])) if quat is not None else float('nan')
                t = step_i * PHYSICS_DT

                k_deg = math.degrees(jp[knee_idx]) if jp is not None else float('nan')
                a_deg = math.degrees(jp[ankle_idx]) if jp is not None else float('nan')
                h_deg = math.degrees(jp[hip_idx]) if jp is not None else float('nan')
                tq_k = torques[knee_idx]

                if not np.isnan(h):
                    final_h = h

                h_bad = (not np.isnan(h)) and h < 0.10
                qw_bad = (not np.isnan(qw)) and qw < 0.5

                if h_bad or qw_bad:
                    fell = True
                    if verbose:
                        log.log(f"  {t:>5.1f}s  {h:>8.4f}  {qw:>8.4f}  {k_deg:>+7.1f}°  "
                                f"{a_deg:>+7.1f}°  {h_deg:>+7.1f}°  {tq_k:>+7.2f}  FELL")
                    break

                if verbose:
                    if np.isnan(h) or np.isnan(qw):
                        status = "NaN"
                    elif h < 0.20:
                        status = "LOW"
                    else:
                        status = "OK"
                    log.log(f"  {t:>5.1f}s  {h:>8.4f}  {qw:>8.4f}  {k_deg:>+7.1f}°  "
                            f"{a_deg:>+7.1f}°  {h_deg:>+7.1f}°  {tq_k:>+7.2f}  {status}")
            except Exception as e:
                if verbose:
                    log.log(f"  t={step_i * PHYSICS_DT:.1f}s  ERROR: {e}")

    if verbose:
        if not fell:
            log.log(f"  SURVIVED {seconds}s (final height: {final_h:.4f}m)")
        log.log("")

    return not fell, final_h


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if args.sweep:
    log.log("=" * 80)
    log.log("PD GAIN SWEEP (5s per config)")
    log.log("=" * 80)
    log.log("")

    configs = [
        (2.0,   0.2,  2.0),
        (5.0,   0.5,  3.0),
        (10.0,  1.0,  3.0),
        (10.0,  1.0,  5.0),
        (20.0,  2.0,  5.0),
        (20.0,  2.0,  10.0),
        (30.0,  3.0,  5.0),
        (50.0,  5.0,  5.0),
        (50.0,  5.0,  10.0),
        (100.0, 10.0, 10.0),
        (100.0, 10.0, 20.0),
        (200.0, 20.0, 20.0),
    ]

    log.log("SWEEP SUMMARY:")
    log.log(f"  {'Kp':>8}  {'Kd':>6}  {'MaxT':>6}  {'Result':>8}  {'FinalH':>10}")
    log.log("  " + "-" * 50)

    sweep_results = []
    for kp, kd, mt in configs:
        survived, final_h = run_pd_trial(kp, kd, mt, seconds=5.0, verbose=False)
        tag = "STABLE" if survived else "FELL"
        log.log(f"  {kp:>8.1f}  {kd:>6.1f}  {mt:>6.1f}  {tag:>8}  {final_h:>10.4f}")
        sweep_results.append((kp, kd, mt, survived, final_h))

    log.log("")

    stable = [(kp, kd, mt, h) for kp, kd, mt, s, h in sweep_results if s]
    if stable:
        best = stable[0]
        log.log(f"BEST (lowest gains): Kp={best[0]}, Kd={best[1]}, MaxTorque={best[2]}Nm")
        log.log("")
        log.log("=" * 80)
        log.log("EXTENDED CONFIRMATION (10s with best config)")
        log.log("=" * 80)
        log.log("")
        run_pd_trial(best[0], best[1], best[2], seconds=10.0, verbose=True)
    else:
        log.log("NO STABLE CONFIG FOUND")
        log.log("")
        log.log("Full telemetry for most aggressive config:")
        log.log("")
        run_pd_trial(200.0, 20.0, 20.0, seconds=5.0, verbose=True)

else:
    log.log(f"Single test: Kp={args.kp}, Kd={args.kd}, MaxTorque={args.max_torque}")
    log.log("")
    run_pd_trial(args.kp, args.kd, args.max_torque, seconds=args.seconds, verbose=True)

log.log("")
log.log("=" * 80)
log.log("DONE")
log.log("=" * 80)
log.close()

try:
    simulation_app.close()
except:
    pass

# Print clean results AFTER all Isaac Sim noise is done
log.dump_to_console()
