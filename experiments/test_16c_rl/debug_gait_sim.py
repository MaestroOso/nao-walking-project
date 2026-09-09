"""
debug_gait_sim.py — Watch joint positions, CoM, and XY position during gait.
=============================================================================
Writes ALL results to a file (separate from Isaac Sim log noise).
Supports --lateral-only to test hip roll only (no pitch) [v1 only].
Supports --gait-version to select v1, v2, v3, or v4.

USAGE:
  # v1 (original gait)
  python debug_gait_sim.py --headless                           # full v1 gait
  python debug_gait_sim.py --headless --lateral-only            # v1 roll only

  # v2 iter 2 (conservative active-stance) — proven stable, ~0 forward
  python debug_gait_sim.py --headless --gait-version v2 --steps 1300

  # v3 (monotonic-stance walking) — large amplitude, falls
  python debug_gait_sim.py --headless --gait-version v3 --steps 1300

  # v4 (persistent-landed-pose walking) — current development target
  # Combines V2 stability with small monotonic stance retreat
  python debug_gait_sim.py --headless --gait-version v4 --steps 1300
  python debug_gait_sim.py --headless --gait-version v4 --steps 3060
"""
import argparse, numpy as np, math

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default="../../robots/nao_isaac_sim/nao.urdf")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--out", default="debug_gait_sim_results.txt")
parser.add_argument("--lateral-only", action="store_true")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--gait-version", choices=["v1", "v2", "v3", "v4"], default="v1",
                    help="v1=original ReferenceGait, v2=ReferenceGaitV2 (active-stance), v3=ReferenceGaitV3 (monotonic-stance), v4=ReferenceGaitV4 (persistent-landed-pose)")
parser.add_argument("--gait-hz", type=float, default=None,
                    help="Override gait frequency (v1 default=0.2, v2 default=0.1)")
args = parser.parse_args()

# Open file BEFORE any Isaac imports — captures output even if init crashes
out_f = open(args.out, "w", buffering=1)

def log(msg):
    out_f.write(msg + "\n")
    out_f.flush()

mode_str = "LATERAL ONLY" if args.lateral_only else "FULL GAIT"
log(f"=== debug_gait_sim starting ({mode_str}, gait={args.gait_version}, {args.steps} steps) ===")

if args.lateral_only and args.gait_version in ("v2", "v3", "v4"):
    log(f"WARNING: --lateral-only is not supported by {args.gait_version} gait; ignoring.")
    args.lateral_only = False

try:
    from nao_walking_env import NaoWalkingEnv, ReferenceGait, ReferenceGaitV2, ReferenceGaitV3, ReferenceGaitV4
    log("imports OK")

    env_kwargs = dict(
        urdf_path=args.urdf,
        render_mode="headless" if args.headless else "human",
        max_episode_steps=args.steps + 50,
        grace_period=10000,       # never terminate — we want to observe
        term_height=0.0,
        term_orient_qw=0.0,
        gait_lateral_only=args.lateral_only,
        gait_version=args.gait_version,
    )
    if args.gait_hz is not None:
        env_kwargs["gait_hz"] = args.gait_hz

    env = NaoWalkingEnv(**env_kwargs)
    log("env created OK")

    obs, _ = env.reset()
    log("reset OK")

    # Build a local gait instance matching the env's gait for logging target values.
    if args.gait_version == "v4":
        gait = ReferenceGaitV4(gait_hz=env.gait_hz)
    elif args.gait_version == "v3":
        gait = ReferenceGaitV3(gait_hz=env.gait_hz)
    elif args.gait_version == "v2":
        gait = ReferenceGaitV2(gait_hz=env.gait_hz)
    else:
        gait = ReferenceGait(gait_hz=env.gait_hz, lateral_only=args.lateral_only)

    # Record starting position
    start_pos, _ = env._nao.get_world_pose()
    start_x = float(start_pos[0]) if start_pos is not None else 0.0
    start_y = float(start_pos[1]) if start_pos is not None else 0.0
    log(f"Start position: x={start_x:.4f}m  y={start_y:.4f}m")
    log(f"Gait: {args.gait_version} @ {env.gait_hz} Hz "
        f"(cycle period = {1.0/env.gait_hz:.1f}s, "
        f"control_dt = {env.control_dt:.4f}s, "
        f"steps per cycle = {int(round(1.0/env.gait_hz/env.control_dt))})")
    log("")

    # Per-cycle tracking: we record the robot's X position at each cycle
    # boundary (every time phase wraps from near-1 back to near-0).
    cycle_records = []  # list of (cycle_num, step, x_at_cycle_end, dx_this_cycle, h)
    last_phase = 0.0
    cycle_start_x = start_x
    cycle_count = 0

    # Ankle tracking lag monitor — records max absolute error between
    # commanded and actual ankle pitch across the whole run. This is the
    # direct test of the "ankle can't keep up with stance push" hypothesis.
    max_r_ankle_err = 0.0
    max_l_ankle_err = 0.0
    max_r_ankle_err_step = -1
    max_l_ankle_err_step = -1

    # Header
    log(f"{'Step':>5} {'Height':>7} {'Tilt':>6} {'X':>8} {'Y':>8} | "
        f"{'Rhip':>6} {'Lhip':>6} {'Rknee':>6} {'Lknee':>6} {'Rank':>6} {'Lank':>6} | "
        f"{'Rroll':>6} {'Lroll':>6} | {'Phase':>6} | "
        f"{'cmd_Rh':>7} {'cmd_Lh':>7} {'cmd_Ra':>7} {'cmd_La':>7}")
    log("-" * 145)

    for step in range(args.steps):
        obs, r, term, trunc, info = env.step(np.zeros(12, dtype=np.float32))

        pos, quat = env._nao.get_world_pose()
        h    = float(pos[2]) if pos is not None else -1.0
        x    = float(pos[0]) if pos is not None else 0.0
        y    = float(pos[1]) if pos is not None else 0.0
        pg   = env._quat_rotate_inverse(quat, np.array([0, 0, -1.0]))
        tilt = float(np.sqrt(pg[0]**2 + pg[1]**2))

        # Displacement from start
        dx = x - start_x
        dy = y - start_y

        jp = env._nao.get_joint_positions()
        if jp is not None:
            lp    = [jp[env._leg_dof_indices[i]] for i in range(12)]
            rhip  = math.degrees(lp[8])
            lhip  = math.degrees(lp[2])
            rknee = math.degrees(lp[9])
            lknee = math.degrees(lp[3])
            rroll = math.degrees(lp[7])
            lroll = math.degrees(lp[1])
            rankp = math.degrees(lp[10])   # RAnklePitch
            lankp = math.degrees(lp[4])    # LAnklePitch
        else:
            rhip = lhip = rknee = lknee = rroll = lroll = 0.0
            rankp = lankp = 0.0
            log(f"  WARNING step {step}: get_joint_positions() returned None")

        if step >= env.gait_warmup_steps:
            phase   = gait.get_phase((step - env.gait_warmup_steps) * env.control_dt)
            targets = gait.get_targets(phase)
            t_rhip  = math.degrees(targets[8])
            t_lhip  = math.degrees(targets[2])
            t_rroll = math.degrees(targets[7])
            t_lroll = math.degrees(targets[1])
            t_rank  = math.degrees(targets[10])   # cmd RAnklePitch
            t_lank  = math.degrees(targets[4])    # cmd LAnklePitch
        else:
            phase = 0.0
            t_rhip = t_lhip = t_rroll = t_lroll = 0.0
            t_rank = t_lank = 0.0

        # Detect cycle boundary: phase wrapped from near-1 back to near-0
        cycle_boundary = False
        if step >= env.gait_warmup_steps and last_phase > 0.9 and phase < 0.1:
            cycle_count += 1
            dx_cycle = x - cycle_start_x
            cycle_records.append({
                "cycle": cycle_count,
                "step": step,
                "x_end": x,
                "dx_cycle": dx_cycle,
                "h": h,
            })
            cycle_start_x = x
            cycle_boundary = True
        last_phase = phase

        # Track max ankle pitch tracking error (only while the gait is active
        # and the robot hasn't fallen — otherwise the error is meaningless)
        if step >= env.gait_warmup_steps and h > 0.20:
            r_err = abs(rankp - t_rank)
            l_err = abs(lankp - t_lank)
            if r_err > max_r_ankle_err:
                max_r_ankle_err = r_err
                max_r_ankle_err_step = step
            if l_err > max_l_ankle_err:
                max_l_ankle_err = l_err
                max_l_ankle_err_step = step

        markers = []
        if step == env.gait_warmup_steps: markers.append("GAIT START")
        if cycle_boundary:                markers.append(f"CYCLE {cycle_count} END (dx={dx_cycle*100:+.2f}cm)")
        if h < 0.28:  markers.append("LOW")
        if h < 0.20:  markers.append("FALLING")
        if h < 0.10:  markers.append("FALLEN")
        marker_str = "  <-- " + ", ".join(markers) if markers else ""

        # Print cadence:
        #  - v1: every step from step 55 onward (matches original behavior)
        #  - v2/v3: every 10 steps (otherwise we flood the log at ~600 steps/cycle)
        #  - Always print on cycle boundaries and on GAIT START
        should_print = False
        if cycle_boundary or step == env.gait_warmup_steps:
            should_print = True
        elif args.gait_version in ("v2", "v3", "v4"):
            should_print = (step % 10 == 0)
        else:
            should_print = (step >= 55 or step % 10 == 0)

        if should_print:
            log(f"{step:>5} {h:>7.3f} {tilt:>6.3f} {dx:>8.4f} {dy:>8.4f} | "
                f"{rhip:>6.1f} {lhip:>6.1f} {rknee:>6.1f} {lknee:>6.1f} "
                f"{rankp:>6.1f} {lankp:>6.1f} | "
                f"{rroll:>6.1f} {lroll:>6.1f} | {phase:>6.3f} | "
                f"{t_rhip:>7.1f} {t_lhip:>7.1f} {t_rank:>7.1f} {t_lank:>7.1f}"
                f"{marker_str}")

    log("")
    log("=== loop complete ===")

    # Final summary
    pos, _ = env._nao.get_world_pose()
    fh = float(pos[2]) if pos is not None else -1.0
    fx = float(pos[0]) - start_x if pos is not None else 0.0
    fy = float(pos[1]) - start_y if pos is not None else 0.0
    log(f"Final height:      {fh:.4f}m")
    log(f"Total X travel:    {fx:+.4f}m  ({'forward' if fx > 0 else 'backward'})")
    log(f"Total Y drift:     {fy:+.4f}m  ({'left' if fy > 0 else 'right'})")
    log(f"Steps run:         {args.steps}")
    log(f"Cycles completed:  {cycle_count}")

    # Per-cycle breakdown
    if cycle_records:
        log("")
        log("=== PER-CYCLE PROGRESS ===")
        log(f"{'Cycle':>6} {'Step':>6} {'x_end':>10} {'dx_cycle':>10} {'height':>8}")
        log("-" * 48)
        for rec in cycle_records:
            direction = "→" if rec["dx_cycle"] > 0 else "←"
            log(f"{rec['cycle']:>6} {rec['step']:>6} "
                f"{rec['x_end']:>+9.4f}m {rec['dx_cycle']*100:>+8.2f}cm {direction} "
                f"{rec['h']:>7.4f}m")
        dx_mean_cm = 100.0 * sum(r["dx_cycle"] for r in cycle_records) / len(cycle_records)
        log(f"\nMean forward progress per cycle: {dx_mean_cm:+.2f} cm")
        # Direction consistency: all cycles moving the same way?
        n_forward  = sum(1 for r in cycle_records if r["dx_cycle"] > 0.005)
        n_backward = sum(1 for r in cycle_records if r["dx_cycle"] < -0.005)
        log(f"Cycles forward: {n_forward}, backward: {n_backward}, "
            f"negligible: {len(cycle_records) - n_forward - n_backward}")

    # Ankle tracking diagnostic — directly tests whether the ankle PD gains
    # are now sufficient to track the commanded stance push.
    log("")
    log("=== ANKLE PITCH TRACKING DIAGNOSTIC ===")
    log(f"  Max R ankle error: {max_r_ankle_err:.2f}° "
        f"(at step {max_r_ankle_err_step})")
    log(f"  Max L ankle error: {max_l_ankle_err:.2f}° "
        f"(at step {max_l_ankle_err_step})")
    worst = max(max_r_ankle_err, max_l_ankle_err)
    if worst < 1.0:
        log(f"  → Tracking is TIGHT ({worst:.2f}° max). Ankles are keeping up with commands.")
    elif worst < 3.0:
        log(f"  → Tracking is ACCEPTABLE ({worst:.2f}° max). Minor lag.")
    elif worst < 8.0:
        log(f"  → Tracking is LOOSE ({worst:.2f}° max). Ankles lagging commands noticeably.")
    else:
        log(f"  → Tracking is POOR ({worst:.2f}° max). Ankles cannot keep up with commands.")
    log(f"  (Errors only measured while h > 0.20m, i.e. while robot was upright.)")

    log("")
    if fh > 0.28:
        log("VERDICT (stability): STABLE ✓ — gait did not destabilize robot")
    elif fh > 0.15:
        log("VERDICT (stability): LOW — leaning but not fallen")
    else:
        log("VERDICT (stability): FELL ✗ — gait destabilizes robot")

    if abs(fx) > 0.01:
        direction = "FORWARD ✓" if fx > 0 else "BACKWARD ✗"
        log(f"VERDICT (direction): {direction} — {abs(fx)*100:.1f}cm in {args.steps} steps")
        if fx > 0:
            log("  → Gait moves the robot forward. Ready for RL residuals.")
        else:
            log("  → Gait still moves backward. Needs amplitude/phase tuning.")
    else:
        log("VERDICT (direction): NO SIGNIFICANT MOTION")
        log("  → Gait may be too conservative. Try increasing stance push amplitudes.")

    env.close()
    log("=== env closed ===")

except Exception as e:
    import traceback
    log(f"\n=== EXCEPTION ===\n{traceback.format_exc()}")

finally:
    out_f.close()

print(f"Done. Results in: {args.out}")
