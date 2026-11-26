#!/bin/bash
# test_26_sac pilot: SAC, SB3 2.7.1 defaults, net_arch/gamma matched to
# test_24 for comparability. Time-bounded, not a 50M-step commitment —
# see APPROACH.md.
#
# v2: REWARD REVISED (body-frame velocity + yaw-rate penalty, see
# APPROACH.md "Reward revision"). The original lineage (RUN_NAME
# "sac_pilot", 145,000 -> 7,060,000 steps across 491 chunk attempts,
# committed) trained under the old world-frame-only reward and is left
# untouched under runs/sac_pilot/ as the historical record of why the
# reward changed. This is a FRESH start — new SAC(), empty replay
# buffer, new run name "sac_pilot_v2" — not a resume, since the old
# buffer's stored rewards are computed under the old formula and would
# be wrong under the new one. CHUNK_STEPS=15,000 carried over unchanged
# (proven across 491 chunk attempts in the original run, one transient
# GPU-driver hang recovered cleanly by the existing retry logic).
#
# Replay-buffer RAM footprint (computed from actual obs/action dims):
#   obs=23, action=12, buffer_size=1e6, float32, handle_timeout_termination=True:
#   observations 92MB + next_observations 92MB + actions 48MB + rewards/dones/timeouts 12MB
#   = ~244MB (worst-case ~2x transient during pickle load/reconstruct ~490MB).
#   Comfortably inside the MIN_RAM_MB gate below (Isaac Sim itself, not the
#   buffer, is what that gate is actually protecting against).

# v3: QUATERNION CONVENTION FIX (see APPROACH.md "Quaternion convention
# fix"). Isaac Sim quaternions are (w,x,y,z); every env through v2 read
# them as (x,y,z,w), so side tilt was invisible to the policy/penalty and
# v2's "body-frame" velocity was effectively still world-frame. The
# observation semantics change (projected gravity, ang vel, heading-frame
# lin_vel), so this is again a FRESH start under a new run name;
# runs/sac_pilot_v2/ (4,995,000 steps, v2 env) is left untouched.
# v3 also LOOPS the reference gait for the whole episode (test_27 looped
# clip, same as PGRL/test_28) instead of holding its last frame after 2.6 s.
# The v2 notes below (MAX_WALLCLOCK comment) describe that previous run.
#
# v4: WARMUP ACTION MASKING (see APPROACH.md "Warmup action masking").
# sac_pilot_v3 (675,000 steps, 8 h) used its residual actions during the
# 60-step warmup stand to lean the robot 6-8 deg sideways before the gait
# started. v4 ignores actions during warmup (env warmup_actions=False);
# everything else is identical to v3. FRESH start: v3's replay buffer holds
# warmup transitions whose actions moved the robot, which is no longer true.
# runs/sac_pilot_v3/ is left untouched.
#
# v5: FALL PENALTY (see APPROACH.md "Fall penalty"). sac_pilot_v4 (720,000
# steps, 8 h) learned to lunge: it fell after ~1.1 s of gait vs 2.0 s for
# the zero-action reference, but scored more because the extra distance
# outweighed the lost alive reward and falling itself cost nothing. v5 adds
# a one-off -100 reward on the falling step (env penalty_fall=100); every
# other setting is identical to v4 (tilt weight still 0.5). FRESH start:
# v4's replay buffer stores rewards without the penalty.
#
# v6: CONSTANT ALIVE BONUS (see APPROACH.md "Constant alive bonus + stronger yaw-rate penalty (v6)").
# sac_pilot_v5 (1,230,000 steps, 12 h) stopped falling but only shuffled
# (~0.06 m/s): the growing alive bonus 1 + step/500 (up to 4/step) made
# survival worth ~8x the forward reward. v6 uses a constant 1.0/step
# (env alive_scale=None), AND raises the yaw-rate penalty 0.5 -> 5.0
# (v5 slowly turned ~90 deg clockwise; a slow turn cost ~0.05/step).
# Everything else is identical to v5 (fall penalty 100, tilt 0.5).
# FRESH start: v5's buffer has the old rewards.
#
# v7: YAW-RATE PENALTY BACK TO 0.5, constant alive bonus kept (see
# APPROACH.md "sac_pilot_v6 outcome and v7"). sac_pilot_v6 (1,275,000
# steps) never survived past ~0.7 s of gait: at w=5 the gait's own yaw
# oscillation cost more per step than the alive bonus, so falling early
# scored best. v7 = v5 with only the constant alive bonus changed.
# FRESH start: v6's buffer has the w=5 rewards.
#
# v8: TIGHTER FALL DETECTION (see APPROACH.md "sac_pilot_v7 outcome and
# v8"). sac_pilot_v7 (840,000 steps, 8 h) learned to fall face-down and lie
# there: base at ~0.11 m, tilt ~106 deg, neither limit (0.10 m / ~145 deg)
# triggered, so it avoided the -100 and collected alive - tilt ~ +0.5/step.
# v8 terminates at height < 0.20 m OR tilt > 60 deg from vertical; reward
# identical to v7. FRESH start: v7's buffer holds lying-down transitions
# that were not terminal.
RUN_NAME="sac_pilot_v8"
MAX_WALLCLOCK_SECONDS=43200  # 12 h batch per launch (v8: first batch 0 -> 1,275,000; resumed 2026-11-25 for 12 h). v7: 8 h. v6: 8 h, stopped at 225k when the machine shut down, resumed 2026-11-23 for 12 h. [v2 run] Previous 12h block (3,255,000
                             # -> 4,290,000 steps) raised training-time
                             # ep_len_mean 76->84 and ep_rew_mean 287->315,
                             # but the DETERMINISTIC checkpoint eval was
                             # identical at both ends of that block: falls
                             # at step 51, same fall height (0.062m), only
                             # forward speed improved (0.238->0.319 m/s).
                             # (Note: the "0.720 m/s / 82 steps" figure
                             # previously written here for the 3,255,000-
                             # step checkpoint could not be corroborated in
                             # APPROACH.md and is believed to have been in
                             # error — superseded by the re-measured
                             # 0.238 m/s / 51 steps above.) Running another
                             # 8h to see whether the deterministic failure
                             # point moves at all.
TOTAL_STEPS=6000000          # backstop only; MAX_WALLCLOCK_SECONDS is the
                             # real limit per batch (raised from 5M in v2)
CHUNK_STEPS=15000            # unchanged (see note above)
URDF="../../walking-solutions/shared/robot/nao.urdf"
RUN_DIR="runs/${RUN_NAME}"
PROGRESS_FILE="${RUN_DIR}/progress.txt"
LOG_FILE="${RUN_DIR}/training_log.txt"

LR=3e-4
ENT_COEF="auto"
REST_SECONDS=2
MIN_RAM_MB=4000
CHUNK_TIMEOUT_SECONDS=1800   # per-chunk hard limit (a normal 15k-step chunk
                             # takes ~10 min). Catches Isaac Sim HANGS, which
                             # never exit on their own — e.g. the v1 run's
                             # rtx.scenedb semaphore deadlock that stalled
                             # ~5 h. The model/replay buffer are only saved at
                             # the end of a chunk, so a killed chunk is simply
                             # retried from the last good final_model.zip.

mkdir -p "$RUN_DIR"

completed_steps=0
[ -f "$PROGRESS_FILE" ] && completed_steps=$(cat "$PROGRESS_FILE")

start_wallclock=$(date +%s)
echo "== SAC pilot from ${completed_steps} steps, max ${MAX_WALLCLOCK_SECONDS}s wallclock ==" | tee -a "$LOG_FILE"
chunk_num=$((completed_steps / CHUNK_STEPS))

while [ "$completed_steps" -lt "$TOTAL_STEPS" ]; do
    now=$(date +%s)
    elapsed_wallclock=$((now - start_wallclock))
    if [ "$elapsed_wallclock" -ge "$MAX_WALLCLOCK_SECONDS" ]; then
        echo "  Reached ${MAX_WALLCLOCK_SECONDS}s wallclock limit, stopping." | tee -a "$LOG_FILE"
        break
    fi

    chunk_num=$((chunk_num + 1))
    remaining=$((TOTAL_STEPS - completed_steps))
    this_chunk=$CHUNK_STEPS
    [ "$this_chunk" -gt "$remaining" ] && this_chunk=$remaining

    echo "  Chunk ${chunk_num}: ${completed_steps} -> $((completed_steps+this_chunk)) (${elapsed_wallclock}s elapsed)" | tee -a "$LOG_FILE"

    # RAM safety check (guards Isaac Sim's footprint, not the replay buffer —
    # see header comment)
    avail=$(free -m | grep Mem | awk '{print $7}')
    while [ "$avail" -lt "$MIN_RAM_MB" ]; do
        echo "  Low RAM (${avail}MB), waiting..." | tee -a "$LOG_FILE"
        sleep 30
        avail=$(free -m | grep Mem | awk '{print $7}')
    done

    OWN_MODEL="${RUN_DIR}/final_model.zip"
    RESUME_FLAG=""
    if [ -f "$OWN_MODEL" ]; then
        RESUME_FLAG="--resume $OWN_MODEL"
        if [ ! -f "${RUN_DIR}/replay_buffer.pkl" ]; then
            echo "  WARNING: resuming but no replay_buffer.pkl present yet." | tee -a "$LOG_FILE"
        fi
    fi

    # timeout: SIGTERM after CHUNK_TIMEOUT_SECONDS, SIGKILL 60 s later if the
    # process ignores it. Exit code 124 (or 137 if SIGKILLed) = timed out.
    timeout --kill-after=60 "$CHUNK_TIMEOUT_SECONDS" \
        python train_sac.py --headless --timesteps "$this_chunk" --run-name "$RUN_NAME" \
        --urdf "$URDF" --checkpoint-freq "$CHUNK_STEPS" --lr "$LR" --ent-coef "$ENT_COEF" \
        $RESUME_FLAG --log-file "$LOG_FILE"
    exit_code=$?

    if [ "$exit_code" -ne 0 ]; then
        if [ "$exit_code" -eq 124 ] || [ "$exit_code" -eq 137 ]; then
            echo "  Chunk ${chunk_num} TIMED OUT after ${CHUNK_TIMEOUT_SECONDS}s (likely Isaac Sim hang)." | tee -a "$LOG_FILE"
        fi
        echo "  Chunk ${chunk_num} failed (exit ${exit_code}), retrying in 10s..." | tee -a "$LOG_FILE"
        sleep 10
        continue
    fi

    completed_steps=$((completed_steps + this_chunk))
    echo "$completed_steps" > "$PROGRESS_FILE"

    [ "$completed_steps" -lt "$TOTAL_STEPS" ] && sleep "$REST_SECONDS"
done

echo "SAC pilot stopped at ${completed_steps} steps ($(( $(date +%s) - start_wallclock ))s wallclock)" | tee -a "$LOG_FILE"
echo "Evaluate with: python evaluate_policy.py --algo sac --model ${RUN_DIR}/final_model.zip --episodes 10"
