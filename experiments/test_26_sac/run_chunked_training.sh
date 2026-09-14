#!/bin/bash
# test_26_sac pilot: SAC, SB3 2.7.1 defaults, net_arch/gamma matched to
# test_24 for comparability. Time-bounded, not a 50M-step commitment —
# see APPROACH.md.
#
# History: 2h initial pilot (145,000 steps, 29 chunks) -> 12h overnight
# extension (to 1,165,000 steps, 68 more chunks, committed to the repo)
# -> 24h extension (to 3,325,000 steps, 144 more chunks, 241 total,
# still clean) -> this 12h extension. CHUNK_STEPS stays at 15,000 —
# resume mechanism verified across 241 boundaries with zero issues.
#
# Replay-buffer RAM footprint (computed from actual obs/action dims):
#   obs=23, action=12, buffer_size=1e6, float32, handle_timeout_termination=True:
#   observations 92MB + next_observations 92MB + actions 48MB + rewards/dones/timeouts 12MB
#   = ~244MB (worst-case ~2x transient during pickle load/reconstruct ~490MB).
#   Comfortably inside the MIN_RAM_MB gate below (Isaac Sim itself, not the
#   buffer, is what that gate is actually protecting against).

RUN_NAME="sac_pilot"
MAX_WALLCLOCK_SECONDS=43200  # 12 hour extension
TOTAL_STEPS=5000000          # unchanged — at ~24.9 steps/sec observed,
                             # 12h adds ~1.08M steps, landing ~4.4M,
                             # still comfortably under this backstop
CHUNK_STEPS=15000            # unchanged (see note above)
URDF="../../walking-solutions/shared/robot/nao.urdf"
RUN_DIR="runs/${RUN_NAME}"
PROGRESS_FILE="${RUN_DIR}/progress.txt"
LOG_FILE="${RUN_DIR}/training_log.txt"

LR=3e-4
ENT_COEF="auto"
REST_SECONDS=2
MIN_RAM_MB=4000

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

    python train_sac.py --headless --timesteps "$this_chunk" --run-name "$RUN_NAME" \
        --urdf "$URDF" --checkpoint-freq "$CHUNK_STEPS" --lr "$LR" --ent-coef "$ENT_COEF" \
        $RESUME_FLAG --log-file "$LOG_FILE"

    if [ $? -ne 0 ]; then
        echo "  Chunk ${chunk_num} failed, retrying in 10s..." | tee -a "$LOG_FILE"
        sleep 10
        continue
    fi

    completed_steps=$((completed_steps + this_chunk))
    echo "$completed_steps" > "$PROGRESS_FILE"

    [ "$completed_steps" -lt "$TOTAL_STEPS" ] && sleep "$REST_SECONDS"
done

echo "SAC pilot stopped at ${completed_steps} steps ($(( $(date +%s) - start_wallclock ))s wallclock)" | tee -a "$LOG_FILE"
echo "Evaluate with: python evaluate_policy.py --algo sac --model ${RUN_DIR}/final_model.zip --episodes 10"
