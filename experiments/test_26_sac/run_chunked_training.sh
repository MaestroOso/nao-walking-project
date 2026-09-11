#!/bin/bash
# test_26_sac pilot: SAC, SB3 2.7.1 defaults, net_arch/gamma matched to
# test_24 for comparability. Time-bounded, not a 50M-step commitment —
# see APPROACH.md.
#
# Extended overnight (12h) after the initial 2h pilot (145,000 steps,
# clean across 29 chunk boundaries) showed a real, steady learning
# trend with no mechanism issues. CHUNK_STEPS raised from 5,000 to
# 15,000 for this longer run — the frequent-boundary pilot already
# verified the resume mechanism thoroughly, so the priority now shifts
# to cutting restart overhead (~15-20s/chunk fixed cost: was ~7-10% of
# wall-clock at 5,000 steps/chunk, drops to ~2% at 15,000) while still
# restarting often enough for Isaac Sim's long-run memory hygiene
# (the reason this project has used chunked training since test_07).
#
# Replay-buffer RAM footprint (computed from actual obs/action dims):
#   obs=23, action=12, buffer_size=1e6, float32, handle_timeout_termination=True:
#   observations 92MB + next_observations 92MB + actions 48MB + rewards/dones/timeouts 12MB
#   = ~244MB (worst-case ~2x transient during pickle load/reconstruct ~490MB).
#   Comfortably inside the MIN_RAM_MB gate below (Isaac Sim itself, not the
#   buffer, is what that gate is actually protecting against).

RUN_NAME="sac_pilot"
MAX_WALLCLOCK_SECONDS=43200  # 12 hour overnight extension
TOTAL_STEPS=3000000          # generous backstop ceiling, not expected to bind
CHUNK_STEPS=15000            # raised from 5,000 (see note above) — resume
                             # mechanism already verified across 29 boundaries
                             # in the initial pilot, so bigger chunks now to
                             # cut restart overhead for the longer run
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
