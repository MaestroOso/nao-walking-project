#!/bin/bash
# SAC baseline_test26: chunked, resumable training (report §4.6.1).
# Promoted from experiments/test_26_sac (run sac_pilot_v5). The full history
# of why each setting is what it is (v1 -> v5) is in that folder's
# APPROACH.md; README.md here summarises it.
#
# Each chunk is a fresh Python/Isaac Sim process that trains CHUNK_STEPS
# steps, then saves final_model.zip + replay_buffer.pkl. Re-running the
# script with the same run name continues the run (same model, same replay
# buffer, step counter keeps counting).
#
# Usage (Isaac Sim venv active, see README.md):
#   ./run_chunked_training.sh [run_name]
#   ./run_chunked_training.sh sac_v5_repro     # new run with the v5 settings
#   ./run_chunked_training.sh sac_pilot_v5     # continue the promoted run

cd "$(dirname "$0")"
RUN_NAME="${1:-sac_pilot_v5}"
MAX_WALLCLOCK_SECONDS=43200  # 12 h per batch (sac_pilot_v5: 12 h, 0 -> 1.23M steps)
TOTAL_STEPS=6000000          # backstop only; MAX_WALLCLOCK_SECONDS is the real limit
CHUNK_STEPS=15000            # ~10 min per chunk on the RTX 4060 Laptop
URDF="../../shared/robot/nao.urdf"
RUN_DIR="runs/${RUN_NAME}"
PROGRESS_FILE="${RUN_DIR}/progress.txt"
LOG_FILE="${RUN_DIR}/training_log.txt"

LR=3e-4
ENT_COEF="auto"
REST_SECONDS=2
MIN_RAM_MB=4000
CHUNK_TIMEOUT_SECONDS=1800   # per-chunk hard limit (a normal 15k-step chunk
                             # takes ~10 min). Catches Isaac Sim HANGS, which
                             # never exit on their own — e.g. the test_26 v1 run's
                             # rtx.scenedb semaphore deadlock that stalled
                             # ~5 h. The model/replay buffer are only saved at
                             # the end of a chunk, so a killed chunk is simply
                             # retried from the last good final_model.zip.

mkdir -p "$RUN_DIR"

completed_steps=0
[ -f "$PROGRESS_FILE" ] && completed_steps=$(cat "$PROGRESS_FILE")

start_wallclock=$(date +%s)
echo "== SAC ${RUN_NAME} from ${completed_steps} steps, max ${MAX_WALLCLOCK_SECONDS}s wallclock ==" | tee -a "$LOG_FILE"
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
    # the ~244 MB replay buffer is not the concern)
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

echo "SAC ${RUN_NAME} stopped at ${completed_steps} steps ($(( $(date +%s) - start_wallclock ))s wallclock)" | tee -a "$LOG_FILE"
echo "Evaluate with: python evaluate_policy.py --algo sac --model ${RUN_DIR}/final_model.zip --episodes 10"
