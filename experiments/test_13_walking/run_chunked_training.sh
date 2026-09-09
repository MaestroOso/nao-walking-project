#!/bin/bash
# ============================================================================
# run_chunked_training.sh — test_13: Parallel multi-robot training
# ============================================================================
#
# Training from SCRATCH (new VecEnv architecture)
#
# Features:
#   - Multiple robots training in parallel (N× faster)
#   - Larger chunks possible (finger fixes stabilized physics)
#   - RAM check before each chunk — waits if memory too low
#   - All output logged to training_log.txt
#
# USAGE:
#   chmod +x run_chunked_training.sh
#   ./run_chunked_training.sh
#
# MONITORING:
#   tail -f runs/test13_walking/training_log.txt
#   tensorboard --logdir runs/
# ============================================================================

# ── Configuration ────────────────────────────────────────────────────────────
RUN_NAME="test13_walking"
TOTAL_STEPS=5000000
CHUNK_STEPS=10240
URDF="../../robots/nao_isaac_sim/nao.urdf"
RUN_DIR="runs/${RUN_NAME}"
PROGRESS_FILE="${RUN_DIR}/progress.txt"
LOG_FILE="${RUN_DIR}/training_log.txt"

# Number of parallel robots
NUM_ENVS=2

# PPO hyperparameters
LR=3e-4
ENT_COEF=0.01

# Rest period between chunks (seconds)
REST_SECONDS=5

# Minimum available RAM (MB) before starting a chunk
MIN_RAM_MB=4000

mkdir -p "$RUN_DIR"

# ── Read progress ───────────────────────────────────────────────────────────
completed_steps=0
if [ -f "$PROGRESS_FILE" ]; then
    completed_steps=$(cat "$PROGRESS_FILE")
    echo "  Found progress file: ${completed_steps} steps completed"
fi

# ── Header ───────────────────────────────────────────────────────────────────
header="
============================================================
  Chunked Training: ${RUN_NAME}
  test_13: Parallel multi-robot training (${NUM_ENVS} robots)
  Total: ${TOTAL_STEPS} steps in chunks of ${CHUNK_STEPS}
  Starting from: ${completed_steps} steps
  LR: ${LR}, Entropy: ${ENT_COEF}
  Rest between chunks: ${REST_SECONDS}s
  Log file: ${LOG_FILE}
============================================================"

echo "$header"
echo "$header" >> "$LOG_FILE"

chunk_num=$((completed_steps / CHUNK_STEPS))

while [ "$completed_steps" -lt "$TOTAL_STEPS" ]; do
    chunk_num=$((chunk_num + 1))
    remaining=$((TOTAL_STEPS - completed_steps))

    if [ "$CHUNK_STEPS" -gt "$remaining" ]; then
        this_chunk=$remaining
    else
        this_chunk=$CHUNK_STEPS
    fi

    chunk_header="
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CHUNK ${chunk_num}: steps ${completed_steps} → $((completed_steps + this_chunk)) / ${TOTAL_STEPS}
  $(date '+%Y-%m-%d %H:%M:%S')
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "$chunk_header"
    echo "$chunk_header" >> "$LOG_FILE"

    # ── RAM check — wait if memory is too low ────────────────────────────
    avail=$(free -m | grep Mem | awk '{print $7}')
    while [ "$avail" -lt "$MIN_RAM_MB" ]; do
        echo "  ⏳ Low RAM (${avail}MB < ${MIN_RAM_MB}MB) — waiting 30s..."
        echo "  ⏳ Low RAM (${avail}MB) — waiting 30s..." >> "$LOG_FILE"
        sleep 30
        avail=$(free -m | grep Mem | awk '{print $7}')
    done
    echo "  RAM OK: ${avail}MB available"

    cmd="python train_ppo.py --headless --timesteps ${this_chunk} --run-name ${RUN_NAME} --urdf ${URDF} --checkpoint-freq ${CHUNK_STEPS} --lr ${LR} --ent-coef ${ENT_COEF} --num-envs ${NUM_ENVS} --log-file ${LOG_FILE}"

    # Resume from own model if it exists
    OWN_MODEL="${RUN_DIR}/final_model.zip"
    if [ -f "$OWN_MODEL" ]; then
        cmd="${cmd} --resume ${OWN_MODEL}"
        echo "  Resuming from: ${OWN_MODEL}"
    else
        echo "  Starting fresh (training from scratch — test_13)"
    fi

    echo "  Command: ${cmd}"
    echo ""

    # Run training chunk
    start_time=$(date +%s)
    eval $cmd
    exit_code=$?
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    if [ $exit_code -ne 0 ]; then
        error_msg="  ⚠ Chunk ${chunk_num} exited with code ${exit_code} — retrying..."
        echo "$error_msg"
        echo "$error_msg" >> "$LOG_FILE"
        sleep 5
        continue
    fi

    # ── Chunk succeeded ─────────────────────────────────────────────────────
    completed_steps=$((completed_steps + this_chunk))
    echo "$completed_steps" > "$PROGRESS_FILE"

    if [ -f "$OWN_MODEL" ]; then
        # Save numbered backup every ~50k steps
        if [ $((completed_steps % 51200)) -eq 0 ]; then
            cp "$OWN_MODEL" "${RUN_DIR}/model_${completed_steps}.zip"
            echo "  Backup saved: model_${completed_steps}.zip"
        fi
    fi

    summary="  Chunk ${chunk_num} complete in ${elapsed}s ($((elapsed / 60))m)
  Progress: ${completed_steps} / ${TOTAL_STEPS} steps"
    echo "$summary"
    echo "$summary" >> "$LOG_FILE"

    # ── Rest period ─────────────────────────────────────────────────────────
    if [ "$completed_steps" -lt "$TOTAL_STEPS" ]; then
        avail=$(free -m | grep Mem | awk '{print $7}')
        rest_msg="  Available RAM: ${avail} MB — resting ${REST_SECONDS}s..."
        echo "$rest_msg"
        echo "$rest_msg" >> "$LOG_FILE"
        sleep "$REST_SECONDS"
    fi
done

final_msg="
============================================================
  Training complete! ${TOTAL_STEPS} steps across ${chunk_num} chunks
  Final model: ${RUN_DIR}/final_model.zip
  Full log: ${LOG_FILE}
  TensorBoard: tensorboard --logdir runs/
============================================================"
echo "$final_msg"
echo "$final_msg" >> "$LOG_FILE"
