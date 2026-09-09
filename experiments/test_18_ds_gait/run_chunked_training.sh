#!/bin/bash
# experiment 17: V4 iter 7 reference gait + PPO residuals
# Two stages: pretrain then PPO chunks
#
# Changes vs experiment 16:
# - RUN_NAME="exp17_rl"
# - CHUNK_STEPS=20480 (10x larger — user authorized with 32GB RAM)
#   This reduces Isaac Sim restart overhead by ~10x. Each chunk now does
#   10 PPO updates (n_steps=2048, so 20480/2048 = 10 updates per chunk)
#   instead of 1 update per chunk in experiment 16.
# - Updated paths and pretrained model location

RUN_NAME="test19_no_dive"
TOTAL_STEPS=3000000
CHUNK_STEPS=40960           # exp17: 10x larger than exp16's 2048
URDF="../../robots/nao_isaac_sim/nao.urdf"
RUN_DIR="runs/${RUN_NAME}"
PROGRESS_FILE="${RUN_DIR}/progress.txt"
LOG_FILE="${RUN_DIR}/training_log.txt"
PRETRAINED="${RUN_DIR}/pretrained_model.zip"

LR=3e-4
ENT_COEF=0.01
REST_SECONDS=15
MIN_RAM_MB=4000

# Backup snapshot every 10 chunks (10 * 20480 = 204800 steps)
BACKUP_EVERY_STEPS=204800

mkdir -p "$RUN_DIR"

echo "============================================================"
echo " Experiment 17: V4 iter 7 ref gait + PPO residuals"
echo " Total: ${TOTAL_STEPS} steps in ${CHUNK_STEPS}-step chunks"
echo " Output: ${RUN_DIR}"
echo "============================================================"

# ── Stage 1: Pre-training ──
if [ ! -f "$PRETRAINED" ]; then
    echo "== Stage 1: Pre-training ==" | tee -a "$LOG_FILE"
    python pretrain.py --collect --headless --episodes 20 --urdf "$URDF" \
        --data-file "${RUN_DIR}/pretrain_data.npz" \
        --output "$PRETRAINED" 2>&1 | tee -a "$LOG_FILE"

    python pretrain.py --train --epochs 50 \
        --data-file "${RUN_DIR}/pretrain_data.npz" \
        --output "$PRETRAINED" 2>&1 | tee -a "$LOG_FILE"

    if [ ! -f "$PRETRAINED" ]; then
        echo "Pre-training failed"; exit 1
    fi
    echo "✓ Pre-training complete" | tee -a "$LOG_FILE"
else
    echo "Pre-trained model exists, skipping" | tee -a "$LOG_FILE"
fi

# ── Stage 2: PPO fine-tuning ──
completed_steps=0
[ -f "$PROGRESS_FILE" ] && completed_steps=$(cat "$PROGRESS_FILE")

echo "== Stage 2: PPO fine-tuning from ${completed_steps} ==" | tee -a "$LOG_FILE"
chunk_num=$((completed_steps / CHUNK_STEPS))

while [ "$completed_steps" -lt "$TOTAL_STEPS" ]; do
    chunk_num=$((chunk_num + 1))
    remaining=$((TOTAL_STEPS - completed_steps))
    this_chunk=$CHUNK_STEPS
    [ "$this_chunk" -gt "$remaining" ] && this_chunk=$remaining

    echo "  Chunk ${chunk_num}: ${completed_steps} → $((completed_steps+this_chunk)) / ${TOTAL_STEPS}" | tee -a "$LOG_FILE"

    avail=$(free -m | grep Mem | awk '{print $7}')
    while [ "$avail" -lt "$MIN_RAM_MB" ]; do
        echo "  Low RAM (${avail}MB), waiting..."; sleep 30
        avail=$(free -m | grep Mem | awk '{print $7}')
    done

    OWN_MODEL="${RUN_DIR}/final_model.zip"
    RESUME_MODEL="$PRETRAINED"
    [ -f "$OWN_MODEL" ] && RESUME_MODEL="$OWN_MODEL"

    python train_ppo.py --headless --timesteps "$this_chunk" --run-name "$RUN_NAME" \
        --urdf "$URDF" --checkpoint-freq "$CHUNK_STEPS" --lr "$LR" --ent-coef "$ENT_COEF" \
        --resume "$RESUME_MODEL" --log-file "$LOG_FILE"

    if [ $? -ne 0 ]; then
        echo "  Chunk failed, retrying..." | tee -a "$LOG_FILE"; sleep 5; continue
    fi

    completed_steps=$((completed_steps + this_chunk))
    echo "$completed_steps" > "$PROGRESS_FILE"

    # Periodic backup snapshots so you can roll back if training degrades
    if [ $((completed_steps % BACKUP_EVERY_STEPS)) -eq 0 ] && [ -f "$OWN_MODEL" ]; then
        cp "$OWN_MODEL" "${RUN_DIR}/model_${completed_steps}.zip"
        echo "  ✓ Backup saved: model_${completed_steps}.zip" | tee -a "$LOG_FILE"
    fi

    [ "$completed_steps" -lt "$TOTAL_STEPS" ] && sleep "$REST_SECONDS"
done

echo "Training complete! ${TOTAL_STEPS} steps" | tee -a "$LOG_FILE"
