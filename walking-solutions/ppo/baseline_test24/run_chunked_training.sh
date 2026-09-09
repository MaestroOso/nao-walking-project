#!/bin/bash
# baseline_test24: Webots gait + residuals, action_scale=1.0, no pretrain
# Optimized: bigger chunks, no pretrain stage, minimal rest, GPU training

RUN_NAME="baseline_test24"
TOTAL_STEPS=50000000
CHUNK_STEPS=50000
URDF="../../shared/robot/nao.urdf"
RUN_DIR="runs/${RUN_NAME}"
PROGRESS_FILE="${RUN_DIR}/progress.txt"
LOG_FILE="${RUN_DIR}/training_log.txt"

LR=3e-4
ENT_COEF=0.01
REST_SECONDS=2
MIN_RAM_MB=4000

mkdir -p "$RUN_DIR"

# No pretrain — start PPO from scratch with random init.
# The Webots reference gait provides structure; residuals start at zero
# and the policy learns corrections from scratch.

completed_steps=0
[ -f "$PROGRESS_FILE" ] && completed_steps=$(cat "$PROGRESS_FILE")

echo "== PPO training (no pretrain) from ${completed_steps} ==" | tee -a "$LOG_FILE"
echo "  action_scale=1.0, chunks=${CHUNK_STEPS}, target=${TOTAL_STEPS}" | tee -a "$LOG_FILE"
chunk_num=$((completed_steps / CHUNK_STEPS))

while [ "$completed_steps" -lt "$TOTAL_STEPS" ]; do
    chunk_num=$((chunk_num + 1))
    remaining=$((TOTAL_STEPS - completed_steps))
    this_chunk=$CHUNK_STEPS
    [ "$this_chunk" -gt "$remaining" ] && this_chunk=$remaining

    echo "  Chunk ${chunk_num}: ${completed_steps} → $((completed_steps+this_chunk)) / ${TOTAL_STEPS}" | tee -a "$LOG_FILE"

    # RAM safety check
    avail=$(free -m | grep Mem | awk '{print $7}')
    while [ "$avail" -lt "$MIN_RAM_MB" ]; do
        echo "  Low RAM (${avail}MB), waiting..." | tee -a "$LOG_FILE"
        sleep 30
        avail=$(free -m | grep Mem | awk '{print $7}')
    done

    # Resume from own model if exists, otherwise fresh start (no --resume)
    OWN_MODEL="${RUN_DIR}/final_model.zip"
    RESUME_FLAG=""
    if [ -f "$OWN_MODEL" ]; then
        RESUME_FLAG="--resume $OWN_MODEL"
    fi

    python train_ppo.py --headless --timesteps "$this_chunk" --run-name "$RUN_NAME" \
        --urdf "$URDF" --checkpoint-freq "$CHUNK_STEPS" --lr "$LR" --ent-coef "$ENT_COEF" \
        $RESUME_FLAG --log-file "$LOG_FILE"

    if [ $? -ne 0 ]; then
        echo "  Chunk ${chunk_num} failed, retrying in 10s..." | tee -a "$LOG_FILE"
        sleep 10
        continue
    fi

    completed_steps=$((completed_steps + this_chunk))
    echo "$completed_steps" > "$PROGRESS_FILE"

    # Save milestone every 500k steps
    [ $((completed_steps % 500000)) -eq 0 ] && [ -f "$OWN_MODEL" ] && \
        cp "$OWN_MODEL" "${RUN_DIR}/model_${completed_steps}.zip" && \
        echo "  Milestone saved: model_${completed_steps}.zip" | tee -a "$LOG_FILE"

    [ "$completed_steps" -lt "$TOTAL_STEPS" ] && sleep "$REST_SECONDS"
done

echo "Training complete! ${TOTAL_STEPS} steps" | tee -a "$LOG_FILE"
echo "Evaluate with: python evaluate_policy.py --model ${RUN_DIR}/final_model.zip --episodes 10"
