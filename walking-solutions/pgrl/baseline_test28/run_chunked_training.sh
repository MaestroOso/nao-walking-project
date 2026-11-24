#!/bin/bash
# PGRL baseline_test28: chunked, resumable training (report §4.6.1).
# Each chunk is a fresh Python/Isaac Sim process running ITER_PER_CHUNK
# iterations (12 episodes each); state is saved after every iteration, so a
# crash loses at most one iteration. Re-running this script continues the run.
#
# Usage (Isaac Sim venv active, see README.md):
#   ./run_chunked_training.sh [run_name] [extra train_pgrl.py args...]
#   ./run_chunked_training.sh pgrl_v2                     # new run, gains [-3, 3]
#   ./run_chunked_training.sh pgrl_v1_repro --specs v1    # reproduce pgrl_v1

cd "$(dirname "$0")"
RUN_NAME="${1:-pgrl_v1}"
shift || true
EXTRA_ARGS=("$@")
MAX_EVALS=2000
POLICIES=12
ITER_PER_CHUNK=10
MIN_RAM_MB=4000
MAX_FAILS=3
RUN_DIR="runs/${RUN_NAME}"
LOG_FILE="${RUN_DIR}/training_log.txt"
mkdir -p "$RUN_DIR"

evals_done() {
    [ -f "${RUN_DIR}/state.json" ] || { echo 0; return; }
    python -c "import json;print(json.load(open('${RUN_DIR}/state.json'))['evals'])"
}

fails=0
echo "== PGRL ${RUN_NAME}: start at $(evals_done)/${MAX_EVALS} evals ($(date)) ==" | tee -a "$LOG_FILE"
while [ "$(evals_done)" -le $((MAX_EVALS - POLICIES)) ]; do
    while [ "$(free -m | awk '/Mem/{print $7}')" -lt "$MIN_RAM_MB" ]; do
        echo "  low RAM, waiting 30 s..." | tee -a "$LOG_FILE"; sleep 30
    done
    before=$(evals_done)
    python -u train_pgrl.py --run-name "$RUN_NAME" --iterations "$ITER_PER_CHUNK" \
        --max-evals "$MAX_EVALS" --policies "$POLICIES" "${EXTRA_ARGS[@]}" 2>&1 \
        | grep --line-buffered -E "^\[pgrl\]|^  it |θ =|Traceback|Error:" | tee -a "$LOG_FILE"
    after=$(evals_done)
    if [ "$after" -le "$before" ]; then
        fails=$((fails + 1))
        echo "  chunk made no progress (${fails}/${MAX_FAILS})" | tee -a "$LOG_FILE"
        [ "$fails" -ge "$MAX_FAILS" ] && { echo "  giving up" | tee -a "$LOG_FILE"; exit 1; }
    else
        fails=0
    fi
    sleep 2
done
echo "== PGRL ${RUN_NAME}: done, $(evals_done) evals ($(date)) ==" | tee -a "$LOG_FILE"
