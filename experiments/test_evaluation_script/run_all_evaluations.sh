#!/bin/bash
# run_all_evaluations.sh — evaluate every algorithm under the same protocol,
# then build the summary table and robustness figure.
#
# One Python process per algorithm (Isaac Sim's SimulationApp can only be
# created once per process, and each algorithm has its own nao_walking_env.py).
# A failing algorithm does not stop the others.
#
# Run inside the Isaac Sim Python environment, e.g.:
#   source ../../nao_walking_project/isaac_env/bin/activate
#   ./run_all_evaluations.sh                       # defaults below
#   DISTANCE=1.5 ./run_all_evaluations.sh --overwrite
#   ALGOS="sac" REPS=3 ./run_all_evaluations.sh    # quick smoke test
# Extra arguments are passed to every evaluate_algorithms.py call.

set -u
cd "$(dirname "$0")"

ALGOS="${ALGOS:-ppo sac pgrl}"     # pgrl also evaluates its base gait (PGRL_base)
DISTANCE="${DISTANCE:-1.0}"        # course length D [m]
DELTAS="${DELTAS:-0 1 2 4}"        # perturbation amplitudes [deg]
REPS="${REPS:-10}"                 # repetitions per delta
BASE_SEED="${BASE_SEED:-0}"

failed=()
for algo in $ALGOS; do
    echo "================ ${algo^^} ================"
    # shellcheck disable=SC2086
    if ! python evaluate_algorithms.py --algo "$algo" --distance "$DISTANCE" \
            --deltas $DELTAS --repetitions "$REPS" --base-seed "$BASE_SEED" "$@"; then
        echo "!! ${algo^^} evaluation FAILED"
        failed+=("$algo")
    fi
    if [ "$algo" = "pgrl" ]; then
        echo "================ PGRL_base (initial parameters) ================"
        # shellcheck disable=SC2086
        if ! python evaluate_algorithms.py --algo pgrl --pgrl-initial --label PGRL_base \
                --distance "$DISTANCE" --deltas $DELTAS --repetitions "$REPS" \
                --base-seed "$BASE_SEED" "$@"; then
            echo "!! PGRL_base evaluation FAILED"
            failed+=("pgrl_base")
        fi
    fi
done

python summarize_results.py

if [ ${#failed[@]} -gt 0 ]; then
    echo "Failed: ${failed[*]}"
    exit 1
fi
