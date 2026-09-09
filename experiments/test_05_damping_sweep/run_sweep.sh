#!/bin/bash
# run_sweep.sh — Run gain calibration sweep (one process per trial)
#
# Each trial runs in a fresh Python process, so PhysX NaN corruption
# from one trial cannot contaminate the next.
#
# USAGE:
#   chmod +x run_sweep.sh
#   ./run_sweep.sh
#
# Results go to individual trial_*.txt files + combined summary.

URDF="../../robots/nao_isaac_sim/nao.urdf"
SECONDS=10
HIP_LEAN=-2
SUMMARY="sweep_summary.txt"

echo "================================================================" | tee "$SUMMARY"
echo "GAIN CALIBRATION SWEEP — $(date)" | tee -a "$SUMMARY"
echo "  Hip lean = ${HIP_LEAN}°, duration = ${SECONDS}s each" | tee -a "$SUMMARY"
echo "================================================================" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# Each line: Kp Kd MaxEffort Label
CONFIGS=(
    # Realistic gains, realistic torque
    "10    1.0    3    Kp=10_Kd=1_eff=3"
    "25    2.5    3    Kp=25_Kd=2.5_eff=3"
    "50    5.0    3    Kp=50_Kd=5_eff=3"
    "100   10.0   3    Kp=100_Kd=10_eff=3"
    # Realistic gains, relaxed torque
    "10    1.0   10    Kp=10_Kd=1_eff=10"
    "25    2.5   10    Kp=25_Kd=2.5_eff=10"
    "50    5.0   10    Kp=50_Kd=5_eff=10"
    "100   10.0  10    Kp=100_Kd=10_eff=10"
    # Moderate gains, high torque
    "500   50.0  100   Kp=500_Kd=50_eff=100"
    "1000  100.0 100   Kp=1000_Kd=100_eff=100"
    # Original for comparison
    "35810 500.0 100   Kp=35810_Kd=500_eff=100_ORIGINAL"
)

TOTAL=${#CONFIGS[@]}
COUNT=0

echo "Kp       Kd      Eff    Verdict     Survive  Height   HipErr  KneeErr" | tee -a "$SUMMARY"
echo "----------------------------------------------------------------------" | tee -a "$SUMMARY"

for cfg in "${CONFIGS[@]}"; do
    read -r KP KD EFF LABEL <<< "$cfg"
    COUNT=$((COUNT + 1))

    echo ""
    echo "[$COUNT/$TOTAL] Running: $LABEL ..."

    python single_trial.py \
        --urdf "$URDF" \
        --headless \
        --kp "$KP" \
        --kd "$KD" \
        --eff "$EFF" \
        --hip-lean "$HIP_LEAN" \
        --seconds "$SECONDS" \
        --output "trial_${LABEL}.txt" \
        2>/dev/null | grep "^Kp=" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "================================================================" | tee -a "$SUMMARY"
echo "SWEEP COMPLETE — $(date)" | tee -a "$SUMMARY"
echo "================================================================" | tee -a "$SUMMARY"

echo ""
echo "Summary saved to: $SUMMARY"
echo "Individual trial logs: trial_*.txt"
