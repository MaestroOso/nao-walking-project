# test_05_damping_sweep

**Focus:** Turn `test_04`'s diagnosis into a systematic fix — a grid
search over PD gains (`Kp`, `Kd`) and joint effort limits, with the
initialization-order bug from `test_04` corrected so damping is actually
applied before the robot is released to gravity.

**What we did:** `damping_sweep.py` / `run_sweep.sh` ran trials across a
range of gain combinations (see the `trial_Kp=*_Kd=*_eff=*` result
filenames in the original run for the exact grid — not migrated here, see
note below); `gain_calibration.py` and `single_trial.py` supported manual
single-configuration checks.

**Outcome — the turning point of Phase 1:** A working configuration was
found. Both **low gains** (Kp=50, Kd=5) and the **URDF's original high
gains** (Kp=35810, Kd=500) worked once damping was correctly applied.

**Key learning:** The fix was *correct damping being applied at all*, not
any particular gain magnitude. This reframed the whole standing problem —
it had never been a tuning problem, it was an application-order/API bug
(see `test_04`).

**Note on raw sweep data:** The individual trial result `.txt`/`.csv`
files (one per gain combination) are not included in this archival
snapshot — they're raw numeric dumps with no code value, and their
conclusion is fully captured above and in `docs/experiment-history.md`.
