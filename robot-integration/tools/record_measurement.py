#!/usr/bin/env python3
"""
record_measurement.py — add the operator's tape-measure readings to a run.

The robot cannot measure how far it walked (odometry is an estimate), so
after each repetition the operator measures it and records it here:

    python3 robot-integration/tools/record_measurement.py ~/nao_robot_runs/<run> \
        --distance 0.62 --lateral -0.05 [--outcome fall] [--note "fell to its right"]

--distance  forward distance from the start line [m] (for a success, the
            course distance D unless measured otherwise)
--lateral   final lateral offset, positive to the robot's LEFT [m]
--outcome   correct the outcome logged by the node (success|fall|timeout|aborted)
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

OUTCOMES = ("success", "fall", "timeout", "aborted")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--distance", type=float, required=True)
    ap.add_argument("--lateral", type=float, default=None)
    ap.add_argument("--outcome", choices=OUTCOMES)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    path = a.run_dir.expanduser() / "run.json"
    if not path.is_file():
        print(f"no run.json in {a.run_dir}")
        return 1
    meta = json.loads(path.read_text())
    meta["measured"] = {
        "distance_m": a.distance, "lateral_left_m": a.lateral,
        "outcome_override": a.outcome, "note": a.note,
        "recorded": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(meta, indent=2))
    print(f"{path}: logged outcome {meta.get('outcome')}"
          f"{' -> ' + a.outcome if a.outcome else ''}, distance {a.distance} m, "
          f"lateral {a.lateral} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
