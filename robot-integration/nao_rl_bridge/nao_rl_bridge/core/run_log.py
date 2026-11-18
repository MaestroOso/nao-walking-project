"""
run_log.py — one folder per robot run.

    <run_root>/<YYYYmmdd_HHMMSS>_<label>_d<delta>_r<rep>/
      run.json    metadata, configuration, outcome, protocol fields
                  (operator measurements are added later by
                  tools/record_measurement.py)
      ticks.csv   one row per control tick: sensors, estimates, observation,
                  action, targets, command, safety flags
"""
from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path

import numpy as np


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


class RunLogger:
    def __init__(self, run_root, label: str, delta_deg: float, repetition: int):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(run_root).expanduser() / f"{stamp}_{label}_d{delta_deg:g}_r{repetition}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self._csv = None
        self._writer = None
        self.meta = {"created": datetime.datetime.now().isoformat(timespec="seconds")}

    def write_meta(self, **fields):
        self.meta.update(fields)
        (self.dir / "run.json").write_text(json.dumps(self.meta, indent=2, default=_jsonable))

    def log_tick(self, row: dict):
        flat = {}
        for k, v in row.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                for i, x in enumerate(np.ravel(v)):
                    flat[f"{k}_{i}"] = float(x)
            else:
                flat[k] = v
        if self._writer is None:
            self._csv = open(self.dir / "ticks.csv", "w", newline="")
            self._writer = csv.DictWriter(self._csv, fieldnames=list(flat), extrasaction="ignore")
            self._writer.writeheader()
        self._writer.writerow(flat)

    def close(self):
        if self._csv is not None:
            self._csv.close()
            self._csv = None
