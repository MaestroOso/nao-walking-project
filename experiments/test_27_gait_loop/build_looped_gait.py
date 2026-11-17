#!/usr/bin/env python3
"""
build_looped_gait.py — test_27: extend the 2.6 s Webots Forwards.motion clip
to cover a full 50 s episode by appending copies of it back to back.

Input  (read-only): walking-solutions/shared/reference_gaits/forwards_*.{npy,txt}
Output (this folder): data/looped_keyframes.npy, data/looped_timestamps.npy,
                      data/looped_joint_names.txt

Timeline construction (keyframes are 40 ms apart, t = 0.00 ... 2.60 s):
    copy k occupies t = k*T + [0.00 ... 2.60]   with T = 2.60 + 0.04 = 2.64 s
so the last frame of copy k and the first frame of copy k+1 are also 40 ms
apart. All frames are kept, including the last one (no seam smoothing): the
known seam jump is ~0.058 rad (~3.3 deg) in hip roll, every other joint
< 0.005 rad.
"""
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "walking-solutions" / "shared" / "reference_gaits"
OUT = HERE / "data"

# Gait playback must last at least the episode length after warm-up:
# (1500 - 60) control steps * (1/30) s = 48 s. Use 50 s plus one spare copy.
TARGET_DURATION_S = 50.0


def main():
    q = np.load(SRC / "forwards_keyframes.npy")
    t = np.load(SRC / "forwards_timestamps.npy")
    names = (SRC / "forwards_joint_names.txt").read_text()

    dt = float(np.median(np.diff(t)))           # 0.04 s
    period = float(t[-1]) + dt                  # 2.64 s per copy
    n_copies = math.ceil(TARGET_DURATION_S / period) + 1

    q_loop = np.tile(q, (n_copies, 1))
    t_loop = np.concatenate([t + k * period for k in range(n_copies)])

    assert np.allclose(np.diff(t_loop), dt), "timestamps must be uniformly spaced"

    OUT.mkdir(exist_ok=True)
    np.save(OUT / "looped_keyframes.npy", q_loop.astype(q.dtype))
    np.save(OUT / "looped_timestamps.npy", t_loop.astype(t.dtype))
    (OUT / "looped_joint_names.txt").write_text(names)

    seam = q[0] - q[-1]
    print(f"Source clip : {q.shape[0]} frames, {t[-1]:.2f} s, dt={dt*1000:.0f} ms")
    print(f"Copies      : {n_copies}  (period {period:.2f} s)")
    print(f"Looped clip : {q_loop.shape[0]} frames, {t_loop[-1]:.2f} s")
    print(f"Seam jump   : max {np.degrees(np.abs(seam)).max():.2f} deg "
          f"(per joint, deg: {np.round(np.degrees(seam), 2).tolist()})")
    print(f"Written to  : {OUT}")


if __name__ == "__main__":
    main()
