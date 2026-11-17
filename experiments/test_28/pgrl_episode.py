"""
pgrl_episode.py — run ONE walking episode of a PGRL policy and score it.

The measurement (start point at the end of warm-up, forward = world −x,
success at the target distance with crossing-time interpolation, fall
classification, F_S) is NOT reimplemented here: it is imported read-only from
the common evaluation script (`experiments/test_evaluation_script/`). The F_S
that PGRL optimises during training is therefore computed by exactly the
same code that produces Table 5-3.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parents[0] / "test_evaluation_script"
sys.dont_write_bytecode = True
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from evaluate_algorithms import make_evaluation_env_class, run_episode  # noqa: E402


class _NoAction:
    """The PGRL controller lives inside the env; the 'action' is unused."""

    def act(self, obs):
        return None


def make_env(render_mode="headless"):
    import os
    from pgrl_env import PGRLWalkingEnv, BASELINE_DIR
    # The baseline env may resolve assets relative to the CWD, exactly as the
    # evaluation script does for PPO/SAC: run from its folder (read-only).
    os.chdir(BASELINE_DIR)
    return make_evaluation_env_class(PGRLWalkingEnv)(render_mode=render_mode)


def evaluate_theta(env, theta=None, phys=None, distance_m=1.0, offset_rad=None):
    """Run one episode with the given parameters; return run_episode's dict."""
    env.set_params(theta=theta, phys=phys)
    if offset_rad is None:
        offset_rad = np.zeros(env.n_legs, dtype=np.float32)   # δ = 0 (training)
    return run_episode(env, _NoAction(), distance_m, offset_rad)
