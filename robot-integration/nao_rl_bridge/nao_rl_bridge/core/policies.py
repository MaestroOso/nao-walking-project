"""
policies.py — run exported policies with NumPy only (no torch / SB3).

Export format (written by robot-integration/export/export_policy.py):

* MLP (PPO, SAC): `policy.npz` with arrays W0, b0, W1, b1, ... (W as
  (out, in), like torch.nn.Linear) and the `policy` block of `policy.json`:
      {"kind": "mlp", "activation": "tanh"|"relu", "output": "clip"|"tanh",
       "obs_dim": 23, "act_dim": 12, "action_low": [...], "action_high": [...]}
  Deterministic action, as SB3 `predict(obs, deterministic=True)`:
      PPO ("clip"): a = clip(mean, low, high)
      SAC ("tanh"): a = low + (tanh(mu) + 1)/2 · (high - low)
* PGRL: `pgrl_params.json` with the 12 physical parameter values by name.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ACTIVATIONS = {"tanh": np.tanh, "relu": lambda x: np.maximum(x, 0.0)}
OUTPUTS = ("clip", "tanh")

PGRL_PARAM_NAMES = (
    "K_ank_pitch", "K_hip_pitch", "K_ank_roll", "K_hip_roll", "f", "a",
    "s_hip_yawpitch", "s_hip_roll", "s_hip_pitch", "s_knee_pitch",
    "s_ankle_pitch", "s_ankle_roll",
)


class MlpPolicy:
    def __init__(self, weights, biases, activation: str, output: str,
                 action_low, action_high):
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation '{activation}'")
        if output not in OUTPUTS:
            raise ValueError(f"unknown output '{output}'")
        if len(weights) != len(biases) or not weights:
            raise ValueError("weights/biases mismatch")
        for i in range(1, len(weights)):
            if weights[i].shape[1] != weights[i - 1].shape[0]:
                raise ValueError(f"layer {i}: shape {weights[i].shape} does not chain")
        self.W = [np.asarray(w, dtype=np.float32) for w in weights]
        self.b = [np.asarray(b, dtype=np.float32) for b in biases]
        self.act_fn = ACTIVATIONS[activation]
        self.activation, self.output = activation, output
        self.low = np.asarray(action_low, dtype=np.float32)
        self.high = np.asarray(action_high, dtype=np.float32)
        self.obs_dim = self.W[0].shape[1]
        self.act_dim = self.W[-1].shape[0]
        if self.low.shape != (self.act_dim,) or self.high.shape != (self.act_dim,):
            raise ValueError("action bounds do not match the output layer")

    @classmethod
    def load(cls, npz_path, meta: dict) -> "MlpPolicy":
        data = np.load(npz_path)
        n = int(meta["n_layers"])
        W = [data[f"W{i}"] for i in range(n)]
        b = [data[f"b{i}"] for i in range(n)]
        pol = cls(W, b, meta["activation"], meta["output"],
                  meta["action_low"], meta["action_high"])
        if pol.obs_dim != meta["obs_dim"] or pol.act_dim != meta["act_dim"]:
            raise ValueError("policy.json dims do not match the weights")
        return pol

    def act(self, obs) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(self.obs_dim)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = W @ x + b
            if i < len(self.W) - 1:
                x = self.act_fn(x)
        if self.output == "clip":
            return np.clip(x, self.low, self.high)
        return self.low + (np.tanh(x) + 1.0) * 0.5 * (self.high - self.low)


class PGRLParams:
    """The 12 physical PGRL parameters (report Table 4-6), by name."""

    def __init__(self, values: dict):
        missing = [n for n in PGRL_PARAM_NAMES if n not in values]
        if missing:
            raise ValueError(f"PGRL parameters missing: {missing}")
        self.values = {n: float(values[n]) for n in PGRL_PARAM_NAMES}
        v = self.values
        self.gains = np.array([v["K_ank_pitch"], v["K_hip_pitch"], v["K_ank_roll"], v["K_hip_roll"]])
        self.f, self.a = v["f"], v["a"]
        self.stiffness = np.array([v[n] for n in PGRL_PARAM_NAMES[6:]])   # per leg DOF type

    @classmethod
    def load(cls, path) -> "PGRLParams":
        return cls(json.loads(Path(path).read_text())["params_physical"])
