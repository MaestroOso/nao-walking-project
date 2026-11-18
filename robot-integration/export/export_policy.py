#!/usr/bin/env python3
"""
export_policy.py — turn a trained solution into a self-contained folder for the robot.

    python robot-integration/export/export_policy.py walking-solutions/<algo>/<variant>
        [--model PATH]      SB3 .zip (overrides deployment.yaml policy.model)
        [--run-dir PATH]    PGRL run folder (overrides deployment.yaml policy.run_dir)
        [--out DIR]         default: robot-integration/exported/<name>
        [--force]           replace an existing export folder

PPO / SAC need torch + stable_baselines3: run inside the Isaac/SB3 env
(`setup/isaac_env/bin/python`). PGRL only needs NumPy + PyYAML.

What it does
------------
1. Loads and validates the solution's deployment.yaml (the trained artefact
   must exist).
2. PPO / SAC: extracts the actor network from the SB3 model as plain arrays
   (policy.npz). The model must be an `MlpPolicy` with a flatten feature
   extractor and no gSDE, so the deterministic action is a pure MLP.
   PGRL: maps the run's normalised θ (state.json) to physical values with the
   parameter ranges that run was trained with (config.json).
3. Verification. PPO / SAC: the NumPy policy must reproduce
   `model.predict(obs, deterministic=True)` on thousands of observations
   (max error < 1e-4), both directly and after re-loading the export folder
   with the robot-side loader. PGRL: the conversion is checked against the
   solution's own `pgrl_params.py` if the folder has one.
   A failed verification aborts the export: nothing is written.
4. Writes the folder (see nao_rl_bridge/core/exported.py for the layout):
   copies of the gait arrays and the URDF, the deployment.yaml with local
   paths, and policy.json with provenance (git commit, sha256 of every input
   and output file) and the verification result.
"""
import argparse
import datetime
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROBOT_DIR = HERE.parent
REPO_ROOT = ROBOT_DIR.parent
sys.path.insert(0, str(ROBOT_DIR / "nao_rl_bridge"))

from nao_rl_bridge.core.deployment_spec import load_deployment  # noqa: E402
from nao_rl_bridge.core.exported import (EXPORT_FORMAT_VERSION, load_export,  # noqa: E402
                                         sha256_of)
from nao_rl_bridge.core.policies import PGRL_PARAM_NAMES, MlpPolicy  # noqa: E402

DEFAULT_URDF = REPO_ROOT / "walking-solutions/shared/robot/nao.urdf"
MAX_ABS_ERROR = 1e-4
N_VERIFY = 4096


class ExportError(RuntimeError):
    pass


# ═════════════════════════════════════════════════════════════════════════════
# SB3 (PPO, SAC)
# ═════════════════════════════════════════════════════════════════════════════
def _linear_stack(seq, head):
    """Linear layers + their (single) activation from an nn.Sequential and a head."""
    import torch.nn as nn
    acts = {nn.Tanh: "tanh", nn.ReLU: "relu"}
    layers, used = [], set()
    for m in seq:
        if isinstance(m, nn.Linear):
            layers.append(m)
        elif type(m) in acts:
            used.add(acts[type(m)])
        else:
            raise ExportError(f"unsupported layer in policy network: {type(m).__name__}")
    if len(used) > 1:
        raise ExportError(f"mixed activations {used} are not supported")
    layers.append(head)
    W = [l.weight.detach().cpu().numpy().astype(np.float32) for l in layers]
    b = [l.bias.detach().cpu().numpy().astype(np.float32) for l in layers]
    return W, b, (used.pop() if used else "tanh")


def extract_sb3(model, sb3_class):
    """(weights, biases, activation, output) of the deterministic actor."""
    from stable_baselines3.common.torch_layers import FlattenExtractor
    pol = model.policy
    if getattr(model, "use_sde", False):
        raise ExportError("gSDE policies are not supported (use_sde=True)")
    if sb3_class == "PPO":
        fe = getattr(pol, "pi_features_extractor", pol.features_extractor)
        seq, head, output = pol.mlp_extractor.policy_net, pol.action_net, "clip"
    elif sb3_class == "SAC":
        actor = pol.actor
        fe = actor.features_extractor
        seq, head, output = actor.latent_pi, actor.mu, "tanh"
    else:
        raise ExportError(f"unsupported SB3 class {sb3_class}")
    if not isinstance(fe, FlattenExtractor):
        raise ExportError(f"unsupported features extractor {type(fe).__name__}")
    W, b, act = _linear_stack(seq, head)
    return W, b, act, output


def export_sb3(spec, model_path: Path):
    try:
        import stable_baselines3 as sb3
    except ImportError:
        raise ExportError("stable_baselines3 not importable: run this in the Isaac/SB3 env "
                          "(setup/isaac_env/bin/python)")
    cls = getattr(sb3, spec.policy.sb3_class)
    model = cls.load(str(model_path), device="cpu")
    obs_space, act_space = model.observation_space, model.action_space
    if tuple(obs_space.shape) != (spec.observation.dim,):
        raise ExportError(f"model observation shape {obs_space.shape} != deployment "
                          f"observation dim {spec.observation.dim}")
    if tuple(act_space.shape) != (12,):
        raise ExportError(f"model action shape {act_space.shape} != (12,)")
    low, high = act_space.low.astype(np.float32), act_space.high.astype(np.float32)
    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        raise ExportError("action space bounds must be finite")
    if list(model_path.parent.glob("*vecnormalize*.pkl")):
        raise ExportError(f"found a VecNormalize file next to {model_path.name}: observation "
                          f"normalisation is not supported by the robot runtime")

    W, b, act, output = extract_sb3(model, spec.policy.sb3_class)
    policy = MlpPolicy(W, b, act, output, low, high)

    rng = np.random.default_rng(0)
    obs = np.concatenate([rng.normal(0, 1, (N_VERIFY // 2, policy.obs_dim)),
                          rng.uniform(-3, 3, (N_VERIFY // 2, policy.obs_dim))]).astype(np.float32)
    ref = model.predict(obs, deterministic=True)[0]
    mine = np.stack([policy.act(o) for o in obs])
    err = float(np.max(np.abs(ref - mine)))
    meta = {
        "kind": "mlp", "weights": "policy.npz", "n_layers": len(W),
        "activation": act, "output": output,
        "obs_dim": policy.obs_dim, "act_dim": policy.act_dim,
        "hidden_sizes": [int(w.shape[0]) for w in W[:-1]],
        "action_low": low.tolist(), "action_high": high.tolist(),
    }
    source = {
        "model": str(model_path), "model_sha256": sha256_of(model_path),
        "sb3_class": spec.policy.sb3_class, "sb3_version": sb3.__version__,
        "model_num_timesteps": int(getattr(model, "num_timesteps", -1)),
    }
    verification = {"method": "SB3 predict(deterministic=True) vs NumPy MLP",
                    "n_samples": int(len(obs)), "max_abs_error": err,
                    "tolerance": MAX_ABS_ERROR, "passed": err < MAX_ABS_ERROR}
    files = {"policy.npz": {f"W{i}": w for i, w in enumerate(W)} | {f"b{i}": x for i, x in enumerate(b)}}
    return meta, source, verification, files, (obs, ref)


# ═════════════════════════════════════════════════════════════════════════════
# PGRL
# ═════════════════════════════════════════════════════════════════════════════
def export_pgrl(spec, run_dir: Path):
    state = json.loads((run_dir / "state.json").read_text())
    cfg = json.loads((run_dir / "config.json").read_text())
    specs = cfg.get("param_specs")
    if not specs:
        raise ExportError(f"{run_dir}/config.json has no 'param_specs': the parameter ranges "
                          f"of this run are unknown, so θ cannot be mapped to physical values")
    names = [s[0] for s in specs]
    if tuple(names) != PGRL_PARAM_NAMES:
        raise ExportError(f"unexpected PGRL parameter list {names} (expected {list(PGRL_PARAM_NAMES)})")
    theta = np.asarray(state["theta"], dtype=np.float64)
    if theta.shape != (12,) or not np.all(np.isfinite(theta)):
        raise ExportError("state.json theta must be 12 finite values")
    low = np.array([s[1] for s in specs], float)
    high = np.array([s[2] for s in specs], float)
    phys = low + np.clip(theta, 0.0, 1.0) * (high - low)

    verification = {"method": "own conversion low + θ·(high - low)", "passed": True}
    helper = spec.path.parent / "pgrl_params.py"
    if helper.is_file():
        mspec = importlib.util.spec_from_file_location("solution_pgrl_params", helper)
        mod = importlib.util.module_from_spec(mspec)
        mspec.loader.exec_module(mod)
        ref = np.asarray(mod.to_physical(theta, [tuple(s) for s in specs]))
        err = float(np.max(np.abs(ref - phys)))
        where = helper.relative_to(REPO_ROOT) if helper.is_relative_to(REPO_ROOT) else helper
        verification.update(method=f"vs {where}.to_physical",
                            max_abs_error=err, passed=err < 1e-12)
    else:
        verification["note"] = f"no pgrl_params.py next to {spec.path.name}: not cross-checked"

    at_bound = [n for n, t in zip(names, theta) if t <= 0.0 or t >= 1.0]
    params = {"params_physical": dict(zip(names, phys.tolist())),
              "param_specs": specs, "theta": theta.tolist(),
              "parameters_at_range_bound": at_bound}
    meta = {"kind": "pgrl", "params": "pgrl_params.json", "controller": spec.pgrl.controller}
    source = {"run_dir": str(run_dir),
              "state_sha256": sha256_of(run_dir / "state.json"),
              "config_sha256": sha256_of(run_dir / "config.json"),
              "iterations": state.get("iteration"), "evaluations": state.get("evals")}
    return meta, source, verification, {"pgrl_params.json": params}, None


# ═════════════════════════════════════════════════════════════════════════════
# Folder assembly
# ═════════════════════════════════════════════════════════════════════════════
def git_info():
    def run(*cmd):
        try:
            return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def write_folder(tmp: Path, spec, urdf: Path, meta, source, verification, files):
    (tmp / "gait").mkdir()
    (tmp / "robot").mkdir()
    gait_files = {}
    for key in ("keyframes", "timestamps", "joint_names"):
        src = getattr(spec.gait, key)
        shutil.copy2(src, tmp / "gait" / src.name)
        gait_files[key] = f"gait/{src.name}"
    shutil.copy2(urdf, tmp / "robot" / "nao.urdf")

    raw = yaml.safe_load(spec.path.read_text())
    raw["reference_gait"].update(gait_files)
    if spec.policy.format == "sb3":
        raw["policy"]["model"] = source["model"]
    else:
        raw["policy"]["run_dir"] = source["run_dir"]
    header = (f"# Exported copy of {spec.path} on {datetime.date.today()}.\n"
              f"# Gait paths point to this folder; the policy path is provenance only.\n")
    (tmp / "deployment.yaml").write_text(header + yaml.safe_dump(raw, sort_keys=False))

    for name, content in files.items():
        if name.endswith(".npz"):
            np.savez(tmp / name, **content)
        else:
            (tmp / name).write_text(json.dumps(content, indent=2))

    out_files = sorted(p.relative_to(tmp).as_posix() for p in tmp.rglob("*") if p.is_file())
    manifest = {
        "format_version": EXPORT_FORMAT_VERSION,
        "name": spec.name, "algorithm": spec.algorithm,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "source": {"deployment_yaml": str(spec.path),
                   "deployment_sha256": sha256_of(spec.path), **source},
        "policy": meta,
        "urdf": "robot/nao.urdf",
        "verification": verification,
        "files_sha256": {f: sha256_of(tmp / f) for f in out_files},
    }
    (tmp / "policy.json").write_text(json.dumps(manifest, indent=2))


def export(solution: Path, model=None, run_dir=None, out=None, force=False, urdf=DEFAULT_URDF):
    yaml_path = solution / "deployment.yaml" if solution.is_dir() else solution
    override = model if model is not None else run_dir
    spec = load_deployment(yaml_path, require_artifacts=override is None)
    if spec.policy.format == "sb3":
        if run_dir is not None:
            raise ExportError("--run-dir is only for PGRL solutions")
        src = Path(model).expanduser().resolve() if model else spec.policy.model
        if not src.is_file():
            raise ExportError(f"model not found: {src}")
        result = export_sb3(spec, src)
    else:
        if model is not None:
            raise ExportError("--model is only for PPO/SAC solutions")
        src = Path(run_dir).expanduser().resolve() if run_dir else spec.policy.run_dir
        for f in ("state.json", "config.json"):
            if not (src / f).is_file():
                raise ExportError(f"PGRL run file not found: {src / f}")
        result = export_pgrl(spec, src)
    meta, source, verification, files, samples = result
    if not verification["passed"]:
        raise ExportError(f"verification failed, nothing written: {verification}")

    out = Path(out).expanduser().resolve() if out else ROBOT_DIR / "exported" / spec.name
    if out.exists():
        if not force:
            raise ExportError(f"{out} exists (use --force to replace it)")
        if not (out / "policy.json").is_file():
            raise ExportError(f"{out} exists and is not an export folder; refusing to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as td:
        tmp = Path(td) / "build"
        tmp.mkdir()
        write_folder(tmp, spec, Path(urdf), meta, source, verification, files)

        # Second check, through the loader the robot uses.
        loaded = load_export(tmp)
        if samples is not None:
            obs, ref = samples
            err = float(np.max(np.abs(np.stack([loaded.mlp.act(o) for o in obs]) - ref)))
            if err >= MAX_ABS_ERROR:
                raise ExportError(f"re-loaded export differs from SB3 by {err}")
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
    return out, verification


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("solution", type=Path, help="solution folder or its deployment.yaml")
    ap.add_argument("--model", type=Path)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    a = ap.parse_args()
    try:
        out, ver = export(a.solution, a.model, a.run_dir, a.out, a.force, a.urdf)
    except ExportError as e:
        print(f"[export] ERROR: {e}")
        return 1
    print(f"[export] verification: {ver}")
    print(f"[export] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
