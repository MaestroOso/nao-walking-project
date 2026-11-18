"""Tests for export_policy.py with throwaway models (no real training results).

* PGRL: a fake run folder, checked against experiments/test_28/pgrl_params.py.
* PPO / SAC: tiny SB3 models on a dummy 23 -> 12 env; needs torch + SB3, so run
  in the Isaac/SB3 env:   setup/isaac_env/bin/python robot-integration/export/test_export.py
  (with system pytest, the SB3 tests are skipped).
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "nao_rl_bridge"))
sys.dont_write_bytecode = True

import export_policy as E  # noqa: E402
from nao_rl_bridge.core.exported import load_export  # noqa: E402

REPO = E.REPO_ROOT
PPO_YAML = REPO / "walking-solutions/ppo/baseline_test24/deployment.yaml"
TEST28 = REPO / "experiments/test_28"


def _solution(tmp: Path, d: dict) -> Path:
    for k in ("keyframes", "timestamps", "joint_names"):
        p = Path(d["reference_gait"][k])
        if not p.is_absolute():
            d["reference_gait"][k] = str((PPO_YAML.parent / p).resolve())
    sol = tmp / "solution"
    sol.mkdir()
    (sol / "deployment.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    return sol


def _ppo_dict():
    return yaml.safe_load(PPO_YAML.read_text())


def test_pgrl_export_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        looped = REPO / "experiments/test_27_gait_loop/data"
        d = _ppo_dict()
        d.update(name="pgrl_t", algorithm="pgrl")
        d["policy"] = {"format": "pgrl_run", "run_dir": str(tmp / "run")}
        d["reference_gait"] = {"keyframes": str(looped / "looped_keyframes.npy"),
                               "timestamps": str(looped / "looped_timestamps.npy"),
                               "joint_names": str(looped / "looped_joint_names.txt"),
                               "after_end": "wrap", "phase_period_s": 2.64}
        del d["action"], d["observation"]
        d["pgrl"] = {"controller": "tilt_compensator_v1"}
        sol = _solution(tmp, d)
        shutil.copy(TEST28 / "pgrl_params.py", sol / "pgrl_params.py")   # enables the cross-check
        (tmp / "run").mkdir()
        theta = np.linspace(0, 1, 12)
        sys.path.insert(0, str(TEST28))
        import pgrl_params as P
        (tmp / "run/state.json").write_text(json.dumps({"iteration": 3, "evals": 36, "theta": theta.tolist()}))
        (tmp / "run/config.json").write_text(json.dumps({"param_specs": P.PARAM_SPECS}))

        out, ver = E.export(sol, out=tmp / "exp")
        assert ver["passed"] and "pgrl_params.py" in ver["method"]
        ex = load_export(out)
        np.testing.assert_allclose([ex.pgrl.values[n] for n in P.NAMES], P.to_physical(theta))
        ctl = ex.make_controller()
        assert ctl.spec.gait.keyframes.parent == out / "gait"
        man = json.loads((out / "policy.json").read_text())
        assert set(man["files_sha256"]) >= {"deployment.yaml", "pgrl_params.json", "robot/nao.urdf"}

        # Tampering with an exported file must be detected.
        (out / "pgrl_params.json").write_text((out / "pgrl_params.json").read_text() + " ")
        try:
            load_export(out)
            raise AssertionError("tampering not detected")
        except ValueError as e:
            assert "sha256" in str(e)

        # Existing folder: refused without --force, replaced with it.
        try:
            E.export(sol, out=tmp / "exp")
            raise AssertionError("overwrite not refused")
        except E.ExportError:
            pass
        E.export(sol, out=tmp / "exp", force=True)


def _sb3_case(algo_cls_name, tmp):
    import gymnasium as gym
    import stable_baselines3 as sb3

    class Dummy(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (23,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)

        def reset(self, seed=None, options=None):
            return np.zeros(23, np.float32), {}

        def step(self, a):
            return np.random.normal(size=23).astype(np.float32), 0.0, False, False, {}

    cls = getattr(sb3, algo_cls_name)
    kw = dict(n_steps=64, batch_size=32) if algo_cls_name == "PPO" else dict(learning_starts=10, buffer_size=500)
    model = cls("MlpPolicy", Dummy(), policy_kwargs={"net_arch": [16, 16]}, device="cpu", seed=0, **kw)
    model.learn(128)
    model.save(tmp / "model.zip")

    d = _ppo_dict()
    d.update(name=f"{algo_cls_name.lower()}_t", algorithm=algo_cls_name.lower())
    d["policy"] = {"format": "sb3", "sb3_class": algo_cls_name, "model": str(tmp / "model.zip")}
    out, ver = E.export(_solution(tmp, d), out=tmp / "exp")
    assert ver["passed"], ver
    ex = load_export(out)
    obs = np.random.default_rng(1).normal(size=(200, 23)).astype(np.float32)
    ref = model.predict(obs, deterministic=True)[0]
    np.testing.assert_allclose(np.stack([ex.mlp.act(o) for o in obs]), ref, atol=1e-5)
    ex.make_controller()
    return ver


def test_ppo_export():
    import pytest
    pytest.importorskip("stable_baselines3")
    with tempfile.TemporaryDirectory() as td:
        _sb3_case("PPO", Path(td))


def test_sac_export():
    import pytest
    pytest.importorskip("stable_baselines3")
    with tempfile.TemporaryDirectory() as td:
        _sb3_case("SAC", Path(td))


if __name__ == "__main__":       # Isaac/SB3 env has no pytest
    test_pgrl_export_roundtrip()
    print("pgrl export: OK")
    for algo in ("PPO", "SAC"):
        with tempfile.TemporaryDirectory() as td:
            v = _sb3_case(algo, Path(td))
            print(f"{algo} export: OK (max |error| {v['max_abs_error']:.2e} on {v['n_samples']} obs)")
