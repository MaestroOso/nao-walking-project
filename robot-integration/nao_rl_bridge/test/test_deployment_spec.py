"""Tests for core/deployment_spec.py.

Run from the repo root:  python3 -m pytest robot-integration/nao_rl_bridge/test -q

Each test starts from the real PPO baseline deployment.yaml (with its relative
paths made absolute so the file can be written to a temp dir), applies one
change, and checks that the validator accepts or rejects it as expected.
"""
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

PKG_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_DIR.parents[1]
sys.path.insert(0, str(PKG_DIR))

from nao_rl_bridge.core.deployment_spec import (  # noqa: E402
    DeploymentSpecError, load_deployment)

PPO_YAML = REPO_ROOT / "walking-solutions/ppo/baseline_test24/deployment.yaml"
LOOPED_GAIT = REPO_ROOT / "experiments/test_27_gait_loop/data"


def _base_dict():
    d = yaml.safe_load(PPO_YAML.read_text())
    for k in ("keyframes", "timestamps", "joint_names"):
        d["reference_gait"][k] = str((PPO_YAML.parent / d["reference_gait"][k]).resolve())
    return d


def _write(tmp_path, d, name="deployment.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


def _errors(tmp_path, d):
    with pytest.raises(DeploymentSpecError) as ei:
        load_deployment(_write(tmp_path, d))
    return "\n".join(ei.value.errors)


# ── the real file ────────────────────────────────────────────────────────────
def test_ppo_baseline_is_valid():
    spec = load_deployment(PPO_YAML)
    assert spec.algorithm == "ppo"
    assert spec.observation.dim == 23
    assert spec.observation.quaternion_convention == "legacy_xyzw"
    assert spec.limits_rad.shape == (12, 2)
    assert math.isclose(spec.control_dt, 1 / 30)
    assert math.isclose(spec.gait.phase_period_s, 2.6, rel_tol=1e-6)
    q, t = spec.load_gait_arrays()
    assert q.shape == (66, 12)
    assert np.degrees(np.abs(q[0] - spec.home_pose_rad)).max() < 0.5


def test_missing_model_is_warning_unless_required(tmp_path):
    d = _base_dict()
    d["policy"]["model"] = str(tmp_path / "nope.zip")
    spec = load_deployment(_write(tmp_path, d))
    assert any("artefacts not found" in w for w in spec.warnings)
    with pytest.raises(DeploymentSpecError):
        load_deployment(_write(tmp_path, d), require_artifacts=True)


def test_existing_model_passes_required(tmp_path):
    d = _base_dict()
    (tmp_path / "m.zip").write_bytes(b"x")
    d["policy"]["model"] = "m.zip"                 # relative to the yaml's folder
    spec = load_deployment(_write(tmp_path, d), require_artifacts=True)
    assert spec.policy.model == (tmp_path / "m.zip").resolve()


# ── the other two algorithms ─────────────────────────────────────────────────
def test_sac_style_spec_is_valid(tmp_path):
    d = _base_dict()
    d.update(name="sac_example", algorithm="sac")
    d["policy"]["sb3_class"] = "SAC"
    d["observation"]["quaternion_convention"] = "wxyz"
    d["observation"]["components"][2]["frame"] = "heading"
    spec = load_deployment(_write(tmp_path, d))
    assert spec.observation.component("linear_velocity").options["frame"] == "heading"


@pytest.mark.skipif(not LOOPED_GAIT.is_dir(), reason="test_27 looped gait not present")
def test_pgrl_style_spec_is_valid(tmp_path):
    d = _base_dict()
    d.update(name="pgrl_example", algorithm="pgrl")
    d["policy"] = {"format": "pgrl_run", "run_dir": str(tmp_path / "run")}
    d["reference_gait"] = {
        "keyframes": str(LOOPED_GAIT / "looped_keyframes.npy"),
        "timestamps": str(LOOPED_GAIT / "looped_timestamps.npy"),
        "joint_names": str(LOOPED_GAIT / "looped_joint_names.txt"),
        "after_end": "wrap", "phase_period_s": 2.64,
    }
    del d["action"], d["observation"]
    d["pgrl"] = {"controller": "tilt_compensator_v1"}
    spec = load_deployment(_write(tmp_path, d))
    assert spec.pgrl.controller == "tilt_compensator_v1"
    assert spec.observation is None and spec.action is None
    assert [p.name for p in spec.artifact_paths()] == ["state.json", "config.json"]


# ── things that must be rejected ─────────────────────────────────────────────
@pytest.mark.parametrize("mutate, expected", [
    (lambda d: d.update(algorithm="td3"), "algorithm 'td3'"),
    (lambda d: d["policy"].update(sb3_class="SAC"), "does not match algorithm"),
    (lambda d: d["policy"].update(format="pgrl_run"), "does not match algorithm"),
    (lambda d: d.update(schema_version=2), "schema_version 2"),
    (lambda d: d["control"].update(warmup_steps=2000), "max_episode_steps must be > warmup"),
    (lambda d: d["joints"]["leg_order"].__setitem__(0, "LHipRoll"),   # duplicate joint
     "each of the 12 NAO leg joints once"),
    (lambda d: d["joints"]["limits_deg"].update(LKneePitch=[121.0, 2.0]), "lower (121.0) must be < upper"),
    (lambda d: d["joints"]["limits_deg"].pop("RAnkleRoll"), "missing joints ['RAnkleRoll']"),
    (lambda d: d["joints"]["home_pose_deg"].update(LKneePitch=65.0), "first frame by"),
    (lambda d: d["joints"]["home_pose_deg"].update(LKneePitch=130.0), "outside its limits"),
    (lambda d: d["joints"]["fixed_pose_deg"].update(LKneePitch=0), "not a NAO non-leg joint"),
    (lambda d: d["reference_gait"].update(keyframes="/no/such.npy"), "files not found"),
    (lambda d: d["reference_gait"].update(after_end="loop"), "after_end 'loop'"),
    (lambda d: d["action"].update(scale_rad=0), "scale_rad must be > 0"),
    (lambda d: d["observation"].update(quaternion_convention="xyzw"), "quaternion_convention 'xyzw'"),
    (lambda d: d["observation"]["components"][0].update(dim=4), "dim must be 3"),
    (lambda d: d["observation"]["components"][2].update(frame="body"), "'body' must be one of"),
    (lambda d: d["observation"]["components"][1].pop("scale"), "missing option 'scale'"),
    (lambda d: d["observation"]["components"].append({"name": "base_height", "dim": 1}), "listed twice"),
    (lambda d: d["observation"]["components"].append({"name": "foot_contacts", "dim": 2}), "unknown component"),
    (lambda d: d.update(pgrl={"controller": "tilt_compensator_v1"}), "only valid for pgrl_run"),
])
def test_rejects(tmp_path, mutate, expected):
    d = copy.deepcopy(_base_dict())
    mutate(d)
    assert expected in _errors(tmp_path, d)


def test_reports_all_errors_at_once(tmp_path):
    d = _base_dict()
    d["action"]["scale_rad"] = -1
    d["observation"]["components"][0]["dim"] = 9
    d["control"]["control_hz"] = 0
    errs = _errors(tmp_path, d)
    assert "scale_rad" in errs and "dim must be 3" in errs and "control_hz" in errs
