"""
exported.py — load an exported policy folder (written by export/export_policy.py).

An export folder is self-contained, so the robot laptop only needs it:

    <export_dir>/
      policy.json          manifest: format version, provenance, policy metadata
      deployment.yaml      copy of the solution's contract (paths made local)
      gait/                reference gait arrays
      robot/nao.urdf       URDF used for the leg kinematics
      policy.npz           MLP weights            (PPO, SAC)
      pgrl_params.json     12 physical parameters (PGRL)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .controller import PGRLController, ResidualController
from .deployment_spec import DeploymentSpec, load_deployment
from .policies import MlpPolicy, PGRLParams

EXPORT_FORMAT_VERSION = 1


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ExportedPolicy:
    path: Path
    manifest: dict
    spec: DeploymentSpec
    urdf_path: Path
    mlp: Optional[MlpPolicy] = None
    pgrl: Optional[PGRLParams] = None

    @property
    def name(self) -> str:
        return self.spec.name

    def make_controller(self, leg_offset_rad=None) -> Union[ResidualController, PGRLController]:
        if self.mlp is not None:
            return ResidualController(self.spec, self.mlp, leg_offset_rad)
        return PGRLController(self.spec, self.pgrl, leg_offset_rad)


def load_export(export_dir) -> ExportedPolicy:
    d = Path(export_dir).expanduser().resolve()
    man_path = d / "policy.json"
    if not man_path.is_file():
        raise FileNotFoundError(f"{d}: no policy.json (is this an exported policy folder?)")
    man = json.loads(man_path.read_text())
    if man.get("format_version") != EXPORT_FORMAT_VERSION:
        raise ValueError(f"{man_path}: format_version {man.get('format_version')} "
                         f"(expected {EXPORT_FORMAT_VERSION}); re-export the policy")
    if not man.get("verification", {}).get("passed", False):
        raise ValueError(f"{man_path}: export verification did not pass; do not deploy")

    spec = load_deployment(d / "deployment.yaml")
    for rel, digest in man.get("files_sha256", {}).items():
        if sha256_of(d / rel) != digest:
            raise ValueError(f"{d / rel} changed after export (sha256 mismatch)")

    urdf = d / man["urdf"]
    pol = man["policy"]
    if pol["kind"] == "mlp":
        return ExportedPolicy(d, man, spec, urdf, mlp=MlpPolicy.load(d / pol["weights"], pol))
    if pol["kind"] == "pgrl":
        return ExportedPolicy(d, man, spec, urdf, pgrl=PGRLParams.load(d / pol["params"]))
    raise ValueError(f"{man_path}: unknown policy kind '{pol['kind']}'")
