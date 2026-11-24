"""
deployment_spec.py — the contract between ``walking-solutions/`` and the robot.

Every trained solution that may be deployed on the real NAO ships a
``deployment.yaml`` in its folder (e.g. ``walking-solutions/ppo/<variant>/``).
That file states, as plain data, everything the robot side must reproduce
exactly as it was during training:

  * which artefact holds the policy (an SB3 ``.zip`` or a PGRL run folder);
  * the control schedule (30 Hz, warm-up, episode limit);
  * the joint order, joint limits, home pose and fixed (non-leg) pose;
  * the reference gait files and what happens after the clip ends;
  * the action mapping (residual on the gait, scale in rad);
  * the observation vector, component by component, including the
    quaternion convention and the frame of the velocity input.

Why a declarative file instead of importing the training env?  The training
env needs Isaac Sim (Python 3.10) and its internals change between versions;
the robot runs ROS 2 Jazzy (Python 3.12) with no simulator. Reading a small,
validated file decouples the two and makes the sim-to-real assumptions
explicit and reviewable (see ``robot-integration/docs/deployment-contract.md``).

This module is pure Python + NumPy + PyYAML so it can be imported both by the
export tools (Isaac env) and by the ROS 2 node (system Python).

Usage
-----
    spec = load_deployment("walking-solutions/ppo/baseline_test24/deployment.yaml")
    spec.limits_rad            # (12,) in spec.leg_order
    spec.observation.dim       # 23

``load_deployment`` collects *all* problems and raises one
``DeploymentSpecError`` listing them; non-fatal findings are returned in
``spec.warnings``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

SCHEMA_VERSION = 1

# ── NAOqi joint names (NAO H25 body; same names as the URDF / Isaac DOFs) ──
NAO_LEG_JOINTS: Tuple[str, ...] = (
    "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
    "RHipYawPitch", "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll",
)
NAO_NON_LEG_JOINTS: Tuple[str, ...] = (
    "HeadYaw", "HeadPitch",
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw", "LHand",
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand",
)

ALGORITHMS = ("ppo", "sac", "pgrl")
POLICY_FORMATS = ("sb3", "pgrl_run")
SB3_CLASSES = ("PPO", "SAC")
ALGO_TO_FORMAT = {"ppo": "sb3", "sac": "sb3", "pgrl": "pgrl_run"}
GAIT_AFTER_END = ("hold_last", "wrap")
ACTION_TYPES = ("residual_on_gait",)
PGRL_CONTROLLERS = ("tilt_compensator_v1",)
QUAT_CONVENTIONS = ("legacy_xyzw", "wxyz")

# Observation vocabulary: name -> (fixed dim, {option: allowed values or type})
OBS_COMPONENTS: Dict[str, Tuple[int, Dict[str, Any]]] = {
    "projected_gravity":   (3,  {}),
    "angular_velocity":    (3,  {"scale": float}),
    "linear_velocity":     (2,  {"frame": ("world_xy", "heading")}),
    "leg_joint_positions": (12, {"normalization": ("limits_to_unit",)}),
    "base_height":         (1,  {}),
    "gait_phase":          (2,  {"warmup_value": list}),
}

HOME_VS_GAIT_TOL_DEG = 0.5   # same tolerance the training env reports


class DeploymentSpecError(ValueError):
    """Raised with every problem found in a deployment.yaml."""

    def __init__(self, path: Path, errors: List[str]):
        self.path = path
        self.errors = errors
        msg = f"{path}: {len(errors)} problem(s)\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(msg)


# ═════════════════════════════════════════════════════════════════════════════
# Data classes (what the rest of robot-integration consumes)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PolicySpec:
    format: str                       # "sb3" | "pgrl_run"
    sb3_class: Optional[str]          # "PPO" | "SAC" (sb3 only)
    model: Optional[Path]             # SB3 .zip (sb3 only)
    run_dir: Optional[Path]           # folder with state.json + config.json (pgrl_run only)


@dataclass(frozen=True)
class GaitSpec:
    keyframes: Path                   # (N, J) float array, radians
    timestamps: Path                  # (N,) seconds, strictly increasing
    joint_names: Path                 # J names, one per line
    after_end: str                    # "hold_last" | "wrap"
    phase_period_s: float             # period of the phase signal (s)


@dataclass(frozen=True)
class ActionSpec:
    type: str                         # "residual_on_gait"
    scale_rad: float                  # max correction per joint for |a| = 1


@dataclass(frozen=True)
class ObsComponent:
    name: str
    dim: int
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationSpec:
    quaternion_convention: str        # "legacy_xyzw" | "wxyz"
    spawn_yaw_deg: float              # world yaw of the robot at spawn
    nonfinite_to_zero: bool
    components: Tuple[ObsComponent, ...]

    @property
    def dim(self) -> int:
        return sum(c.dim for c in self.components)

    def component(self, name: str) -> Optional[ObsComponent]:
        return next((c for c in self.components if c.name == name), None)


@dataclass(frozen=True)
class PGRLSpec:
    controller: str                   # "tilt_compensator_v1"


@dataclass(frozen=True)
class DeploymentSpec:
    path: Path
    name: str
    algorithm: str
    description: str
    policy: PolicySpec
    control_hz: float
    warmup_steps: int
    max_episode_steps: int
    leg_order: Tuple[str, ...]
    limits_rad: np.ndarray            # (12, 2) [lower, upper], in leg_order
    home_pose_rad: np.ndarray         # (12,), in leg_order
    fixed_pose_rad: Dict[str, float]  # non-leg joint -> rad
    gait: GaitSpec
    action: Optional[ActionSpec]      # sb3 only
    observation: Optional[ObservationSpec]  # sb3 only
    pgrl: Optional[PGRLSpec]          # pgrl only
    warnings: Tuple[str, ...] = ()
    # False: the residual action is ignored during warm-up (SAC since
    # sac_pilot_v4, env warmup_actions=False). Optional, default True (PPO).
    warmup_actions: bool = True

    @property
    def control_dt(self) -> float:
        return 1.0 / self.control_hz

    def artifact_paths(self) -> List[Path]:
        """Files the export step needs (may be absent before training ends)."""
        if self.policy.format == "sb3":
            return [self.policy.model]
        return [self.policy.run_dir / "state.json", self.policy.run_dir / "config.json"]

    def load_gait_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """(keyframes (N,12) in leg_order [rad], timestamps (N,) [s])."""
        q, t, names = _read_gait(self.gait)
        cols = [names.index(n) for n in self.leg_order]
        return q[:, cols].astype(np.float64), t.astype(np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# Loading and validation
# ═════════════════════════════════════════════════════════════════════════════
class _Collector:
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def err(self, msg: str):
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)


def _resolve(base: Path, value: str) -> Path:
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def _read_gait(g: GaitSpec):
    q = np.load(g.keyframes)
    t = np.load(g.timestamps)
    names = [n.strip() for n in Path(g.joint_names).read_text().splitlines() if n.strip()]
    return q, t, names


def _req(d: dict, key: str, typ, c: _Collector, where: str):
    """Required key of a given type (int accepted where float is expected)."""
    if not isinstance(d, dict) or key not in d:
        c.err(f"{where}: missing required key '{key}'")
        return None
    v = d[key]
    ok = isinstance(v, typ) or (typ is float and isinstance(v, int) and not isinstance(v, bool))
    if not ok:
        c.err(f"{where}.{key}: expected {getattr(typ, '__name__', typ)}, got {type(v).__name__}")
        return None
    return float(v) if typ is float else v


def _joint_table(d: Any, names, where: str, c: _Collector, pair: bool) -> Dict[str, Any]:
    """Validate a {joint: deg} or {joint: [lo, hi]} table covering exactly `names`."""
    if not isinstance(d, dict):
        c.err(f"{where}: expected a mapping of joint name -> value")
        return {}
    missing = [n for n in names if n not in d]
    extra = [n for n in d if n not in names]
    if missing:
        c.err(f"{where}: missing joints {missing}")
    if extra:
        c.err(f"{where}: unexpected joints {extra}")
    out = {}
    for n in names:
        if n not in d:
            continue
        v = d[n]
        if pair:
            if (not isinstance(v, (list, tuple)) or len(v) != 2
                    or not all(isinstance(x, (int, float)) for x in v)):
                c.err(f"{where}.{n}: expected [lower, upper] in degrees")
                continue
            if not v[0] < v[1]:
                c.err(f"{where}.{n}: lower ({v[0]}) must be < upper ({v[1]})")
                continue
            out[n] = (float(v[0]), float(v[1]))
        else:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                c.err(f"{where}.{n}: expected a number (degrees)")
                continue
            out[n] = float(v)
    return out


def load_deployment(path, require_artifacts: bool = False) -> DeploymentSpec:
    """Load and validate a deployment.yaml.

    require_artifacts: if True, a missing model / PGRL run is an error
    (use when exporting or deploying); otherwise it is only a warning,
    because trained artefacts are gitignored and may not exist yet.
    """
    path = Path(path).resolve()
    base = path.parent
    c = _Collector()
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as e:
        raise DeploymentSpecError(path, [f"cannot read YAML: {e}"])
    if not isinstance(raw, dict):
        raise DeploymentSpecError(path, ["top level must be a mapping"])

    # ── header ──
    ver = _req(raw, "schema_version", int, c, "root")
    if ver is not None and ver != SCHEMA_VERSION:
        c.err(f"schema_version {ver} not supported (expected {SCHEMA_VERSION})")
    name = _req(raw, "name", str, c, "root") or ""
    algorithm = _req(raw, "algorithm", str, c, "root")
    if algorithm is not None and algorithm not in ALGORITHMS:
        c.err(f"algorithm '{algorithm}' must be one of {ALGORITHMS}")
        algorithm = None
    description = raw.get("description", "") or ""

    # ── policy ──
    pol = raw.get("policy")
    policy = None
    if not isinstance(pol, dict):
        c.err("root: missing required section 'policy'")
    else:
        fmt = _req(pol, "format", str, c, "policy")
        if fmt is not None and fmt not in POLICY_FORMATS:
            c.err(f"policy.format '{fmt}' must be one of {POLICY_FORMATS}")
            fmt = None
        if fmt and algorithm and ALGO_TO_FORMAT[algorithm] != fmt:
            c.err(f"policy.format '{fmt}' does not match algorithm '{algorithm}' "
                  f"(expected '{ALGO_TO_FORMAT[algorithm]}')")
        sb3_class = model = run_dir = None
        if fmt == "sb3":
            sb3_class = _req(pol, "sb3_class", str, c, "policy")
            if sb3_class is not None and sb3_class not in SB3_CLASSES:
                c.err(f"policy.sb3_class '{sb3_class}' must be one of {SB3_CLASSES}")
            if sb3_class and algorithm and sb3_class.lower() != algorithm:
                c.err(f"policy.sb3_class '{sb3_class}' does not match algorithm '{algorithm}'")
            m = _req(pol, "model", str, c, "policy")
            model = _resolve(base, m) if m else None
        elif fmt == "pgrl_run":
            r = _req(pol, "run_dir", str, c, "policy")
            run_dir = _resolve(base, r) if r else None
        if fmt:
            policy = PolicySpec(fmt, sb3_class, model, run_dir)

    # ── control ──
    ctl = raw.get("control")
    control_hz = warmup = max_steps = None
    warmup_actions = True
    if not isinstance(ctl, dict):
        c.err("root: missing required section 'control'")
    else:
        control_hz = _req(ctl, "control_hz", float, c, "control")
        warmup = _req(ctl, "warmup_steps", int, c, "control")
        max_steps = _req(ctl, "max_episode_steps", int, c, "control")
        if control_hz is not None and control_hz <= 0:
            c.err("control.control_hz must be > 0")
        if warmup is not None and warmup < 0:
            c.err("control.warmup_steps must be >= 0")
        if max_steps is not None and warmup is not None and max_steps <= warmup:
            c.err("control.max_episode_steps must be > warmup_steps")
        if "warmup_actions" in ctl:
            if isinstance(ctl["warmup_actions"], bool):
                warmup_actions = ctl["warmup_actions"]
            else:
                c.err("control.warmup_actions must be true or false")

    # ── joints ──
    jn = raw.get("joints")
    leg_order: Tuple[str, ...] = ()
    limits = home = None
    fixed: Dict[str, float] = {}
    if not isinstance(jn, dict):
        c.err("root: missing required section 'joints'")
    else:
        lo = jn.get("leg_order")
        if (not isinstance(lo, list) or len(lo) != 12 or len(set(lo)) != 12
                or set(lo) != set(NAO_LEG_JOINTS)):
            c.err(f"joints.leg_order must list each of the 12 NAO leg joints once: "
                  f"{list(NAO_LEG_JOINTS)}")
        else:
            leg_order = tuple(lo)
        if leg_order:
            lim = _joint_table(jn.get("limits_deg"), leg_order, "joints.limits_deg", c, pair=True)
            hp = _joint_table(jn.get("home_pose_deg"), leg_order, "joints.home_pose_deg", c, pair=False)
            if len(lim) == 12:
                limits = np.radians(np.array([lim[n] for n in leg_order], dtype=np.float64))
            if len(hp) == 12:
                home = np.radians(np.array([hp[n] for n in leg_order], dtype=np.float64))
            if limits is not None and home is not None:
                for i, n in enumerate(leg_order):
                    if not limits[i, 0] <= home[i] <= limits[i, 1]:
                        c.err(f"joints.home_pose_deg.{n} = {math.degrees(home[i]):.2f} "
                              f"is outside its limits")
        fp = jn.get("fixed_pose_deg")
        if not isinstance(fp, dict):
            c.err("joints: missing required table 'fixed_pose_deg'")
        else:
            for n, v in fp.items():
                if n not in NAO_NON_LEG_JOINTS:
                    c.err(f"joints.fixed_pose_deg.{n}: not a NAO non-leg joint "
                          f"(allowed: {list(NAO_NON_LEG_JOINTS)})")
                elif not isinstance(v, (int, float)) or isinstance(v, bool):
                    c.err(f"joints.fixed_pose_deg.{n}: expected a number (degrees)")
                else:
                    fixed[n] = math.radians(float(v))
            unset = [n for n in NAO_NON_LEG_JOINTS if n not in fp]
            if unset:
                c.warn(f"joints.fixed_pose_deg does not set {unset}; "
                       f"the robot side will leave them where they are")

    # ── reference gait ──
    gd = raw.get("reference_gait")
    gait = None
    if not isinstance(gd, dict):
        c.err("root: missing required section 'reference_gait'")
    else:
        kf = _req(gd, "keyframes", str, c, "reference_gait")
        ts = _req(gd, "timestamps", str, c, "reference_gait")
        nm = _req(gd, "joint_names", str, c, "reference_gait")
        after = _req(gd, "after_end", str, c, "reference_gait")
        if after is not None and after not in GAIT_AFTER_END:
            c.err(f"reference_gait.after_end '{after}' must be one of {GAIT_AFTER_END}")
            after = None
        period = gd.get("phase_period_s")
        if period is not None and (not isinstance(period, (int, float)) or period <= 0):
            c.err("reference_gait.phase_period_s must be a positive number or null")
            period = None
        if kf and ts and nm and after:
            files = [_resolve(base, kf), _resolve(base, ts), _resolve(base, nm)]
            missing = [str(f) for f in files if not f.is_file()]
            if missing:
                c.err(f"reference_gait: files not found: {missing}")
            else:
                g0 = GaitSpec(*files, after_end=after, phase_period_s=0.0)
                try:
                    q, t, names = _read_gait(g0)
                except Exception as e:  # corrupt .npy etc.
                    c.err(f"reference_gait: cannot read files: {e}")
                else:
                    ok = True
                    if q.ndim != 2 or q.shape[1] != len(names):
                        c.err(f"reference_gait: keyframes shape {q.shape} does not match "
                              f"{len(names)} joint names")
                        ok = False
                    if t.ndim != 1 or len(t) != q.shape[0]:
                        c.err(f"reference_gait: {len(t)} timestamps for {q.shape[0]} keyframes")
                        ok = False
                    elif len(t) < 2 or not np.all(np.diff(t) > 0):
                        c.err("reference_gait: timestamps must be strictly increasing")
                        ok = False
                    if leg_order and not set(leg_order) <= set(names):
                        c.err(f"reference_gait: joint names file lacks "
                              f"{sorted(set(leg_order) - set(names))}")
                        ok = False
                    if not np.all(np.isfinite(q)):
                        c.err("reference_gait: keyframes contain non-finite values")
                        ok = False
                    if ok:
                        # Default phase period: the clip length, as in WebotsReferenceGait.
                        gait = GaitSpec(*files, after_end=after,
                                        phase_period_s=float(period) if period else float(t[-1]))
                        if leg_order and home is not None:
                            cols = [names.index(n) for n in leg_order]
                            diff = np.degrees(np.abs(q[0, cols] - home)).max()
                            if diff > HOME_VS_GAIT_TOL_DEG:
                                c.err(f"joints.home_pose_deg differs from the gait's first "
                                      f"frame by {diff:.3f} deg (> {HOME_VS_GAIT_TOL_DEG}); "
                                      f"warm-up end would jump")
                            if limits is not None:
                                qq = q[:, cols]
                                out = (qq < limits[:, 0]) | (qq > limits[:, 1])
                                if out.any():
                                    bad = sorted({leg_order[j] for j in np.where(out.any(0))[0]})
                                    c.warn(f"reference_gait exceeds joint limits on {bad} "
                                           f"(clipped at run time, as in training)")

    # ── action / observation (sb3) or pgrl section ──
    action = observation = pgrl = None
    fmt = policy.format if policy else None
    if fmt == "sb3":
        ad = raw.get("action")
        if not isinstance(ad, dict):
            c.err("root: section 'action' is required for sb3 policies")
        else:
            at = _req(ad, "type", str, c, "action")
            if at is not None and at not in ACTION_TYPES:
                c.err(f"action.type '{at}' must be one of {ACTION_TYPES}")
            sc = _req(ad, "scale_rad", float, c, "action")
            if sc is not None and sc <= 0:
                c.err("action.scale_rad must be > 0")
            if at in ACTION_TYPES and sc and sc > 0:
                action = ActionSpec(at, sc)
        observation = _load_observation(raw.get("observation"), c)
        if "pgrl" in raw:
            c.err("section 'pgrl' is only valid for pgrl_run policies")
    elif fmt == "pgrl_run":
        pd = raw.get("pgrl")
        if not isinstance(pd, dict):
            c.err("root: section 'pgrl' is required for pgrl_run policies")
        else:
            ctrl = _req(pd, "controller", str, c, "pgrl")
            if ctrl is not None and ctrl not in PGRL_CONTROLLERS:
                c.err(f"pgrl.controller '{ctrl}' must be one of {PGRL_CONTROLLERS}")
            elif ctrl:
                pgrl = PGRLSpec(ctrl)
        for k in ("action", "observation"):
            if k in raw:
                c.err(f"section '{k}' is only valid for sb3 policies")

    # ── artefacts ──
    if policy is not None:
        needed = ([policy.model] if policy.format == "sb3"
                  else [policy.run_dir / "state.json", policy.run_dir / "config.json"]
                  if policy.run_dir else [])
        missing = [str(p) for p in needed if p is not None and not p.is_file()]
        if missing:
            msg = f"policy artefacts not found: {missing}"
            (c.err if require_artifacts else c.warn)(msg)

    if c.errors:
        raise DeploymentSpecError(path, c.errors)

    return DeploymentSpec(
        path=path, name=name, algorithm=algorithm, description=description.strip(),
        policy=policy, control_hz=control_hz, warmup_steps=warmup,
        max_episode_steps=max_steps, leg_order=leg_order, limits_rad=limits,
        home_pose_rad=home, fixed_pose_rad=fixed, gait=gait, action=action,
        observation=observation, pgrl=pgrl, warnings=tuple(c.warnings),
        warmup_actions=warmup_actions,
    )


def _load_observation(od: Any, c: _Collector) -> Optional[ObservationSpec]:
    if not isinstance(od, dict):
        c.err("root: section 'observation' is required for sb3 policies")
        return None
    qc = _req(od, "quaternion_convention", str, c, "observation")
    if qc is not None and qc not in QUAT_CONVENTIONS:
        c.err(f"observation.quaternion_convention '{qc}' must be one of {QUAT_CONVENTIONS}")
        qc = None
    yaw = _req(od, "spawn_yaw_deg", float, c, "observation")
    nz = od.get("nonfinite_to_zero", True)
    if not isinstance(nz, bool):
        c.err("observation.nonfinite_to_zero must be true/false")
    comps_raw = od.get("components")
    if not isinstance(comps_raw, list) or not comps_raw:
        c.err("observation.components must be a non-empty list")
        return None
    comps: List[ObsComponent] = []
    seen = set()
    for i, cd in enumerate(comps_raw):
        where = f"observation.components[{i}]"
        if not isinstance(cd, dict) or "name" not in cd:
            c.err(f"{where}: expected a mapping with a 'name'")
            continue
        nm = cd["name"]
        if nm not in OBS_COMPONENTS:
            c.err(f"{where}: unknown component '{nm}' (known: {list(OBS_COMPONENTS)})")
            continue
        if nm in seen:
            c.err(f"{where}: component '{nm}' listed twice")
            continue
        seen.add(nm)
        dim, allowed = OBS_COMPONENTS[nm]
        if cd.get("dim") != dim:
            c.err(f"{where} ({nm}): dim must be {dim}, got {cd.get('dim')}")
        opts = {k: v for k, v in cd.items() if k not in ("name", "dim")}
        for k in opts:
            if k not in allowed:
                c.err(f"{where} ({nm}): unknown option '{k}'")
        for k, rule in allowed.items():
            if k not in opts:
                c.err(f"{where} ({nm}): missing option '{k}'")
                continue
            v = opts[k]
            if isinstance(rule, tuple) and v not in rule:
                c.err(f"{where} ({nm}).{k}: '{v}' must be one of {rule}")
            elif rule is float and (not isinstance(v, (int, float)) or isinstance(v, bool)):
                c.err(f"{where} ({nm}).{k}: expected a number")
            elif rule is list and (not isinstance(v, list) or len(v) != dim):
                c.err(f"{where} ({nm}).{k}: expected a list of {dim} numbers")
        comps.append(ObsComponent(nm, dim, opts))
    if qc is None or yaw is None:
        return None
    return ObservationSpec(qc, yaw, bool(nz), tuple(comps))
