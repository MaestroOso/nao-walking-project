"""
pgrl_env.py — PGRL walking environment (walking-solutions/pgrl/baseline_test28).

Subclasses `nao_walking_env.NaoWalkingEnv`, a verbatim copy (kept in this
folder so it is self-contained) of `walking-solutions/ppo/baseline_test24/
nao_walking_env.py`. Physics, PD tiers, spawn, warm-up (60 steps), grace
period (50 steps), fall criteria and the 1500-step limit are therefore
identical to PPO. If the PPO env changes, this copy does NOT follow.

What the subclass changes
-------------------------
1. Base gait: the looped Webots clip
   (`walking-solutions/shared/reference_gaits/forwards_looped_*`, built by
   `build_looped_gait.py`). Gait time wraps around the full looped duration,
   so any frequency factor f is safe.
2. Controller (replaces the neural network). Every control step:

       q*(t) = clip( q_home + a·(q_ref(f·t) − q_home) + Δq_comp , q_min, q_max )

   Δq_comp (same correction on BOTH legs):
       ankle pitch += K_ank_pitch · g_x     hip pitch += K_hip_pitch · g_x
       ankle roll  += K_ank_roll  · g_y     hip roll  += K_hip_roll  · g_y
   g_x, g_y = horizontal components of gravity projected into the torso
   frame (the IMU tilt). The compensator is active only after warm-up, so the
   warm-up is identical to PPO/SAC.
3. Stiffness: the PD gains (kp and kd) of each leg joint type are scaled by
   the factor s_j, re-applied after every world.reset().
4. step(action): `action` is IGNORED — the controller computes the targets
   from the parameters set with set_params(). This keeps the Gymnasium
   signature, so the common evaluation script can drive this env.

Quaternion convention: Isaac Sim returns (w, x, y, z). The baseline's
`_quat_rotate_inverse` reads (x, y, z, w) (found in experiments/test_26_sac,
see its APPROACH.md, "Quaternion convention fix"), which hides roll and turns
yaw into fake tilt. The tilt fed to the compensator is computed here with the
CORRECT (w, x, y, z) convention. Fall termination still uses the baseline
check (which is correct: |quat[3]| < 0.3 happens to test the right thing), so
falls count exactly as for PPO.
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHARED_DIR = HERE.parents[1] / "shared"                  # walking-solutions/shared
LOOPED_GAIT_DIR = SHARED_DIR / "reference_gaits"
LOOPED_GAIT_PREFIX = "forwards_looped_"
URDF_PATH = SHARED_DIR / "robot" / "nao.urdf"

sys.dont_write_bytecode = True
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from nao_walking_env import NaoWalkingEnv            # noqa: E402  (local copy)

import pgrl_params as P                              # noqa: E402

# Leg joint order (same as NaoWalkingEnv.LEG_JOINT_NAMES):
# 0 LHipYawPitch 1 LHipRoll 2 LHipPitch 3 LKneePitch 4 LAnklePitch 5 LAnkleRoll
# 6 RHipYawPitch 7 RHipRoll 8 RHipPitch 9 RKneePitch 10 RAnklePitch 11 RAnkleRoll
LEG_HIP_ROLL = [1, 7]
LEG_HIP_PITCH = [2, 8]
LEG_ANKLE_PITCH = [4, 10]
LEG_ANKLE_ROLL = [5, 11]


class LoopedGait:
    """Linear interpolation of the looped Webots clip, wrapping in time."""

    def __init__(self, data_dir=LOOPED_GAIT_DIR, joint_order=None,
                 prefix=LOOPED_GAIT_PREFIX):
        q = np.load(data_dir / f"{prefix}keyframes.npy")
        t = np.load(data_dir / f"{prefix}timestamps.npy")
        names = (data_dir / f"{prefix}joint_names.txt").read_text().split()
        cols = [names.index(n) for n in joint_order]
        self.q = q[:, cols].astype(np.float64)
        self.t = t.astype(np.float64)
        dt = float(np.median(np.diff(self.t)))
        # The file is N back-to-back copies of a periodic clip, so wrapping at
        # (last timestamp + dt) continues it seamlessly at any length.
        self.duration = float(self.t[-1]) + dt
        self.first_frame = self.q[0].copy()

    def targets(self, t):
        tq = float(t) % self.duration
        if tq > self.t[-1]:      # inside the last 40 ms: blend last -> first
            frac = (tq - self.t[-1]) / (self.duration - self.t[-1])
            return (1 - frac) * self.q[-1] + frac * self.q[0]
        return np.array([np.interp(tq, self.t, self.q[:, c])
                         for c in range(self.q.shape[1])])


class PGRLWalkingEnv(NaoWalkingEnv):

    def __init__(self, render_mode="headless", **kwargs):
        kwargs.setdefault("urdf_path", str(URDF_PATH))
        super().__init__(render_mode=render_mode, **kwargs)
        self._looped = LoopedGait(joint_order=self.LEG_JOINT_NAMES)
        diff = np.degrees(np.abs(self._looped.first_frame - self._leg_home)).max()
        print(f"[PGRLWalkingEnv] looped gait: {self._looped.duration:.2f}s, "
              f"first frame vs home pose max diff {diff:.3f} deg")
        self._phys = P.INIT_PHYS.copy()
        self.last_tilt = np.zeros(2)
        self.last_comp = np.zeros(self.n_legs)

    # ── parameters ──────────────────────────────────────────────────────────
    def set_params(self, theta=None, phys=None):
        """Set the policy from physical `phys` values (preferred) or from a
        normalised `theta` interpreted with the CURRENT ranges in
        pgrl_params.PARAM_SPECS. Callers working with a specific run must pass
        phys = to_physical(theta, that run's specs). Takes effect on the next
        reset() (stiffness is applied there)."""
        if phys is None:
            phys = P.to_physical(theta if theta is not None else P.initial_theta())
        self._phys = np.asarray(phys, dtype=np.float64).copy()

    # ── stiffness (overrides the baseline's three-tier gains) ──────────────
    def _apply_gains(self):
        high_gain_set = {
            "LHipYawPitch", "LHipRoll", "LHipPitch",
            "RHipYawPitch", "RHipRoll", "RHipPitch",
            "LKneePitch", "RKneePitch",
        }
        finger_set = set(self.FINGER_JOINT_NAMES +
                         ["LWristYaw", "RWristYaw", "LHand", "RHand"])
        stiff = self._phys[P.IDX_STIFF]                      # 6 factors
        kps = np.full(self._num_dof, float(self.kp), dtype=np.float32)
        kds = np.full(self._num_dof, float(self.kd), dtype=np.float32)
        for i, name in enumerate(self._dof_names):
            if name in finger_set:
                kps[i], kds[i] = float(self.finger_kp), float(self.finger_kd)
            elif name in high_gain_set:
                kps[i], kds[i] = float(self.knee_kp), float(self.knee_kd)
            if name in self.LEG_JOINT_NAMES:                 # leg joint -> scale
                s = float(stiff[self.LEG_JOINT_NAMES.index(name) % 6])
                kps[i] *= s
                kds[i] *= s
        self._controller.set_gains(kps, kds)
        self._controller.set_max_efforts(
            np.full(self._num_dof, float(self.max_effort), dtype=np.float32))

    # ── tilt (correct w,x,y,z convention) ──────────────────────────────────
    @staticmethod
    def projected_gravity_wxyz(quat):
        w, x, y, z = (float(e) for e in quat)
        q = np.array([-x, -y, -z])
        v = np.array([0.0, 0.0, -1.0])
        t = 2.0 * np.cross(q, v)
        return v + w * t + np.cross(q, t)

    def _tilt(self):
        _, quat = self._nao.get_world_pose()
        if quat is None or not np.all(np.isfinite(quat)):
            return np.zeros(2)
        g = self.projected_gravity_wxyz(quat)
        return np.array([g[0], g[1]])

    # ── controller ─────────────────────────────────────────────────────────
    def compute_leg_targets(self):
        """Joint targets for the current control step (before stepping)."""
        K_ap, K_hp, K_ar, K_hr = self._phys[P.IDX_GAINS]
        f, a = self._phys[P.IDX_F], self._phys[P.IDX_A]
        comp = np.zeros(self.n_legs)
        if self._step_count < self.gait_warmup_steps:
            q = self._leg_home.astype(np.float64).copy()
            self._phase = 0.0
        else:
            t = (self._step_count - self.gait_warmup_steps) * self.control_dt
            q_ref = self._looped.targets(f * t)
            q = self._leg_home + a * (q_ref - self._leg_home)
            self._phase = (f * t % 2.64) / 2.64          # bookkeeping only
            g_x, g_y = self._tilt()
            self.last_tilt = np.array([g_x, g_y])
            comp[LEG_ANKLE_PITCH] = K_ap * g_x
            comp[LEG_HIP_PITCH] = K_hp * g_x
            comp[LEG_ANKLE_ROLL] = K_ar * g_y
            comp[LEG_HIP_ROLL] = K_hr * g_y
        self.last_comp = comp
        return np.clip(q + comp, self._leg_lower, self._leg_upper).astype(np.float32)

    def step(self, action=None):
        """Gymnasium step; `action` is ignored (see module docstring)."""
        self._step_count += 1
        leg_targets = self.compute_leg_targets()
        full_target = self._full_home_pose.copy()
        for i, idx in enumerate(self._leg_dof_indices):
            full_target[idx] = leg_targets[i]
        for _ in range(self.decimation):             # same loop as baseline
            self._nao.apply_action(self._ArticulationAction(joint_positions=full_target))
            self._world.step(render=(self.render_mode == "human"))
        obs = self._get_obs()                        # unused by PGRL
        terminated = self._check_termination()       # baseline fall criteria
        truncated = self._step_count >= self.max_episode_steps
        return obs, 0.0, terminated, truncated, {}
