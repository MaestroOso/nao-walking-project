"""
nao_walking_env.py — Gymnasium RL Environment for NAO Walking
==============================================================
test_10: Linear velocity observation + simplified reward

Key changes:
  1. NEW OBSERVATION — Linear velocity (from get_linear_velocity() API):
     - vel_x (forward), vel_y (lateral) added to observation space
     - Agent can now DIRECTLY observe the quantity it's being rewarded for
     - Observation space: 43 → 45 values
  2. SIMPLIFIED REWARD — Only 3 components:
     - alive_bonus: +1.0 (stay upright)
     - forward_vel: +5.0 × vel_x (move forward)
     - orientation: -0.5 × tilt (don't fall)
     - REMOVED: energy, smoothness, joint_limit, vel_target
     - Clean signal — agent learns what matters without noise
  3. TRAIN FROM SCRATCH:
     - New observation space means no model resume
     - Clean start with better information

Observation space (45 values):
  [0:3]   Projected gravity vector (body frame) — balance signal
  [3:6]   Base angular velocity (body frame) — rotation rate
  [6:8]   Base linear velocity (vel_x, vel_y) — NEW: forward & lateral speed
  [8:20]  Leg joint positions (12 joints, normalized)
  [20:32] Leg joint velocities (12 joints, scaled)
  [32:44] Previous actions (12 values)
  [44]    Base height

Action space (12 values — unchanged):
  Position deltas for 12 leg joints, ±0.4 rad from home pose

Reward:
  r = alive_bonus(+1.0) + forward_vel(+5.0 × vx) - orientation(0.5 × tilt)
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NaoWalkingEnv(gym.Env):
    """Isaac Sim NAO walking environment for Stable-Baselines3."""

    metadata = {"render_modes": ["human", "headless"]}

    LEG_JOINT_NAMES = [
        "LHipYawPitch", "LHipRoll", "LHipPitch",
        "LKneePitch", "LAnklePitch", "LAnkleRoll",
        "RHipYawPitch", "RHipRoll", "RHipPitch",
        "RKneePitch", "RAnklePitch", "RAnkleRoll",
    ]

    FIXED_JOINT_POSES_DEG = {
        "LShoulderPitch": 84, "RShoulderPitch": 84,
        "LShoulderRoll": 10, "RShoulderRoll": -10,
        "LElbowYaw": -70, "RElbowYaw": 70,
        "LElbowRoll": -25, "RElbowRoll": 25,
        "HeadYaw": 0, "HeadPitch": 0,
    }

    LEG_HOME_POSE_DEG = {
        "LHipYawPitch": 0, "LHipRoll": 0, "LHipPitch": -2,
        "LKneePitch": 4, "LAnklePitch": 2, "LAnkleRoll": 0,
        "RHipYawPitch": 0, "RHipRoll": 0, "RHipPitch": -2,
        "RKneePitch": 4, "RAnklePitch": 2, "RAnkleRoll": 0,
    }

    LEG_JOINT_LIMITS_DEG = {
        "LHipYawPitch": (-65.6, 42.4), "LHipRoll": (-21.7, 45.3),
        "LHipPitch": (-88.0, 27.7), "LKneePitch": (2.0, 121.0),
        "LAnklePitch": (-68.1, 52.9), "LAnkleRoll": (-22.8, 44.1),
        "RHipYawPitch": (-65.6, 42.4), "RHipRoll": (-45.3, 21.7),
        "RHipPitch": (-88.0, 27.7), "RKneePitch": (2.0, 121.0),
        "RAnklePitch": (-68.0, 53.4), "RAnkleRoll": (-44.1, 22.8),
    }

    def __init__(
        self,
        urdf_path="../../robots/nao_isaac_sim/nao.urdf",
        render_mode="headless",
        physics_dt=1.0 / 120.0,
        control_dt=1.0 / 30.0,
        max_episode_steps=1500,
        spawn_height=0.34,
        kp=35810.0,
        kd=500.0,
        max_effort=100.0,
        action_scale=0.4,
        # ── Simplified reward — only 3 components ──
        reward_forward_vel=5.0,
        reward_alive=1.0,
        penalty_orientation=0.5,
        # Termination
        term_height=0.10,
        term_orient_qw=0.3,
        grace_period=50,
    ):
        super().__init__()

        self.urdf_path = urdf_path
        self.render_mode = render_mode
        self.physics_dt = physics_dt
        self.control_dt = control_dt
        self.decimation = int(control_dt / physics_dt)
        self.max_episode_steps = max_episode_steps
        self.spawn_height = spawn_height
        self.kp = kp
        self.kd = kd
        self.max_effort = max_effort
        self.action_scale = action_scale

        # Reward weights — SIMPLIFIED
        self.w_forward = reward_forward_vel
        self.w_alive = reward_alive
        self.w_orient = penalty_orientation

        # Termination
        self.term_height = term_height
        self.term_orient_qw = term_orient_qw
        self.grace_period = grace_period

        self.n_legs = len(self.LEG_JOINT_NAMES)

        # ── Spaces — NEW: 45 observations (was 43) ───────────────────────────
        # Added: linear velocity (vel_x, vel_y) = 2 extra values
        self.n_obs = 3 + 3 + 2 + self.n_legs + self.n_legs + self.n_legs + 1  # 45
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_obs,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_legs,), dtype=np.float32,
        )

        # ── Internal state ────────────────────────────────────────────────────
        self._sim_initialized = False
        self._prev_action = np.zeros(self.n_legs, dtype=np.float32)
        self._step_count = 0

        self._leg_lower = np.zeros(self.n_legs, dtype=np.float32)
        self._leg_upper = np.zeros(self.n_legs, dtype=np.float32)
        self._leg_home = np.zeros(self.n_legs, dtype=np.float32)

        for i, name in enumerate(self.LEG_JOINT_NAMES):
            lo, hi = self.LEG_JOINT_LIMITS_DEG[name]
            self._leg_lower[i] = math.radians(lo)
            self._leg_upper[i] = math.radians(hi)
            self._leg_home[i] = math.radians(self.LEG_HOME_POSE_DEG[name])

    # ══════════════════════════════════════════════════════════════════════════
    # Simulation initialization
    # ══════════════════════════════════════════════════════════════════════════
    def _init_sim(self):
        from isaacsim import SimulationApp

        headless = self.render_mode == "headless"
        self._sim_app = SimulationApp({
            "headless": headless,
            "width": 1280,
            "height": 720,
        })

        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.utils.types import ArticulationAction
        import omni.kit.commands
        import isaacsim.asset.importer.urdf._urdf as urdf_module
        from pxr import UsdPhysics, UsdGeom, Usd, Gf

        self._Articulation = Articulation
        self._ArticulationAction = ArticulationAction
        self._UsdPhysics = UsdPhysics
        self._Usd = Usd

        self._world = World(
            physics_dt=self.physics_dt,
            rendering_dt=1.0 / 60.0,
        )
        self._world.scene.add_default_ground_plane()

        config = urdf_module.ImportConfig()
        config.merge_fixed_joints = False
        config.fix_base = False
        config.import_inertia_tensor = True
        config.distance_scale = 1.0

        result = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=self.urdf_path,
            import_config=config,
        )
        self._nao_prim_path = result[1]
        stage = self._world.stage
        root = stage.GetPrimAtPath(self._nao_prim_path)

        # ── FIX 1: Disable instancing ────────────────────────────────────────
        for prim in Usd.PrimRange(root):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)

        # ── FIX 2: CollisionAPI on foot meshes ───────────────────────────────
        for col_path in [
            f"{self._nao_prim_path}/l_ankle/collisions",
            f"{self._nao_prim_path}/r_ankle/collisions",
            f"{self._nao_prim_path}/LAnklePitch/collisions",
            f"{self._nao_prim_path}/RAnklePitch/collisions",
        ]:
            col_prim = stage.GetPrimAtPath(col_path)
            if col_prim:
                for child in Usd.PrimRange(col_prim):
                    if child.GetTypeName() == "Mesh":
                        if not child.HasAPI(UsdPhysics.CollisionAPI):
                            UsdPhysics.CollisionAPI.Apply(child)

        # ── FIX 3: Foot sole boxes ──────────────────────────────────────────
        for link, box in [("l_ankle", "l_sole_box"), ("r_ankle", "r_sole_box")]:
            bp = f"{self._nao_prim_path}/{link}/collisions/{box}"
            box_prim = UsdGeom.Cube.Define(stage, bp)
            box_prim.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(box_prim.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(0.02, 0.0, -0.04511))
            xf.AddScaleOp().Set(Gf.Vec3d(0.10, 0.06, 0.005))
            UsdPhysics.CollisionAPI.Apply(box_prim.GetPrim())

        # ── FIX 4: Ground material ──────────────────────────────────────────
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.MaterialAPI):
                mat = UsdPhysics.MaterialAPI(prim)
                mat.GetRestitutionAttr().Set(0.0)
                mat.GetStaticFrictionAttr().Set(1.0)
                mat.GetDynamicFrictionAttr().Set(0.8)

        # ── FIX 5: USD gains before Articulation ────────────────────────────
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.RevoluteJoint):
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                if drive:
                    sa = drive.GetStiffnessAttr()
                    if sa:
                        sa.Set(float(self.kp))
                    else:
                        drive.CreateStiffnessAttr(float(self.kp))
                    da = drive.GetDampingAttr()
                    if da:
                        da.Set(float(self.kd))
                    else:
                        drive.CreateDampingAttr(float(self.kd))

        # ── FIX 6: Knee lower limit → +2° ──────────────────────────────────
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.RevoluteJoint):
                if "KneePitch" in prim.GetName():
                    UsdPhysics.RevoluteJoint(prim).GetLowerLimitAttr().Set(2.0)

        # ── Create Articulation ─────────────────────────────────────────────
        self._nao = self._world.scene.add(
            Articulation(self._nao_prim_path, name="nao")
        )
        self._world.reset()

        self._dof_names = self._nao.dof_names
        self._num_dof = self._nao.num_dof
        self._controller = self._nao.get_articulation_controller()

        kps = np.full(self._num_dof, float(self.kp), dtype=np.float32)
        kds = np.full(self._num_dof, float(self.kd), dtype=np.float32)
        self._controller.set_gains(kps, kds)
        self._controller.set_max_efforts(
            np.full(self._num_dof, float(self.max_effort), dtype=np.float32)
        )

        self._leg_dof_indices = np.array([
            self._dof_names.index(name) for name in self.LEG_JOINT_NAMES
        ], dtype=np.int32)

        self._full_home_pose = np.zeros(self._num_dof, dtype=np.float32)
        for name, deg in self.FIXED_JOINT_POSES_DEG.items():
            if name in self._dof_names:
                self._full_home_pose[self._dof_names.index(name)] = math.radians(deg)
        for i, idx in enumerate(self._leg_dof_indices):
            self._full_home_pose[idx] = self._leg_home[i]

        # ── Angular velocity API ─────────────────────────────────────────────
        self._has_ang_vel_api = hasattr(self._nao, 'get_angular_velocity')
        if self._has_ang_vel_api:
            try:
                test_av = self._nao.get_angular_velocity()
                if test_av is None:
                    self._has_ang_vel_api = False
            except Exception:
                self._has_ang_vel_api = False

        # ── Linear velocity API (NEW — verified in test_09) ──────────────────
        self._has_lin_vel_api = hasattr(self._nao, 'get_linear_velocity')
        if self._has_lin_vel_api:
            try:
                test_lv = self._nao.get_linear_velocity()
                if test_lv is None:
                    self._has_lin_vel_api = False
            except Exception:
                self._has_lin_vel_api = False

        print(f"[NaoWalkingEnv] Angular velocity API: {self._has_ang_vel_api}")
        print(f"[NaoWalkingEnv] Linear velocity API:  {self._has_lin_vel_api}")

        self._prev_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        self._settle_physics()

        self._sim_initialized = True
        print(f"[NaoWalkingEnv] test_10 — Velocity obs + simplified reward")
        print(f"[NaoWalkingEnv] Observation space: {self.n_obs} values "
              f"(was 43, added vel_x/vel_y)")
        print(f"[NaoWalkingEnv] Reward: alive({self.w_alive}) + "
              f"forward({self.w_forward}×vx) - orient({self.w_orient}×tilt)")
        print(f"[NaoWalkingEnv] action_scale={self.action_scale} rad "
              f"({math.degrees(self.action_scale):.1f}°)")
        print(f"[NaoWalkingEnv] grace={self.grace_period}, "
              f"max_steps={self.max_episode_steps}")

    def _settle_physics(self):
        self._teleport_to_start()
        for _ in range(120):
            self._nao.apply_action(
                self._ArticulationAction(joint_positions=self._full_home_pose)
            )
            self._world.step(render=(self.render_mode == "human"))
        self._teleport_to_start()
        for _ in range(10):
            self._nao.apply_action(
                self._ArticulationAction(joint_positions=self._full_home_pose)
            )
            self._world.step(render=(self.render_mode == "human"))

    def _teleport_to_start(self):
        self._nao.set_world_pose(
            position=np.array([0.0, 0.0, self.spawn_height]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        self._nao.set_joint_positions(self._full_home_pose)
        self._nao.set_joint_velocities(np.zeros(self._num_dof))

    # ══════════════════════════════════════════════════════════════════════════
    # Gymnasium interface
    # ══════════════════════════════════════════════════════════════════════════
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self._sim_initialized:
            self._init_sim()

        self._teleport_to_start()
        for _ in range(10):
            self._nao.apply_action(
                self._ArticulationAction(joint_positions=self._full_home_pose)
            )
            self._world.step(render=(self.render_mode == "human"))

        self._prev_action = np.zeros(self.n_legs, dtype=np.float32)
        self._step_count = 0

        _, quat = self._nao.get_world_pose()
        if quat is not None and not np.any(np.isnan(quat)):
            self._prev_quat = quat.copy().astype(np.float32)
        else:
            self._prev_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        leg_targets = self._leg_home + action * self.action_scale
        leg_targets = np.clip(leg_targets, self._leg_lower, self._leg_upper)

        full_target = self._full_home_pose.copy()
        for i, idx in enumerate(self._leg_dof_indices):
            full_target[idx] = leg_targets[i]

        for _ in range(self.decimation):
            self._nao.apply_action(
                self._ArticulationAction(joint_positions=full_target)
            )
            self._world.step(render=(self.render_mode == "human"))

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_termination()
        truncated = self._step_count >= self.max_episode_steps

        self._prev_action = action.copy()

        _, quat = self._nao.get_world_pose()
        if quat is not None and not np.any(np.isnan(quat)):
            self._prev_quat = quat.copy().astype(np.float32)

        return obs, reward, terminated, truncated, {}

    # ══════════════════════════════════════════════════════════════════════════
    # Observations — NEW: includes linear velocity
    # ══════════════════════════════════════════════════════════════════════════
    def _get_obs(self):
        """Build observation vector (45 values).

        New layout:
          [0:3]   Projected gravity (body frame)
          [3:6]   Angular velocity (body frame, scaled ×0.25)
          [6:8]   Linear velocity (vel_x, vel_y in world frame, scaled ×1.0)
          [8:20]  Leg joint positions (normalized [-1, 1])
          [20:32] Leg joint velocities (scaled ×0.1)
          [32:44] Previous actions
          [44]    Base height
        """
        pos, quat = self._nao.get_world_pose()
        jp = self._nao.get_joint_positions()
        jv = self._nao.get_joint_velocities()

        if pos is None or np.any(np.isnan(pos)):
            return np.zeros(self.n_obs, dtype=np.float32)

        base_height = float(pos[2])
        proj_gravity = self._quat_rotate_inverse(quat, np.array([0, 0, -1.0]))
        ang_vel = self._get_angular_velocity(quat)

        # ── Linear velocity (NEW in test_10) ─────────────────────────────────
        lin_vel = self._get_linear_velocity()

        # Leg joints
        leg_pos = np.zeros(self.n_legs, dtype=np.float32)
        leg_vel = np.zeros(self.n_legs, dtype=np.float32)
        if jp is not None:
            for i, idx in enumerate(self._leg_dof_indices):
                leg_pos[i] = jp[idx]
                if jv is not None:
                    leg_vel[i] = jv[idx]

        joint_range = self._leg_upper - self._leg_lower
        joint_mid = (self._leg_upper + self._leg_lower) / 2.0
        leg_pos_norm = np.where(
            joint_range > 0,
            2.0 * (leg_pos - joint_mid) / joint_range,
            0.0,
        ).astype(np.float32)
        leg_vel_scaled = (leg_vel * 0.1).astype(np.float32)

        obs = np.concatenate([
            proj_gravity.astype(np.float32),     # [0:3]
            ang_vel,                              # [3:6]
            lin_vel,                              # [6:8]  — NEW
            leg_pos_norm,                         # [8:20]
            leg_vel_scaled,                       # [20:32]
            self._prev_action,                    # [32:44]
            np.array([base_height], dtype=np.float32),  # [44]
        ])

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _get_linear_velocity(self):
        """Get base linear velocity (vel_x, vel_y) in world frame.

        Returns 2 values: forward (x) and lateral (y) velocity.
        Scaling: raw m/s — typical walking is 0.1-0.3 m/s, already in good range.
        """
        if self._has_lin_vel_api:
            try:
                lin_vel_world = self._nao.get_linear_velocity()
                if lin_vel_world is not None and not np.any(np.isnan(lin_vel_world)):
                    # Return only x and y (forward and lateral)
                    return np.array([
                        float(lin_vel_world[0]),
                        float(lin_vel_world[1]),
                    ], dtype=np.float32)
            except Exception:
                pass
        return np.zeros(2, dtype=np.float32)

    def _get_angular_velocity(self, quat):
        if self._has_ang_vel_api:
            try:
                ang_vel_world = self._nao.get_angular_velocity()
                if ang_vel_world is not None and not np.any(np.isnan(ang_vel_world)):
                    ang_vel_body = self._quat_rotate_inverse(quat, ang_vel_world)
                    return (ang_vel_body * 0.25).astype(np.float32)
            except Exception:
                pass

        if quat is not None and not np.any(np.isnan(quat)):
            px, py, pz, pw = self._prev_quat
            cx, cy, cz, cw = quat
            dw = cw*pw + cx*px + cy*py + cz*pz
            dx = -cw*px + cx*pw - cy*pz + cz*py
            dy = -cw*py + cx*pz + cy*pw - cz*px
            dz = -cw*pz - cx*py + cy*px + cz*pw
            if dw < 0:
                dx, dy, dz, dw = -dx, -dy, -dz, -dw
            ang_vel = 2.0 * np.array([dx, dy, dz], dtype=np.float32) / self.control_dt
            return (ang_vel * 0.25).astype(np.float32)

        return np.zeros(3, dtype=np.float32)

    def _get_base_position(self):
        pos, _ = self._nao.get_world_pose()
        if pos is not None and not np.any(np.isnan(pos)):
            return pos.copy()
        return np.zeros(3)

    @staticmethod
    def _quat_rotate_inverse(quat, vec):
        x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        q = np.array([-x, -y, -z])
        t = 2.0 * np.cross(q, vec)
        result = vec + w * t + np.cross(q, t)
        return result.astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # Reward — SIMPLIFIED: 3 components only
    # ══════════════════════════════════════════════════════════════════════════
    def _compute_reward(self, action):
        """Simple, clean reward: stay upright + move forward + don't tilt.

        r = alive_bonus + forward_vel_reward - orientation_penalty

        Standing still: ~1.0/step (alive only)
        Walking at 0.1 m/s: ~1.5/step (alive + 0.5 forward)
        Walking at 0.2 m/s: ~2.0/step (alive + 1.0 forward)
        """
        # Forward velocity — use API directly (more accurate than finite diff)
        forward_vel = 0.0
        if self._has_lin_vel_api:
            try:
                lin_vel = self._nao.get_linear_velocity()
                if lin_vel is not None and not np.any(np.isnan(lin_vel)):
                    forward_vel = float(lin_vel[0])
            except Exception:
                pass

        r_forward = self.w_forward * forward_vel
        r_alive = self.w_alive

        # Orientation penalty
        _, quat = self._nao.get_world_pose()
        if quat is not None and not np.any(np.isnan(quat)):
            proj_grav = self._quat_rotate_inverse(quat, np.array([0, 0, -1.0]))
            orient_error = proj_grav[0]**2 + proj_grav[1]**2
        else:
            orient_error = 1.0
        r_orient = -self.w_orient * orient_error

        reward = r_forward + r_alive + r_orient
        return float(reward)

    # ══════════════════════════════════════════════════════════════════════════
    # Termination
    # ══════════════════════════════════════════════════════════════════════════
    def _check_termination(self):
        pos, quat = self._nao.get_world_pose()

        if pos is None or np.any(np.isnan(pos)):
            return True

        if self._step_count <= self.grace_period:
            return False

        if pos[2] < self.term_height:
            return True

        if quat is not None and not np.any(np.isnan(quat)):
            qw = abs(float(quat[3]))
            if qw < self.term_orient_qw:
                return True

        return False

    # ══════════════════════════════════════════════════════════════════════════
    # Cleanup
    # ══════════════════════════════════════════════════════════════════════════
    def close(self):
        if self._sim_initialized:
            try:
                self._sim_app.close()
            except Exception:
                pass
            self._sim_initialized = False
