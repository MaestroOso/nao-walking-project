"""
nao_walking_env.py — Vectorized Multi-Robot NAO Walking Environment
====================================================================
test_13: Parallel robot training via SB3 VecEnv

KEY CHANGE: Multiple NAO robots train simultaneously in one Isaac Sim world.
- N robots spawned side-by-side, each with independent state
- One world.step() advances all robots in parallel
- Implements SB3's VecEnv interface directly (not gym.Env)
- PPO collects N experiences per simulation step → N× faster training

Everything else unchanged from test_12:
  - All 9 physics fixes (including finger stabilization FIX 7-9)
  - 45-value observation space (with linear velocity)
  - Survival-scaled alive bonus (base × (1 + step/100))
  - Same action space (12 leg joints, ±0.4 rad), same termination
  - Forward velocity reward and orientation penalty unchanged

USAGE:
  env = NaoVecEnv(num_envs=2, urdf_path="...", render_mode="headless")
  # env implements SB3 VecEnv — pass directly to PPO
  model = PPO("MlpPolicy", env)
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class NaoVecEnv(VecEnv):
    """
    Vectorized Isaac Sim NAO walking environment for Stable-Baselines3.

    Spawns `num_envs` NAO robots in a single Isaac Sim world, spaced apart.
    All robots share one physics step but have independent episode state.
    Implements SB3's VecEnv interface directly.
    """

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
        # ── NEW in test_11: wrists + hands at 0° ──
        "LWristYaw": 0, "RWristYaw": 0,
        "LHand": 0, "RHand": 0,
    }

    # ── NEW in test_11: finger/thumb joints all at 0° (closed) ──
    FINGER_JOINT_NAMES = [
        "LFinger11", "LFinger12", "LFinger13",
        "LFinger21", "LFinger22", "LFinger23",
        "LThumb1", "LThumb2",
        "RFinger11", "RFinger12", "RFinger13",
        "RFinger21", "RFinger22", "RFinger23",
        "RThumb1", "RThumb2",
    ]

    HAND_COLLISION_LINKS = [
        "l_wrist", "r_wrist", "l_gripper", "r_gripper",
        "LFinger11_link", "LFinger12_link", "LFinger13_link",
        "LFinger21_link", "LFinger22_link", "LFinger23_link",
        "LThumb1_link", "LThumb2_link",
        "RFinger11_link", "RFinger12_link", "RFinger13_link",
        "RFinger21_link", "RFinger22_link", "RFinger23_link",
        "RThumb1_link", "RThumb2_link",
    ]

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
        num_envs=2,
        urdf_path="../../robots/nao_isaac_sim/nao.urdf",
        render_mode="headless",
        physics_dt=1.0 / 120.0,
        control_dt=1.0 / 30.0,
        max_episode_steps=1500,
        spawn_height=0.34,
        robot_spacing=3.0,        # meters between robots
        kp=35810.0,
        kd=500.0,
        max_effort=100.0,
        action_scale=0.4,
        finger_kp=100.0,
        finger_kd=10.0,
        # Reward (survival-scaled from test_12)
        reward_forward_vel=5.0,
        reward_alive=1.0,
        penalty_orientation=0.5,
        alive_scale=100.0,
        # Termination
        term_height=0.10,
        term_orient_qw=0.3,
        grace_period=50,
    ):
        self._num_envs = num_envs
        self.urdf_path = urdf_path
        self.render_mode = render_mode
        self.physics_dt = physics_dt
        self.control_dt = control_dt
        self.decimation = int(control_dt / physics_dt)
        self.max_episode_steps = max_episode_steps
        self.spawn_height = spawn_height
        self.robot_spacing = robot_spacing
        self.kp = kp
        self.kd = kd
        self.max_effort = max_effort
        self.action_scale = action_scale
        self.finger_kp = finger_kp
        self.finger_kd = finger_kd

        # Reward weights
        self.w_forward = reward_forward_vel
        self.w_alive = reward_alive
        self.w_orient = penalty_orientation
        self.alive_scale = alive_scale

        # Termination
        self.term_height = term_height
        self.term_orient_qw = term_orient_qw
        self.grace_period = grace_period

        self.n_legs = len(self.LEG_JOINT_NAMES)

        # Observation and action spaces (per robot — same as test_12)
        self.n_obs = 3 + 3 + 2 + self.n_legs + self.n_legs + self.n_legs + 1  # = 45
        obs_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_obs,), dtype=np.float32,
        )
        act_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_legs,), dtype=np.float32,
        )

        # Initialize VecEnv base class
        super().__init__(num_envs, obs_space, act_space)

        # Per-robot state arrays
        self._step_counts = np.zeros(num_envs, dtype=np.int32)
        self._prev_actions = np.zeros((num_envs, self.n_legs), dtype=np.float32)
        self._prev_quats = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (num_envs, 1),
        )

        # Per-robot spawn positions (spread along Y axis so they don't collide)
        self._spawn_positions = np.zeros((num_envs, 3), dtype=np.float32)
        for i in range(num_envs):
            self._spawn_positions[i] = [0.0, i * self.robot_spacing, self.spawn_height]

        # Joint limits (shared — same robot model)
        self._leg_lower = np.zeros(self.n_legs, dtype=np.float32)
        self._leg_upper = np.zeros(self.n_legs, dtype=np.float32)
        self._leg_home = np.zeros(self.n_legs, dtype=np.float32)
        for i, name in enumerate(self.LEG_JOINT_NAMES):
            lo, hi = self.LEG_JOINT_LIMITS_DEG[name]
            self._leg_lower[i] = math.radians(lo)
            self._leg_upper[i] = math.radians(hi)
            self._leg_home[i] = math.radians(self.LEG_HOME_POSE_DEG[name])

        # Pending actions for step_async/step_wait pattern
        self._pending_actions = None

        # Sim state
        self._sim_initialized = False
        self._naos = []  # list of Articulation objects

    # ══════════════════════════════════════════════════════════════════════════
    # Simulation initialization — spawn N robots
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

        # ── Fix ground material (FIX 4) ──────────────────────────────────
        stage = self._world.stage
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.MaterialAPI):
                mat = UsdPhysics.MaterialAPI(prim)
                mat.GetRestitutionAttr().Set(0.0)
                mat.GetStaticFrictionAttr().Set(1.0)
                mat.GetDynamicFrictionAttr().Set(0.8)

        # ── Spawn N robots ────────────────────────────────────────────────
        finger_joint_names_set = set(
            self.FINGER_JOINT_NAMES +
            ["LWristYaw", "RWristYaw", "LHand", "RHand"]
        )

        config = urdf_module.ImportConfig()
        config.merge_fixed_joints = False
        config.fix_base = False
        config.import_inertia_tensor = True
        config.distance_scale = 1.0

        nao_prim_paths = []

        for env_idx in range(self._num_envs):
            print(f"[NaoVecEnv] Spawning robot {env_idx}...")

            result = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=self.urdf_path,
                import_config=config,
            )
            prim_path = result[1]
            nao_prim_paths.append(prim_path)

            root = stage.GetPrimAtPath(prim_path)

            # ── FIX 1: Disable instancing ────────────────────────────────
            for prim in Usd.PrimRange(root):
                if prim.IsInstanceable():
                    prim.SetInstanceable(False)

            # ── FIX 2: CollisionAPI on foot meshes ───────────────────────
            for col_path in [
                f"{prim_path}/l_ankle/collisions",
                f"{prim_path}/r_ankle/collisions",
                f"{prim_path}/LAnklePitch/collisions",
                f"{prim_path}/RAnklePitch/collisions",
            ]:
                col_prim = stage.GetPrimAtPath(col_path)
                if col_prim:
                    for child in Usd.PrimRange(col_prim):
                        if child.GetTypeName() == "Mesh":
                            if not child.HasAPI(UsdPhysics.CollisionAPI):
                                UsdPhysics.CollisionAPI.Apply(child)

            # ── FIX 3: Foot sole boxes ───────────────────────────────────
            for link, box in [("l_ankle", "l_sole_box"), ("r_ankle", "r_sole_box")]:
                bp = f"{prim_path}/{link}/collisions/{box}"
                box_prim = UsdGeom.Cube.Define(stage, bp)
                box_prim.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(box_prim.GetPrim())
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(0.02, 0.0, -0.04511))
                xf.AddScaleOp().Set(Gf.Vec3d(0.10, 0.06, 0.005))
                UsdPhysics.CollisionAPI.Apply(box_prim.GetPrim())

            # ── FIX 5: USD gains (standard + low for fingers FIX 8) ──────
            for prim in Usd.PrimRange(root):
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                    if drive:
                        joint_name = prim.GetName()
                        if joint_name in finger_joint_names_set:
                            use_kp = float(self.finger_kp)
                            use_kd = float(self.finger_kd)
                        else:
                            use_kp = float(self.kp)
                            use_kd = float(self.kd)

                        sa = drive.GetStiffnessAttr()
                        if sa:
                            sa.Set(use_kp)
                        else:
                            drive.CreateStiffnessAttr(use_kp)
                        da = drive.GetDampingAttr()
                        if da:
                            da.Set(use_kd)
                        else:
                            drive.CreateDampingAttr(use_kd)

            # Also handle continuous joints (fingers)
            for prim in Usd.PrimRange(root):
                if prim.GetTypeName() == "PhysicsJoint":
                    joint_name = prim.GetName()
                    if joint_name in finger_joint_names_set:
                        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                        if drive:
                            sa = drive.GetStiffnessAttr()
                            if sa:
                                sa.Set(float(self.finger_kp))
                            da = drive.GetDampingAttr()
                            if da:
                                da.Set(float(self.finger_kd))

            # ── FIX 6: Knee lower limit → +2° ───────────────────────────
            for prim in Usd.PrimRange(root):
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    if "KneePitch" in prim.GetName():
                        UsdPhysics.RevoluteJoint(prim).GetLowerLimitAttr().Set(2.0)

            # ── FIX 9: Disable collision on hand/finger meshes ───────────
            disabled_count = 0
            for link_name in self.HAND_COLLISION_LINKS:
                col_path = f"{prim_path}/{link_name}/collisions"
                col_prim = stage.GetPrimAtPath(col_path)
                if col_prim:
                    for child in Usd.PrimRange(col_prim):
                        if child.HasAPI(UsdPhysics.CollisionAPI):
                            col_api = UsdPhysics.CollisionAPI(child)
                            col_api.GetCollisionEnabledAttr().Set(False)
                            disabled_count += 1
                link_prim = stage.GetPrimAtPath(f"{prim_path}/{link_name}")
                if link_prim:
                    for child in Usd.PrimRange(link_prim):
                        if child.HasAPI(UsdPhysics.CollisionAPI) and child.GetTypeName() == "Mesh":
                            col_api = UsdPhysics.CollisionAPI(child)
                            col_api.GetCollisionEnabledAttr().Set(False)
                            disabled_count += 1

            print(f"[NaoVecEnv]   Robot {env_idx}: FIX 9 disabled {disabled_count} "
                  f"collision meshes")

        # ── Create Articulations and add to scene ─────────────────────────
        self._naos = []
        for env_idx, prim_path in enumerate(nao_prim_paths):
            nao = self._world.scene.add(
                Articulation(prim_path, name=f"nao_{env_idx}")
            )
            self._naos.append(nao)

        self._world.reset()

        # ── Post-reset setup for each robot ───────────────────────────────
        # Use robot 0 as reference for DOF names (all identical)
        self._dof_names = self._naos[0].dof_names
        self._num_dof = self._naos[0].num_dof

        # Build leg DOF index mapping (shared across all robots)
        self._leg_dof_indices = np.array([
            self._dof_names.index(name) for name in self.LEG_JOINT_NAMES
        ], dtype=np.int32)

        # Build full home pose (shared across all robots)
        self._full_home_pose = np.zeros(self._num_dof, dtype=np.float32)
        for name, deg in self.FIXED_JOINT_POSES_DEG.items():
            if name in self._dof_names:
                self._full_home_pose[self._dof_names.index(name)] = math.radians(deg)
        for i, idx in enumerate(self._leg_dof_indices):
            self._full_home_pose[idx] = self._leg_home[i]
        for name in self.FINGER_JOINT_NAMES:
            if name in self._dof_names:
                self._full_home_pose[self._dof_names.index(name)] = 0.0

        # Set per-joint gains on each robot (FIX 8 controller level)
        for env_idx, nao in enumerate(self._naos):
            controller = nao.get_articulation_controller()
            kps = np.full(self._num_dof, float(self.kp), dtype=np.float32)
            kds = np.full(self._num_dof, float(self.kd), dtype=np.float32)
            for i, name in enumerate(self._dof_names):
                if name in finger_joint_names_set:
                    kps[i] = float(self.finger_kp)
                    kds[i] = float(self.finger_kd)
            controller.set_gains(kps, kds)
            controller.set_max_efforts(
                np.full(self._num_dof, float(self.max_effort), dtype=np.float32)
            )

        # ── Detect velocity APIs (check on robot 0, all identical) ────────
        self._has_ang_vel_api = hasattr(self._naos[0], 'get_angular_velocity')
        if self._has_ang_vel_api:
            try:
                test_av = self._naos[0].get_angular_velocity()
                if test_av is None:
                    self._has_ang_vel_api = False
            except Exception:
                self._has_ang_vel_api = False

        self._has_lin_vel_api = hasattr(self._naos[0], 'get_linear_velocity')
        if self._has_lin_vel_api:
            try:
                test_lv = self._naos[0].get_linear_velocity()
                if test_lv is None:
                    self._has_lin_vel_api = False
            except Exception:
                self._has_lin_vel_api = False

        # ── Settle physics for all robots ─────────────────────────────────
        self._teleport_all_to_start()
        for _ in range(120):
            for nao in self._naos:
                nao.apply_action(
                    self._ArticulationAction(joint_positions=self._full_home_pose)
                )
            self._world.step(render=(self.render_mode == "human"))
        self._teleport_all_to_start()
        for _ in range(10):
            for nao in self._naos:
                nao.apply_action(
                    self._ArticulationAction(joint_positions=self._full_home_pose)
                )
            self._world.step(render=(self.render_mode == "human"))

        self._sim_initialized = True

        controlled_count = (len(self.FIXED_JOINT_POSES_DEG) +
                           len(self.LEG_JOINT_NAMES) +
                           len(self.FINGER_JOINT_NAMES))
        print(f"[NaoVecEnv] test_13 — Parallel training with {self._num_envs} robots")
        print(f"[NaoVecEnv] FIX 7: {controlled_count}/{self._num_dof} DOFs have "
              f"explicit home poses")
        print(f"[NaoVecEnv] Obs: {self.n_obs}, Reward: alive({self.w_alive}×scale) + "
              f"fwd({self.w_forward}×vx) - orient({self.w_orient}×tilt)")
        print(f"[NaoVecEnv] alive_scale={self.alive_scale} "
              f"(bonus doubles at step {int(self.alive_scale)})")
        print(f"[NaoVecEnv] grace={self.grace_period}, "
              f"max_steps={self.max_episode_steps}")
        print(f"[NaoVecEnv] Angular velocity API: {self._has_ang_vel_api}")
        print(f"[NaoVecEnv] Linear velocity API: {self._has_lin_vel_api}")

    # ══════════════════════════════════════════════════════════════════════════
    # Teleport helpers
    # ══════════════════════════════════════════════════════════════════════════
    def _teleport_all_to_start(self):
        """Teleport all robots to their spawn positions."""
        for env_idx, nao in enumerate(self._naos):
            nao.set_world_pose(
                position=self._spawn_positions[env_idx].copy(),
                orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            )
            nao.set_joint_positions(self._full_home_pose)
            nao.set_joint_velocities(np.zeros(self._num_dof))

    def _teleport_robot(self, env_idx):
        """Teleport a single robot to its spawn position."""
        nao = self._naos[env_idx]
        nao.set_world_pose(
            position=self._spawn_positions[env_idx].copy(),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        nao.set_joint_positions(self._full_home_pose)
        nao.set_joint_velocities(np.zeros(self._num_dof))

    def _settle_robot(self, env_idx):
        """Settle a single robot after teleport (brief stabilization)."""
        nao = self._naos[env_idx]
        for _ in range(10):
            nao.apply_action(
                self._ArticulationAction(joint_positions=self._full_home_pose)
            )
            # NOTE: We don't call world.step here — that happens in the main loop.
            # This just queues the action. The next world.step will apply it.
        # We can't step physics for just one robot — instead we rely on the
        # grace period to give the robot time to stabilize after reset.

    # ══════════════════════════════════════════════════════════════════════════
    # VecEnv interface: reset
    # ══════════════════════════════════════════════════════════════════════════
    def reset(self):
        """Reset all environments, return stacked observations."""
        if not self._sim_initialized:
            self._init_sim()

        self._teleport_all_to_start()

        # Brief settle with home pose commands
        for _ in range(10):
            for nao in self._naos:
                nao.apply_action(
                    self._ArticulationAction(joint_positions=self._full_home_pose)
                )
            self._world.step(render=(self.render_mode == "human"))

        # Reset per-robot state
        self._step_counts[:] = 0
        self._prev_actions[:] = 0.0
        self._prev_quats[:] = [0.0, 0.0, 0.0, 1.0]

        # Update prev_quats from actual poses
        for env_idx, nao in enumerate(self._naos):
            _, quat = nao.get_world_pose()
            if quat is not None and not np.any(np.isnan(quat)):
                self._prev_quats[env_idx] = quat.copy().astype(np.float32)

        # Collect observations
        obs = self._get_all_obs()
        return obs

    def _reset_single(self, env_idx):
        """Reset a single robot (called when an episode ends mid-rollout)."""
        self._teleport_robot(env_idx)
        self._step_counts[env_idx] = 0
        self._prev_actions[env_idx] = 0.0
        self._prev_quats[env_idx] = [0.0, 0.0, 0.0, 1.0]

        # Queue home pose action (will take effect on next world.step)
        self._naos[env_idx].apply_action(
            self._ArticulationAction(joint_positions=self._full_home_pose)
        )

    # ══════════════════════════════════════════════════════════════════════════
    # VecEnv interface: step_async / step_wait
    # ══════════════════════════════════════════════════════════════════════════
    def step_async(self, actions):
        """Store actions to be applied on next step_wait."""
        self._pending_actions = np.clip(actions, -1.0, 1.0).astype(np.float32)

    def step_wait(self):
        """
        Apply pending actions, step physics, return (obs, rewards, dones, infos).

        Returns match SB3 VecEnv contract:
          obs:     np.ndarray shape (num_envs, n_obs)
          rewards: np.ndarray shape (num_envs,)
          dones:   np.ndarray shape (num_envs,) — bool
          infos:   list of dicts, one per env
        """
        actions = self._pending_actions  # shape (num_envs, n_legs)

        # Increment step counts
        self._step_counts += 1

        # Build full joint targets for each robot and apply actions
        for env_idx in range(self._num_envs):
            action = actions[env_idx]
            leg_targets = self._leg_home + action * self.action_scale
            leg_targets = np.clip(leg_targets, self._leg_lower, self._leg_upper)

            full_target = self._full_home_pose.copy()
            for i, idx in enumerate(self._leg_dof_indices):
                full_target[idx] = leg_targets[i]

            self._naos[env_idx].apply_action(
                self._ArticulationAction(joint_positions=full_target)
            )

        # Step physics (all robots advance together)
        for _ in range(self.decimation):
            # Re-apply actions each sub-step for consistent PD control
            self._world.step(render=(self.render_mode == "human"))

        # Collect results
        obs = self._get_all_obs()
        rewards = self._compute_all_rewards(actions)
        dones = self._check_all_terminations()
        infos = [{} for _ in range(self._num_envs)]

        # Store prev_actions
        self._prev_actions[:] = actions

        # Update prev_quats
        for env_idx in range(self._num_envs):
            _, quat = self._naos[env_idx].get_world_pose()
            if quat is not None and not np.any(np.isnan(quat)):
                self._prev_quats[env_idx] = quat.copy().astype(np.float32)

        # Handle auto-reset for done environments
        # SB3 VecEnv contract: when done=True, the obs should be the NEW
        # episode's first observation (auto-reset), and the terminal obs
        # goes into info["terminal_observation"]
        for env_idx in range(self._num_envs):
            if dones[env_idx]:
                infos[env_idx]["terminal_observation"] = obs[env_idx].copy()
                # Also store episode stats for monitoring
                infos[env_idx]["episode"] = {
                    "r": float(rewards[env_idx]),  # will be overwritten by VecMonitor
                    "l": int(self._step_counts[env_idx]),
                }
                self._reset_single(env_idx)
                obs[env_idx] = self._get_obs(env_idx)

        return obs, rewards, dones, infos

    # ══════════════════════════════════════════════════════════════════════════
    # Observations
    # ══════════════════════════════════════════════════════════════════════════
    def _get_all_obs(self):
        """Get observations for all robots. Shape: (num_envs, n_obs)."""
        obs = np.zeros((self._num_envs, self.n_obs), dtype=np.float32)
        for env_idx in range(self._num_envs):
            obs[env_idx] = self._get_obs(env_idx)
        return obs

    def _get_obs(self, env_idx):
        """Get observation for a single robot. Shape: (n_obs,)."""
        nao = self._naos[env_idx]
        pos, quat = nao.get_world_pose()
        jp = nao.get_joint_positions()
        jv = nao.get_joint_velocities()

        if pos is None or np.any(np.isnan(pos)):
            return np.zeros(self.n_obs, dtype=np.float32)

        base_height = float(pos[2])
        proj_gravity = self._quat_rotate_inverse(quat, np.array([0, 0, -1.0]))
        ang_vel = self._get_angular_velocity(env_idx, quat)
        lin_vel = self._get_linear_velocity(env_idx)

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
            proj_gravity.astype(np.float32),           # [0:3]
            ang_vel,                                     # [3:6]
            lin_vel,                                     # [6:8]
            leg_pos_norm,                                # [8:20]
            leg_vel_scaled,                              # [20:32]
            self._prev_actions[env_idx],                 # [32:44]
            np.array([base_height], dtype=np.float32),   # [44]
        ])
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _get_linear_velocity(self, env_idx):
        if self._has_lin_vel_api:
            try:
                lin_vel_world = self._naos[env_idx].get_linear_velocity()
                if lin_vel_world is not None and not np.any(np.isnan(lin_vel_world)):
                    return np.array([
                        float(lin_vel_world[0]),
                        float(lin_vel_world[1]),
                    ], dtype=np.float32)
            except Exception:
                pass
        return np.zeros(2, dtype=np.float32)

    def _get_angular_velocity(self, env_idx, quat):
        if self._has_ang_vel_api:
            try:
                ang_vel_world = self._naos[env_idx].get_angular_velocity()
                if ang_vel_world is not None and not np.any(np.isnan(ang_vel_world)):
                    ang_vel_body = self._quat_rotate_inverse(quat, ang_vel_world)
                    return (ang_vel_body * 0.25).astype(np.float32)
            except Exception:
                pass

        if quat is not None and not np.any(np.isnan(quat)):
            px, py, pz, pw = self._prev_quats[env_idx]
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

    @staticmethod
    def _quat_rotate_inverse(quat, vec):
        x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        q = np.array([-x, -y, -z])
        t = 2.0 * np.cross(q, vec)
        result = vec + w * t + np.cross(q, t)
        return result.astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # Rewards (test_12 survival-scaled, applied per robot)
    # ══════════════════════════════════════════════════════════════════════════
    def _compute_all_rewards(self, actions):
        """Compute rewards for all robots. Shape: (num_envs,)."""
        rewards = np.zeros(self._num_envs, dtype=np.float32)
        for env_idx in range(self._num_envs):
            rewards[env_idx] = self._compute_reward(env_idx, actions[env_idx])
        return rewards

    def _compute_reward(self, env_idx, action):
        """Compute reward for a single robot."""
        nao = self._naos[env_idx]

        forward_vel = 0.0
        if self._has_lin_vel_api:
            try:
                lin_vel = nao.get_linear_velocity()
                if lin_vel is not None and not np.any(np.isnan(lin_vel)):
                    forward_vel = float(lin_vel[0])
            except Exception:
                pass

        r_forward = self.w_forward * forward_vel

        # Survival-scaled alive bonus (from test_12)
        scale = max(self.alive_scale, 1.0)
        r_alive = self.w_alive * (1.0 + self._step_counts[env_idx] / scale)

        _, quat = nao.get_world_pose()
        if quat is not None and not np.any(np.isnan(quat)):
            proj_grav = self._quat_rotate_inverse(quat, np.array([0, 0, -1.0]))
            orient_error = proj_grav[0]**2 + proj_grav[1]**2
        else:
            orient_error = 1.0
        r_orient = -self.w_orient * orient_error

        reward = r_forward + r_alive + r_orient
        return float(reward)

    # ══════════════════════════════════════════════════════════════════════════
    # Termination (same logic as test_12, applied per robot)
    # ══════════════════════════════════════════════════════════════════════════
    def _check_all_terminations(self):
        """Check termination for all robots. Shape: (num_envs,) bool."""
        dones = np.zeros(self._num_envs, dtype=bool)
        for env_idx in range(self._num_envs):
            dones[env_idx] = self._check_termination(env_idx)
        return dones

    def _check_termination(self, env_idx):
        """Check termination for a single robot."""
        nao = self._naos[env_idx]
        pos, quat = nao.get_world_pose()

        if pos is None or np.any(np.isnan(pos)):
            return True

        if self._step_counts[env_idx] <= self.grace_period:
            return False

        if self._step_counts[env_idx] >= self.max_episode_steps:
            return True

        if pos[2] < self.term_height:
            return True

        if quat is not None and not np.any(np.isnan(quat)):
            qw = abs(float(quat[3]))
            if qw < self.term_orient_qw:
                return True

        return False

    # ══════════════════════════════════════════════════════════════════════════
    # VecEnv required methods
    # ══════════════════════════════════════════════════════════════════════════
    def close(self):
        if self._sim_initialized:
            try:
                self._sim_app.close()
            except Exception:
                pass
            self._sim_initialized = False

    def seed(self, seed=None):
        """Set random seeds. VecEnv contract."""
        return [seed] * self._num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if envs are wrapped. We're not wrapping individual envs."""
        if indices is None:
            return [False] * self._num_envs
        return [False] * len(indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call a method on sub-environments. Limited support for VecEnv compat."""
        if indices is None:
            indices = range(self._num_envs)
        return [None] * len(list(indices))

    def get_attr(self, attr_name, indices=None):
        """Get attribute from sub-environments."""
        if indices is None:
            indices = range(self._num_envs)
        # Return our own attributes (all envs share the same config)
        val = getattr(self, attr_name, None)
        return [val] * len(list(indices))

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute on sub-environments."""
        setattr(self, attr_name, value)

    # ══════════════════════════════════════════════════════════════════════════
    # Helper: get base position for a robot (used by evaluate_policy)
    # ══════════════════════════════════════════════════════════════════════════
    def get_base_position(self, env_idx=0):
        """Get world position of a robot (for evaluation)."""
        pos, _ = self._naos[env_idx].get_world_pose()
        if pos is not None and not np.any(np.isnan(pos)):
            return pos.copy()
        return np.zeros(3)

    def get_spawn_position(self, env_idx=0):
        """Get spawn position of a robot (for distance calculation)."""
        return self._spawn_positions[env_idx].copy()
