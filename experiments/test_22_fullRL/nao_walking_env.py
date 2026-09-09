"""
nao_walking_env.py — Gymnasium RL Environment for NAO Walking
==============================================================
test_15: Reference gait + residual actions (BUILT ON test16_rl BASE)

CHANGES FROM test16_rl:
  - Crouched home pose (hip=-20, knee=40, ankle=-20)
  - True alternating gait: only ONE leg moves at a time
  - Stance leg knee uses high PD gains (kp=300k) same as hips
  - Stance ankle push-off: keeps CoM over support foot during swing
  - Reduced hip swing amplitude (12deg vs 25deg) to prevent forward lunge
  - Swing leg lifts knee (foot clears ground) then swings forward

PHYSICS FIXES (all mandatory, from test_01–test16_rl):
  FIX 1: Disable instancing
  FIX 2: CollisionAPI on foot meshes
  FIX 3: Foot sole boxes
  FIX 4: Ground material (no bouncing)
  FIX 5: USD DriveAPI gains before Articulation
  FIX 6: Knee lower limit +2deg
  FIX 7: Finger joints at 0deg
  FIX 8: Low gains for finger/wrist joints
  FIX 9: Disable collision on hand/finger meshes
  + spawn_height=0.3344 (measured, not guessed)
  + world.reset() in every reset() call
  + _apply_gains() after every world.reset()
  + hip_kp=300k, knee_kp=300k (load-bearing joints)
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from webots_reference_gait import WebotsReferenceGait


# ══════════════════════════════════════════════════════════════════════════
# Reference Gait
# ══════════════════════════════════════════════════════════════════════════
class ReferenceGait:
    """
    True alternating gait with ankle push-off.

    Phase 0.0-0.5: right leg swings forward, left leg is stance
    Phase 0.5-1.0: left leg swings forward, right leg is stance

    The stance leg holds position with ankle push-off to keep CoM
    over the support foot. The swing leg lifts its knee (foot clears
    ground) then swings the hip forward.
    """

    def __init__(self, gait_hz=0.2, lateral_only=False):
        self.gait_hz      = gait_hz
        self.lateral_only = lateral_only

        # Base pose — matches crouched home pose
        self.hip_pitch_base  = math.radians(-20.0)
        self.knee_pitch_base = math.radians(40.0)
        self.ankle_pitch_base = math.radians(-20.0)

        # Swing leg: hip forward arc (reduced — 25deg caused forward lunge)
        self.hip_pitch_amp   = math.radians(10.0)
        # Swing leg: knee lift (foot clears ground)
        self.knee_pitch_amp  = math.radians(10.0)

        # Stance leg: ankle push-off (plantarflexion keeps CoM over support foot)
        # Peaks at mid-stance, zero at start/end of stance
        self.ankle_pushoff_amp = math.radians(3.0)

        # Hip roll: lateral weight shift toward stance leg
        self.hip_roll_amp = math.radians(6.0)

    def get_phase(self, t):
        return (t * self.gait_hz) % 1.0

    def get_phase_signal(self, phase):
        angle = 2.0 * math.pi * phase
        return math.sin(angle), math.cos(angle)

    def _swing_targets(self, phi_swing):
        t = phi_swing / 0.5
        hip_pitch   = self.hip_pitch_base - self.hip_pitch_amp * math.sin(math.pi * t)
        knee_pitch  = self.knee_pitch_base + self.knee_pitch_amp * math.sin(math.pi * t)
        ankle_pitch = self.ankle_pitch_base
        return hip_pitch, knee_pitch, ankle_pitch

    def _stance_targets(self, phi_stance):
        t = phi_stance / 0.5
        hip_pitch   = self.hip_pitch_base
        knee_pitch  = self.knee_pitch_base
        ankle_push  = self.ankle_pushoff_amp * math.sin(math.pi * t)
        ankle_pitch = self.ankle_pitch_base - ankle_push
        return hip_pitch, knee_pitch, ankle_pitch

    def get_targets(self, phase):
        """
        Return target joint angles for 12 leg joints (LEG_JOINT_NAMES order).

        phase in [0, 1):
          0.0-0.5: right leg swings, left leg is stance
          0.5-1.0: left leg swings, right leg is stance

        Order: [LHipYawPitch, LHipRoll, LHipPitch, LKneePitch, LAnklePitch, LAnkleRoll,
                RHipYawPitch, RHipRoll, RHipPitch, RKneePitch, RAnklePitch, RAnkleRoll]
        """
        if self.lateral_only:
            # Diagnostic: only hip roll, everything else at home
            lateral    = self.hip_roll_amp * math.sin(2.0 * math.pi * phase)
            R_hip_roll = -lateral
            L_hip_roll =  lateral
            return np.array([
                0.0, L_hip_roll,          self.hip_pitch_base,
                self.knee_pitch_base,     self.ankle_pitch_base, 0.0,
                0.0, R_hip_roll,          self.hip_pitch_base,
                self.knee_pitch_base,     self.ankle_pitch_base, 0.0,
            ], dtype=np.float32)

        if phase < 0.5:
            # RIGHT leg swings, LEFT leg is stance
            phi_swing  = phase        # 0 → 0.5
            phi_stance = phase        # 0 → 0.5

            R_hip, R_knee, R_ankle = self._swing_targets(phi_swing)
            L_hip, L_knee, L_ankle = self._stance_targets(phi_stance)

            # Lean left (onto stance leg) while right swings
            roll_mag   = self.hip_roll_amp * math.sin(math.pi * (phase / 0.5))
            R_hip_roll = -roll_mag
            L_hip_roll =  roll_mag
            R_hip_yaw =  0.0
            L_hip_yaw =  math.radians(2.0) * math.sin(math.pi * (phase / 0.5))

        else:
            # LEFT leg swings, RIGHT leg is stance
            phi_swing  = phase - 0.5  # 0 → 0.5
            phi_stance = phase - 0.5  # 0 → 0.5

            L_hip, L_knee, L_ankle = self._swing_targets(phi_swing)
            R_hip, R_knee, R_ankle = self._stance_targets(phi_stance)

            # Lean right (onto stance leg) while left swings
            roll_mag   = self.hip_roll_amp * math.sin(math.pi * ((phase - 0.5) / 0.5))
            R_hip_roll =  roll_mag
            L_hip_roll = -roll_mag
            R_hip_yaw =  math.radians(2.0) * math.sin(math.pi * ((phase - 0.5) / 0.5))
            L_hip_yaw =  0.0

        targets = np.array([
            L_hip_yaw, L_hip_roll, L_hip,
            L_knee,    L_ankle,    0.0,
            R_hip_yaw, R_hip_roll, R_hip,
            R_knee,    R_ankle,    0.0,
        ], dtype=np.float32)

        return targets


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
        "LShoulderRoll": 10,  "RShoulderRoll": -10,
        "LElbowYaw": -70,     "RElbowYaw": 70,
        "LElbowRoll": -25,    "RElbowRoll": 25,
        "HeadYaw": 0,         "HeadPitch": 0,
        "LWristYaw": 0,       "RWristYaw": 0,
        "LHand": 0,           "RHand": 0,
    }

    FINGER_JOINT_NAMES = [
        "LFinger11", "LFinger12", "LFinger13",
        "LFinger21", "LFinger22", "LFinger23",
        "LThumb1",   "LThumb2",
        "RFinger11", "RFinger12", "RFinger13",
        "RFinger21", "RFinger22", "RFinger23",
        "RThumb1",   "RThumb2",
    ]

    HAND_COLLISION_LINKS = [
        "l_wrist", "r_wrist", "l_gripper", "r_gripper",
        "LFinger11_link", "LFinger12_link", "LFinger13_link",
        "LFinger21_link", "LFinger22_link", "LFinger23_link",
        "LThumb1_link",   "LThumb2_link",
        "RFinger11_link", "RFinger12_link", "RFinger13_link",
        "RFinger21_link", "RFinger22_link", "RFinger23_link",
        "RThumb1_link",   "RThumb2_link",
    ]

    # Crouched home pose — matches real NAO walking posture
    LEG_HOME_POSE_DEG = {
        "LHipYawPitch": 0, "LHipRoll": 0,   "LHipPitch": -20,
        "LKneePitch": 40,  "LAnklePitch": -20, "LAnkleRoll": 0,
        "RHipYawPitch": 0, "RHipRoll": 0,   "RHipPitch": -20,
        "RKneePitch": 40,  "RAnklePitch": -20, "RAnkleRoll": 0,
    }

    LEG_JOINT_LIMITS_DEG = {
        "LHipYawPitch": (-65.6, 42.4), "LHipRoll": (-21.7, 45.3),
        "LHipPitch":    (-88.0, 27.7), "LKneePitch": (2.0, 121.0),
        "LAnklePitch":  (-68.1, 52.9), "LAnkleRoll": (-22.8, 44.1),
        "RHipYawPitch": (-65.6, 42.4), "RHipRoll": (-45.3, 21.7),
        "RHipPitch":    (-88.0, 27.7), "RKneePitch": (2.0, 121.0),
        "RAnklePitch":  (-68.0, 53.4), "RAnkleRoll": (-44.1, 22.8),
    }

    def __init__(
        self,
        urdf_path="../../robots/nao_isaac_sim/nao.urdf",
        render_mode="headless",
        physics_dt=1.0 / 120.0,
        control_dt=1.0 / 30.0,
        max_episode_steps=1500,
        spawn_height=0.3344,
        kp=35810.0,
        kd=500.0,
        max_effort=150.0,
        hip_kp=300_000.0,
        hip_kd=3_000.0,
        knee_kp=300_000.0,    # knees also need high gains — load-bearing
        knee_kd=3_000.0,
        action_scale=0.25,
        finger_kp=100.0,
        finger_kd=10.0,
        gait_hz=0.2,
        gait_warmup_steps=60,
        gait_lateral_only=False,
        reward_forward_vel=5.0,
        reward_alive=1.0,
        penalty_orientation=0.5,
        alive_scale=100.0,
        term_height=0.10,
        term_orient_qw=0.3,
        grace_period=50,
    ):
        super().__init__()

        self.urdf_path         = urdf_path
        self.render_mode       = render_mode
        self.physics_dt        = physics_dt
        self.control_dt        = control_dt
        self.decimation        = int(control_dt / physics_dt)
        self.max_episode_steps = max_episode_steps
        self.spawn_height      = spawn_height
        self.kp                = kp
        self.kd                = kd
        self.max_effort        = max_effort
        self.hip_kp            = hip_kp
        self.hip_kd            = hip_kd
        self.knee_kp           = knee_kp
        self.knee_kd           = knee_kd
        self.action_scale      = action_scale
        self.finger_kp         = finger_kp
        self.finger_kd         = finger_kd
        self.gait_hz           = gait_hz
        self.gait_warmup_steps = gait_warmup_steps
        self.gait_lateral_only = gait_lateral_only

        self.w_forward      = reward_forward_vel
        self.w_alive        = reward_alive
        self.w_orient       = penalty_orientation
        self.alive_scale    = alive_scale
        self.term_height    = term_height
        self.term_orient_qw = term_orient_qw
        self.grace_period   = grace_period

        self.n_legs = len(self.LEG_JOINT_NAMES)
        self._gait = WebotsReferenceGait(
            leg_home=None,               # assigned below once _leg_home exists
            ramp_duration_s=0.2,         # tiny ramp — home and phase-0 are aligned
            amplitude_scale=1.0,         # full amplitude — no mismatch to soften
            lateral_only=gait_lateral_only,
        )
        self._phase = 0.0

        self.n_obs = 3 + 3 + 2 + self.n_legs + 1 + 2  # = 23
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_obs,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_legs,), dtype=np.float32,
        )

        self._sim_initialized = False
        self._prev_action     = np.zeros(self.n_legs, dtype=np.float32)
        self._step_count      = 0

        self._leg_lower = np.zeros(self.n_legs, dtype=np.float32)
        self._leg_upper = np.zeros(self.n_legs, dtype=np.float32)
        self._leg_home  = np.zeros(self.n_legs, dtype=np.float32)

        # === Webots t=0 pose (EXACT match to Forwards.motion first frame) ===
        # Values from /usr/local/webots/.../Forwards.motion row 1 (t=0, Pose1),
        # converted from radians to degrees. This is Webots' initial standing-
        # ready pose, symmetric by construction. We must use this as home so
        # that when warmup ends and the gait begins at t=0, the commanded
        # target exactly matches NAO's current pose — zero discontinuity.
        WEBOTS_T0_POSE_DEG = {
            "LHipYawPitch":  0.0,
            "LHipRoll":     +1.547,    # 0.027 rad
            "LHipPitch":   -28.935,    # -0.505 rad
            "LKneePitch":  +59.702,    # 1.042 rad
            "LAnklePitch": -30.768,    # -0.537 rad
            "LAnkleRoll":   -1.547,    # -0.027 rad
            "RHipYawPitch":  0.0,
            "RHipRoll":     +1.547,
            "RHipPitch":   -28.935,
            "RKneePitch":  +59.702,
            "RAnklePitch": -30.768,
            "RAnkleRoll":   -1.547,
        }
        for i, name in enumerate(self.LEG_JOINT_NAMES):
            lo, hi = self.LEG_JOINT_LIMITS_DEG[name]
            self._leg_lower[i] = math.radians(lo)
            self._leg_upper[i] = math.radians(hi)
            self._leg_home[i]  = math.radians(WEBOTS_T0_POSE_DEG[name])

        # Sanity-check: our home pose must match the gait's first-frame exactly.
        # If they differ by more than ~0.5°, warmup-end will have a discontinuity.
        first_frame = self._gait._first_frame
        diff_deg = np.degrees(np.abs(self._leg_home - first_frame))
        print(f"[NaoWalkingEnv] home-vs-gait[t=0] max diff: {diff_deg.max():.3f} deg "
              f"(should be < 0.5)")

    # ══════════════════════════════════════════════════════════════════════════
    # Simulation init
    # ══════════════════════════════════════════════════════════════════════════
    def _init_sim(self):
        from isaacsim import SimulationApp

        headless = self.render_mode == "headless"
        self._sim_app = SimulationApp({"headless": headless, "width": 1280, "height": 720})

        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.utils.types import ArticulationAction
        import omni.kit.commands
        import isaacsim.asset.importer.urdf._urdf as urdf_module
        from pxr import UsdPhysics, UsdGeom, Usd, Gf, UsdShade

        self._Articulation       = Articulation
        self._ArticulationAction = ArticulationAction
        self._UsdPhysics         = UsdPhysics
        self._Usd                = Usd

        self._world = World(physics_dt=self.physics_dt, rendering_dt=1.0 / 60.0)
        self._world.scene.add_default_ground_plane()

        config = urdf_module.ImportConfig()
        config.merge_fixed_joints    = False
        config.fix_base              = False
        config.import_inertia_tensor = True
        config.distance_scale        = 1.0

        result = omni.kit.commands.execute(
            "URDFParseAndImportFile", urdf_path=self.urdf_path, import_config=config,
        )
        self._nao_prim_path = result[1]
        stage = self._world.stage
        root  = stage.GetPrimAtPath(self._nao_prim_path)

        # FIX 1: Disable instancing
        for prim in Usd.PrimRange(root):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)

        # FIX 2: CollisionAPI on foot meshes
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

        # FIX 3: Foot sole boxes
        for link, box in [("l_ankle", "l_sole_box"), ("r_ankle", "r_sole_box")]:
            bp       = f"{self._nao_prim_path}/{link}/collisions/{box}"
            box_prim = UsdGeom.Cube.Define(stage, bp)
            box_prim.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(box_prim.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(0.02, 0.0, -0.04511))
            xf.AddScaleOp().Set(Gf.Vec3d(0.10, 0.06, 0.005))
            UsdPhysics.CollisionAPI.Apply(box_prim.GetPrim())

            # FIX 4b: Friction material bound directly to the foot boxes.
            # Matches the ground-material values (1.0 static, 0.8 dynamic) so
            # the default PhysX material on the foot box doesn't end up lower
            # than the ground and cause inconsistent friction behaviour.
            foot_mat_path = f"{self._nao_prim_path}/{link}/collisions/foot_mat"
            foot_mat = UsdShade.Material.Define(stage, foot_mat_path)
            UsdPhysics.MaterialAPI.Apply(foot_mat.GetPrim())
            mat_api = UsdPhysics.MaterialAPI(foot_mat.GetPrim())
            mat_api.GetStaticFrictionAttr().Set(1.0)
            mat_api.GetDynamicFrictionAttr().Set(0.8)
            mat_api.GetRestitutionAttr().Set(0.0)
            UsdShade.MaterialBindingAPI.Apply(box_prim.GetPrim()).Bind(
                foot_mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics"
            )

        # FIX 4: Ground material
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.MaterialAPI):
                mat = UsdPhysics.MaterialAPI(prim)
                mat.GetRestitutionAttr().Set(0.0)
                mat.GetStaticFrictionAttr().Set(1.0)
                mat.GetDynamicFrictionAttr().Set(0.8)

        # FIX 5: USD gains before Articulation
        finger_set = set(self.FINGER_JOINT_NAMES + ["LWristYaw", "RWristYaw", "LHand", "RHand"])
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.RevoluteJoint):
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                if drive:
                    jn     = prim.GetName()
                    use_kp = float(self.finger_kp) if jn in finger_set else float(self.kp)
                    use_kd = float(self.finger_kd) if jn in finger_set else float(self.kd)
                    sa = drive.GetStiffnessAttr()
                    if sa: sa.Set(use_kp)
                    else:  drive.CreateStiffnessAttr(use_kp)
                    da = drive.GetDampingAttr()
                    if da: da.Set(use_kd)
                    else:  drive.CreateDampingAttr(use_kd)

        for prim in Usd.PrimRange(root):
            if prim.GetTypeName() == "PhysicsJoint" and prim.GetName() in finger_set:
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                if drive:
                    sa = drive.GetStiffnessAttr()
                    if sa: sa.Set(float(self.finger_kp))
                    da = drive.GetDampingAttr()
                    if da: da.Set(float(self.finger_kd))

        # FIX 6: Knee lower limit +2deg
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.RevoluteJoint) and "KneePitch" in prim.GetName():
                UsdPhysics.RevoluteJoint(prim).GetLowerLimitAttr().Set(2.0)

        # FIX 9: Disable hand/finger collision meshes
        n_disabled = 0
        for link_name in self.HAND_COLLISION_LINKS:
            for suffix in ["", "/collisions", "/visuals"]:
                p = stage.GetPrimAtPath(f"{self._nao_prim_path}/{link_name}{suffix}")
                if p and p.IsValid():
                    for child in Usd.PrimRange(p):
                        if child.HasAPI(UsdPhysics.CollisionAPI):
                            attr = UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr()
                            if attr:
                                attr.Set(False)
                                n_disabled += 1
        print(f"[NaoWalkingEnv] FIX 9: disabled {n_disabled} hand/finger collision meshes")

        # Create Articulation
        self._nao = self._Articulation(prim_path=self._nao_prim_path)
        self._world.scene.add(self._nao)
        self._world.reset()

        self._controller = self._nao.get_articulation_controller()
        self._num_dof    = self._nao.num_dof
        self._dof_names  = list(self._nao.dof_names)
        print(f"[NaoWalkingEnv] DOFs: {self._num_dof}")

        self._apply_gains()

        self._leg_dof_indices = []
        for name in self.LEG_JOINT_NAMES:
            if name in self._dof_names:
                self._leg_dof_indices.append(self._dof_names.index(name))
            else:
                raise ValueError(f"Leg joint '{name}' not found in: {self._dof_names}")

        self._full_home_pose = np.zeros(self._num_dof, dtype=np.float32)
        for name, deg in self.FIXED_JOINT_POSES_DEG.items():
            if name in self._dof_names:
                self._full_home_pose[self._dof_names.index(name)] = math.radians(deg)
        for i, idx in enumerate(self._leg_dof_indices):
            self._full_home_pose[idx] = self._leg_home[i]
        for name in self.FINGER_JOINT_NAMES:
            if name in self._dof_names:
                self._full_home_pose[self._dof_names.index(name)] = 0.0

        n_ctrl = (len(self.FIXED_JOINT_POSES_DEG) +
                  len(self.LEG_JOINT_NAMES) +
                  len(self.FINGER_JOINT_NAMES))
        print(f"[NaoWalkingEnv] FIX 7: {n_ctrl}/{self._num_dof} DOFs have explicit home poses")

        self._has_ang_vel_api = hasattr(self._nao, 'get_angular_velocity')
        if self._has_ang_vel_api:
            try:
                if self._nao.get_angular_velocity() is None:
                    self._has_ang_vel_api = False
            except Exception:
                self._has_ang_vel_api = False

        self._has_lin_vel_api = hasattr(self._nao, 'get_linear_velocity')
        if self._has_lin_vel_api:
            try:
                if self._nao.get_linear_velocity() is None:
                    self._has_lin_vel_api = False
            except Exception:
                self._has_lin_vel_api = False

        self._prev_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._settle_physics()
        self._sim_initialized = True

        lat = " [LATERAL ONLY]" if self.gait_lateral_only else ""
        gait_name = type(self._gait).__name__
        print(f"[NaoWalkingEnv] Using {gait_name}{lat}, warmup: {self.gait_warmup_steps} steps")

        # Describe whichever gait source is active.
        if hasattr(self._gait, "hip_pitch_amp"):
            # Legacy hand-coded ReferenceGait (sinusoidal, fixed amplitudes)
            print(f"[NaoWalkingEnv] Gait: {self.gait_hz}Hz")
            print(f"[NaoWalkingEnv] Hip amp: {math.degrees(self._gait.hip_pitch_amp):.1f}deg, "
                  f"Knee amp: {math.degrees(self._gait.knee_pitch_amp):.1f}deg, "
                  f"Ankle push-off: {math.degrees(self._gait.ankle_pushoff_amp):.1f}deg")
        else:
            # WebotsReferenceGait (trajectory playback) — report keyframe-derived stats
            q_min = np.degrees(self._gait._q.min(axis=0))
            q_max = np.degrees(self._gait._q.max(axis=0))
            period = self._gait._period
            print(f"[NaoWalkingEnv] Webots gait — period={period:.3f}s ({1 / period:.2f} Hz)")
            print(f"[NaoWalkingEnv]   Hip pitch range (L/R): "
                  f"[{q_min[2]:.1f},{q_max[2]:.1f}] / [{q_min[8]:.1f},{q_max[8]:.1f}] deg")
            print(f"[NaoWalkingEnv]   Knee pitch range (L/R): "
                  f"[{q_min[3]:.1f},{q_max[3]:.1f}] / [{q_min[9]:.1f},{q_max[9]:.1f}] deg")

    def _apply_gains(self):
        """
        Three-tier PD gains. Called after every world.reset().

        Tier 1 - Body/arm joints: kp=35810, kd=500
        Tier 2 - Hips + Knees:    kp=300k,  kd=3000  (load-bearing)
        Tier 3 - Fingers/wrists:  kp=100,   kd=10
        """
        high_gain_set = {
            "LHipYawPitch", "LHipRoll", "LHipPitch",
            "RHipYawPitch", "RHipRoll", "RHipPitch",
            "LKneePitch",   "RKneePitch",   # knees added — hold crouch under load
        }
        finger_set = set(self.FINGER_JOINT_NAMES +
                         ["LWristYaw", "RWristYaw", "LHand", "RHand"])

        kps = np.full(self._num_dof, float(self.kp),  dtype=np.float32)
        kds = np.full(self._num_dof, float(self.kd),  dtype=np.float32)
        for i, name in enumerate(self._dof_names):
            if name in finger_set:
                kps[i], kds[i] = float(self.finger_kp), float(self.finger_kd)
            elif name in high_gain_set:
                kps[i], kds[i] = float(self.knee_kp), float(self.knee_kd)

        self._controller.set_gains(kps, kds)
        self._controller.set_max_efforts(
            np.full(self._num_dof, float(self.max_effort), dtype=np.float32)
        )

    def _settle_physics(self):
        """10-step settle — robot cannot fall in this window."""
        self._teleport_to_start()
        for _ in range(10):
            self._nao.apply_action(
                self._ArticulationAction(joint_positions=self._full_home_pose)
            )
            self._world.step(render=(self.render_mode == "human"))
        self._nao.set_joint_velocities(np.zeros(self._num_dof))
        try:
            self._nao.set_linear_velocity(np.zeros(3))
            self._nao.set_angular_velocity(np.zeros(3))
        except Exception:
            pass

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

        self._world.reset()     # clear accumulated physics state
        self._apply_gains()     # world.reset() clears gains — restore them
        self._teleport_to_start()

        # 80-step settle — long enough for crouched pose to reach symmetry
        for _ in range(80):
            self._nao.apply_action(
                self._ArticulationAction(joint_positions=self._full_home_pose)
            )
            self._world.step(render=(self.render_mode == "human"))

        self._nao.set_joint_velocities(np.zeros(self._num_dof))
        try:
            self._nao.set_linear_velocity(np.zeros(3))
            self._nao.set_angular_velocity(np.zeros(3))
        except Exception:
            pass

        self._teleport_to_start()
        for _ in range(5):
            self._nao.apply_action(
                self._ArticulationAction(joint_positions=self._full_home_pose)
            )
            self._world.step(render=(self.render_mode == "human"))

        self._prev_action = np.zeros(self.n_legs, dtype=np.float32)
        self._step_count  = 0
        self._phase       = 0.0

        _, quat = self._nao.get_world_pose()
        self._prev_quat = (quat.copy().astype(np.float32)
                           if quat is not None and not np.any(np.isnan(quat))
                           else np.array([0., 0., 0., 1.], dtype=np.float32))

        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        if self._step_count < self.gait_warmup_steps:
            gait_target = self._leg_home.copy()
            self._phase = 0.0
        else:
            t = (self._step_count - self.gait_warmup_steps) * self.control_dt
            self._phase = self._gait.get_phase(t)
            gait_target = self._gait.get_targets(self._phase, t_since_warmup=t)

        leg_targets = np.clip(
            gait_target + action * self.action_scale,
            self._leg_lower, self._leg_upper,
        )

        full_target = self._full_home_pose.copy()
        for i, idx in enumerate(self._leg_dof_indices):
            full_target[idx] = leg_targets[i]

        # CRITICAL: apply_action INSIDE decimation loop
        for _ in range(self.decimation):
            self._nao.apply_action(
                self._ArticulationAction(joint_positions=full_target)
            )
            self._world.step(render=(self.render_mode == "human"))

        obs        = self._get_obs()
        reward     = self._compute_reward(action)
        terminated = self._check_termination()
        truncated  = self._step_count >= self.max_episode_steps

        self._prev_action = action.copy()
        _, quat = self._nao.get_world_pose()
        if quat is not None and not np.any(np.isnan(quat)):
            self._prev_quat = quat.copy().astype(np.float32)

        return obs, reward, terminated, truncated, {}

    # ══════════════════════════════════════════════════════════════════════════
    # Observations
    # ══════════════════════════════════════════════════════════════════════════
    def _get_obs(self):
        pos, quat = self._nao.get_world_pose()
        jp        = self._nao.get_joint_positions()

        if pos is None or np.any(np.isnan(pos)):
            return np.zeros(self.n_obs, dtype=np.float32)

        proj_gravity = self._quat_rotate_inverse(quat, np.array([0, 0, -1.0]))
        ang_vel      = self._get_angular_velocity(quat)
        lin_vel      = self._get_linear_velocity()

        leg_pos = np.zeros(self.n_legs, dtype=np.float32)
        if jp is not None:
            for i, idx in enumerate(self._leg_dof_indices):
                leg_pos[i] = jp[idx]

        joint_range  = self._leg_upper - self._leg_lower
        joint_mid    = (self._leg_upper + self._leg_lower) / 2.0
        leg_pos_norm = np.where(
            joint_range > 0, 2.0 * (leg_pos - joint_mid) / joint_range, 0.0
        ).astype(np.float32)

        if self._step_count < self.gait_warmup_steps:
            phase_sin, phase_cos = 0.0, 1.0
        else:
            phase_sin, phase_cos = self._gait.get_phase_signal(self._phase)

        obs = np.concatenate([
            proj_gravity.astype(np.float32),
            ang_vel,
            lin_vel,
            leg_pos_norm,
            np.array([float(pos[2])], dtype=np.float32),
            np.array([phase_sin, phase_cos], dtype=np.float32),
        ])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_linear_velocity(self):
        if self._has_lin_vel_api:
            try:
                v = self._nao.get_linear_velocity()
                if v is not None and not np.any(np.isnan(v)):
                    return np.array([float(v[0]), float(v[1])], dtype=np.float32)
            except Exception:
                pass
        return np.zeros(2, dtype=np.float32)

    def _get_angular_velocity(self, quat):
        if self._has_ang_vel_api:
            try:
                w = self._nao.get_angular_velocity()
                if w is not None and not np.any(np.isnan(w)):
                    return (self._quat_rotate_inverse(quat, w) * 0.25).astype(np.float32)
            except Exception:
                pass

        if quat is not None and not np.any(np.isnan(quat)):
            px, py, pz, pw = self._prev_quat
            cx, cy, cz, cw = quat
            dw =  cw*pw + cx*px + cy*py + cz*pz
            dx = -cw*px + cx*pw - cy*pz + cz*py
            dy = -cw*py + cx*pz + cy*pw - cz*px
            dz = -cw*pz - cx*py + cy*px + cz*pw
            if dw < 0: dx, dy, dz, dw = -dx, -dy, -dz, -dw
            return (2.0 * np.array([dx, dy, dz], dtype=np.float32) / self.control_dt * 0.25)

        return np.zeros(3, dtype=np.float32)

    def _get_base_position(self):
        pos, _ = self._nao.get_world_pose()
        return pos.copy() if pos is not None and not np.any(np.isnan(pos)) else np.zeros(3)

    @staticmethod
    def _quat_rotate_inverse(quat, vec):
        x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        q = np.array([-x, -y, -z])
        t = 2.0 * np.cross(q, vec)
        return (vec + w * t + np.cross(q, t)).astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # Reward
    # ══════════════════════════════════════════════════════════════════════════
    def _compute_reward(self, action):
        forward_vel = 0.0
        if self._has_lin_vel_api:
            try:
                v = self._nao.get_linear_velocity()
                if v is not None and not np.any(np.isnan(v)):
                    forward_vel = float(v[0])
            except Exception:
                pass

        r_forward = self.w_forward * forward_vel
        r_alive   = self.w_alive * (1.0 + self._step_count / max(self.alive_scale, 1.0))

        _, quat = self._nao.get_world_pose()
        if quat is not None and not np.any(np.isnan(quat)):
            pg = self._quat_rotate_inverse(quat, np.array([0, 0, -1.0]))
            orient_error = pg[0]**2 + pg[1]**2
        else:
            orient_error = 1.0
        r_orient = -self.w_orient * orient_error

        return float(r_forward + r_alive + r_orient)

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
            if abs(float(quat[3])) < self.term_orient_qw:
                return True
        return False

    def close(self):
        if self._sim_initialized:
            try:
                self._sim_app.close()
            except Exception:
                pass
            self._sim_initialized = False
