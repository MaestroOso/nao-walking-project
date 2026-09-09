"""
nao_walking_env.py — Gymnasium RL Environment for NAO Walking
==============================================================
test_17 (Stage 1 forward-fix): Reference gait + residual actions

STAGE 1 FIXES (this revision):
  FIX A: Forward velocity is computed in BODY frame, not world frame.
         Old code rewarded world-X velocity, which only equals "forward"
         if the robot's body-X axis happens to point along world +X. With
         the imported NAO URDF, this assumption was wrong (or unverified),
         so PPO learned to walk in the direction that maximized world-X,
         which corresponded to the robot's MECHANICAL BACKWARD direction.
  FIX B: A heading reward keeps the robot facing its spawn heading
         (prevents yaw-spinning to game the world-frame reward).
  FIX C: Reward weights rebalanced — alive bonus is now a small constant
         tiebreaker, not a growing dominant term that drowns out forward.
  FIX D: forward_axis_sign constant lets you flip the sign in ONE place
         if your URDF's body-X actually points backward. Run a quick
         "push test" the first time you train (see VERIFY_FORWARD_SIGN
         in this file) — if printed body_vx is negative when the robot
         visibly moves forward, set forward_axis_sign=-1.

Earlier (test16_rl) notes preserved below for reference.

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


# ══════════════════════════════════════════════════════════════════════════
# Reference Gait
# ══════════════════════════════════════════════════════════════════════════
class ReferenceGait:
    """
    Bipedal walking gait with DOUBLE SUPPORT phases.

    The previous (alternating) gait had instantaneous swing-leg switches
    at phase 0.5, which injected asymmetric reaction impulses on the body
    every cycle. The body never returned to symmetric state, so a small
    leftward drift accumulated linearly with no restoring force.

    This version splits each cycle into FOUR phases:
      0.00–0.10  (10%): double support, weight transfers L→R complete
      0.10–0.50  (40%): RIGHT leg swings, LEFT leg stance
      0.50–0.60  (10%): double support, weight transfers R→L
      0.60–1.00  (40%): LEFT leg swings, RIGHT leg stance

    During double support phases, BOTH legs are at the home pose. Only
    the hip roll changes (continuously) to shift weight between feet.
    The legs land/lift at the start/end of single-support windows, so
    transition impulses are absorbed by both feet being on the ground.
    """

    # Phase boundaries
    DS1_END   = 0.10   # end of first double-support
    SS1_END   = 0.50   # end of right-swing single-support
    DS2_END   = 0.60   # end of second double-support
    # SS2 runs DS2_END → 1.0

    def __init__(self, gait_hz=0.2, lateral_only=False):
        self.gait_hz      = gait_hz
        self.lateral_only = lateral_only

        # Base pose — matches crouched home pose
        self.hip_pitch_base   = math.radians(-20.0)
        self.knee_pitch_base  = math.radians(40.0)
        self.ankle_pitch_base = math.radians(-20.0)

        # Swing leg amplitudes (back to 8° now that double support absorbs
        # the transition impulses)
        self.hip_pitch_amp    = math.radians(8.0)
        self.knee_pitch_amp   = math.radians(8.0)

        # Stance leg ankle push-off
        self.ankle_pushoff_amp = math.radians(3.0)

        # Hip roll: lateral weight shift toward stance leg
        self.hip_roll_amp = math.radians(8.0)

    def get_phase(self, t):
        return (t * self.gait_hz) % 1.0

    def get_phase_signal(self, phase):
        angle = 2.0 * math.pi * phase
        return math.sin(angle), math.cos(angle)

    def _swing_targets(self, t):
        """t in [0, 1] — full swing trajectory parameter."""
        hip_pitch   = self.hip_pitch_base - self.hip_pitch_amp * math.sin(math.pi * t)
        knee_pitch  = self.knee_pitch_base + self.knee_pitch_amp * math.sin(math.pi * t)
        ankle_pitch = self.ankle_pitch_base
        return hip_pitch, knee_pitch, ankle_pitch

    def _stance_targets(self, t):
        """t in [0, 1] — stance trajectory parameter for ankle push-off."""
        hip_pitch   = self.hip_pitch_base
        knee_pitch  = self.knee_pitch_base
        ankle_push  = self.ankle_pushoff_amp * math.sin(math.pi * t)
        ankle_pitch = self.ankle_pitch_base - ankle_push
        return hip_pitch, knee_pitch, ankle_pitch

    def _home_leg(self):
        """Both legs at the crouched home pose."""
        return (self.hip_pitch_base, self.knee_pitch_base, self.ankle_pitch_base)

    def _roll_for_phase(self, phase):
        """
        Continuous roll signal across the entire cycle.
        Returns (L_hip_roll, R_hip_roll).

        Designed so that:
          - At phase 0.0: roll = 0 (neutral)
          - During DS1 (0.0–0.1): roll smoothly transfers to LEFT (lean L)
          - During right-swing SS1 (0.1–0.5): held leaning LEFT (peak)
          - During DS2 (0.5–0.6): roll smoothly transfers to RIGHT
          - During left-swing SS2 (0.6–1.0): held leaning RIGHT
        """
        if phase < self.DS1_END:
            # Ramp 0 → +1 over [0, 0.1]
            r = phase / self.DS1_END
            mag = self.hip_roll_amp * 0.5 * (1 - math.cos(math.pi * r))
        elif phase < self.SS1_END:
            # Hold at +max during right-swing
            mag = self.hip_roll_amp
        elif phase < self.DS2_END:
            # Ramp +1 → -1 over [0.5, 0.6]
            r = (phase - self.SS1_END) / (self.DS2_END - self.SS1_END)
            # Cosine sweep from +1 to -1
            mag = self.hip_roll_amp * math.cos(math.pi * r)
        else:
            # Hold at -max during left-swing
            mag = -self.hip_roll_amp
        # Lean LEFT (onto left foot) means L_hip_roll positive, R_hip_roll negative
        return mag, -mag

    def get_targets(self, phase):
        """
        Return target joint angles for 12 leg joints (LEG_JOINT_NAMES order).

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

        # ── Lateral weight shift (continuous across all 4 phases) ──
        L_hip_roll, R_hip_roll = self._roll_for_phase(phase)

        # ── Leg pitch trajectories per phase ──
        if phase < self.DS1_END:
            # Double support: both legs at home
            L_hip, L_knee, L_ankle = self._home_leg()
            R_hip, R_knee, R_ankle = self._home_leg()

        elif phase < self.SS1_END:
            # Right swings, left stance. Map phase ∈ [0.1, 0.5] → t ∈ [0, 1]
            t = (phase - self.DS1_END) / (self.SS1_END - self.DS1_END)
            R_hip, R_knee, R_ankle = self._swing_targets(t)
            L_hip, L_knee, L_ankle = self._stance_targets(t)

        elif phase < self.DS2_END:
            # Double support: both legs at home
            L_hip, L_knee, L_ankle = self._home_leg()
            R_hip, R_knee, R_ankle = self._home_leg()

        else:
            # Left swings, right stance. Map phase ∈ [0.6, 1.0] → t ∈ [0, 1]
            t = (phase - self.DS2_END) / (1.0 - self.DS2_END)
            L_hip, L_knee, L_ankle = self._swing_targets(t)
            R_hip, R_knee, R_ankle = self._stance_targets(t)

        # Hip yaws always zero (driving them independently caused asymmetry)
        L_hip_yaw = 0.0
        R_hip_yaw = 0.0

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
        action_scale=0.1,  # reduced from 0.2 — was launching robot at gait start
        finger_kp=100.0,
        finger_kd=10.0,
        gait_hz=0.2,
        gait_warmup_steps=60,
        gait_lateral_only=False,
        # Accepted for forward-compat with train_ppo.py (which passes
        # gait_version="v4"). Only one ReferenceGait implementation exists
        # in this file, so the value is recorded but not branched on. If
        # you later add a V4 gait class, switch on self.gait_version below
        # where self._gait is constructed.
        gait_version="v1",
        # Linear ramp from zero residual to full residual over this many
        # control steps, starting when the gait begins. Prevents the launch
        # artifact we saw at step 60: previously the residual jumped from 0
        # to full (±0.1 rad uniform random on 12 joints) in a single step,
        # which kicked the robot sideways before it could stabilize.
        action_ramp_steps=30,
        # ── Stage 1 reward rebalance ──
        # Tuning history:
        #   v0 (buggy):   forward=5, alive=1 (×growing), orient=0.5
        #                 → alive dominated, world-frame vx broken, walked backward
        #   v1 (stage 1): forward=10, alive=0.5 (flat), orient=0.5, heading=1.0
        #                 → survived 6k steps, but r_forward was only ~4% of total
        #                   signal, so policy optimized heading/alive first
        #   v2 (now):     forward=30, alive=0.3 (flat), orient=0.5, heading=0.3
        #                 → with body_vx ~ ±0.03 typical, r_forward=±0.9/step
        #                   decisively dominates r_alive=+0.3/step. PPO's
        #                   gradient now unambiguously points toward forward.
        reward_forward_vel=30.0,
        reward_alive=0.3,
        penalty_orientation=0.5,
        reward_heading=0.3,
        alive_scale=1e9,        # effectively disables the growing alive bonus
        # Set this to -1.0 if your URDF's body-X axis points backward.
        # See _compute_reward — body_vx is printed when VERIFY_FORWARD_SIGN=True.
        forward_axis_sign=1.0,
        term_height=0.10,
        term_orient_qw=0.3,
        # Grace period covers warmup (60) + action ramp (30) + a few extra
        # steps of gait settling (10) = 100. Termination cannot fire until
        # the policy is actually at full authority and the robot has had
        # a moment to stabilize into the gait.
        grace_period=100,
        # Catch-all so future train_ppo.py kwargs don't break this env.
        # Unknown kwargs are warned about but tolerated.
        **kwargs,
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
        self.gait_version      = gait_version
        self.action_ramp_steps = int(action_ramp_steps)

        if kwargs:
            print(f"[NaoWalkingEnv] Warning: unknown kwargs ignored: {list(kwargs.keys())}")
        if gait_version not in ("v1",):
            print(f"[NaoWalkingEnv] Note: gait_version={gait_version!r} requested but only "
                  f"the v1 ReferenceGait is implemented in this file. Using v1.")
        self.w_forward      = reward_forward_vel
        self.w_alive        = reward_alive
        self.w_orient       = penalty_orientation
        self.w_heading      = reward_heading
        self.alive_scale    = alive_scale
        self.forward_axis_sign = float(forward_axis_sign)
        self.term_height    = term_height
        self.term_orient_qw = term_orient_qw
        self.grace_period   = grace_period

        self.n_legs = len(self.LEG_JOINT_NAMES)
        self._gait  = ReferenceGait(gait_hz=gait_hz, lateral_only=gait_lateral_only)
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

        for i, name in enumerate(self.LEG_JOINT_NAMES):
            lo, hi = self.LEG_JOINT_LIMITS_DEG[name]
            self._leg_lower[i] = math.radians(lo)
            self._leg_upper[i] = math.radians(hi)
            self._leg_home[i]  = math.radians(self.LEG_HOME_POSE_DEG[name])

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
        from pxr import UsdPhysics, UsdGeom, Usd, Gf

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
        print(f"[NaoWalkingEnv] test_16 — alternating gait + ankle push-off{lat}")
        print(f"[NaoWalkingEnv] Gait: {self.gait_hz}Hz, warmup: {self.gait_warmup_steps} steps")
        print(f"[NaoWalkingEnv] Hip amp: {math.degrees(self._gait.hip_pitch_amp):.1f}deg, "
              f"Knee amp: {math.degrees(self._gait.knee_pitch_amp):.1f}deg, "
              f"Ankle push-off: {math.degrees(self._gait.ankle_pushoff_amp):.1f}deg")

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

        # ── Warmup: hold the home pose with NO residuals ──
        # Previous bug: during the warmup window, the gait_target was set to
        # _leg_home but `action * action_scale` was still added on top of it.
        # An untrained PPO policy emits uniform random actions in [-1, 1], so
        # the robot was getting ±0.2 rad of random twitching on every leg joint
        # for the entire 60-step warmup — which is exactly when it's supposed
        # to be settling. The robot would fall before the gait even started,
        # making it impossible for the policy to learn anything.
        #
        # Fix: zero out the residual contribution during warmup. The policy
        # can still observe and produce actions (so its internal state stays
        # consistent), but those actions don't affect the simulation until
        # after the warmup window closes.
        if self._step_count < self.gait_warmup_steps:
            gait_target = self._leg_home.copy()
            self._phase = 0.0
            effective_action = np.zeros_like(action)
        else:
            t = (self._step_count - self.gait_warmup_steps) * self.control_dt
            self._phase = self._gait.get_phase(t)
            gait_target = self._gait.get_targets(self._phase)
            # Ramp action from 0 → 1 over the first action_ramp_steps after
            # gait start. Previously the policy went from zero residual to
            # full ±action_scale in a single control step, which launched
            # the robot sideways at step 60 (verify logs showed world_v
            # magnitudes up to 0.29 m/s from a standing start — that's not
            # walking, that's the untrained policy kicking the legs out).
            steps_since_warmup = self._step_count - self.gait_warmup_steps
            if steps_since_warmup < self.action_ramp_steps:
                ramp = float(steps_since_warmup) / float(self.action_ramp_steps)
                effective_action = action * ramp
            else:
                effective_action = action

        leg_targets = np.clip(
            gait_target + effective_action * self.action_scale,
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
    # Reward (Stage 1 forward-fix)
    # ══════════════════════════════════════════════════════════════════════════
    #
    # Set this to True the FIRST time you train after applying Stage 1.
    # It will print body_vx, world_vx, and the body's projected forward axis
    # roughly once per second of sim time. Watch the printout while the robot
    # is upright and moving:
    #
    #   - If body_vx is POSITIVE when the robot visibly moves forward → leave
    #     forward_axis_sign=+1.0 (the default).
    #   - If body_vx is NEGATIVE when the robot visibly moves forward → set
    #     forward_axis_sign=-1.0 in the constructor (one place, one number).
    #
    # Once you've confirmed the sign, set VERIFY_FORWARD_SIGN back to False.
    # Sign has been confirmed as +1.0 via verify_sign6 run (see docs).
    # Turn this back on only if debugging future reward changes.
    VERIFY_FORWARD_SIGN = False

    def _compute_reward(self, action):
        pos, quat = self._nao.get_world_pose()

        # ── 1. Body-frame forward velocity ──
        # The bug in the previous revision was reading world-X velocity directly.
        # That only equals "forward" if the body-X axis is locked to world-X,
        # which is not guaranteed by the URDF import and degrades the moment
        # the robot yaws even slightly. Rotating the world-frame velocity into
        # the body frame gives us a yaw-invariant "forward" measurement.
        body_vx = 0.0
        world_v = None
        if self._has_lin_vel_api and quat is not None and not np.any(np.isnan(quat)):
            try:
                v = self._nao.get_linear_velocity()
                if v is not None and not np.any(np.isnan(v)):
                    world_v = np.array([float(v[0]), float(v[1]), float(v[2])],
                                       dtype=np.float32)
                    # _quat_rotate_inverse takes a world-frame vector and returns
                    # the same vector expressed in the body frame.
                    body_v = self._quat_rotate_inverse(quat, world_v)
                    body_vx = float(body_v[0]) * self.forward_axis_sign
            except Exception:
                pass

        # Clip body_vx to a reasonable range before rewarding it. The
        # verify6 log showed that when the robot is tumbling (heading_dot<0.7)
        # we get spurious body_vx readings like -1.1 because world velocity
        # contains a large vertical fall component being rotated into the
        # body frame. Clipping to ±0.5 m/s (roughly 2× target walking speed)
        # prevents these tumble frames from producing huge negative reward
        # spikes that destabilize the value function. A real NAO walks at
        # ~0.1 m/s; we'll never legitimately exceed 0.3 m/s in this env.
        body_vx = float(np.clip(body_vx, -0.5, 0.5))

        r_forward = self.w_forward * body_vx

        # ── 2. Heading reward (reshaped) ──
        # Previous: r_heading = w * max(0, heading_dot). That was a linear
        # ramp that gave partial credit for being halfway sideways, which
        # PPO happily farmed by trotting sideways at 45 degrees.
        # New: thresholded ramp — full reward above 0.9 (robot is "basically
        # forward"), zero reward below 0.7 (robot is clearly sideways or
        # worse). Between 0.7 and 0.9 we interpolate. This removes the
        # gradient around most upright poses (so the policy focuses on
        # forward progress) while still pushing back hard on yaw drift.
        heading_dot = 0.0
        if quat is not None and not np.any(np.isnan(quat)):
            x, y, z, w = (float(quat[0]), float(quat[1]),
                          float(quat[2]), float(quat[3]))
            # Rotation matrix first column = body +X expressed in world coords
            fwd_world_x = 1.0 - 2.0 * (y * y + z * z)
            heading_dot = self.forward_axis_sign * fwd_world_x
        # Thresholded ramp: 0 below 0.7, linear 0.7→0.9, saturates at 1.0
        heading_shaped = float(np.clip((heading_dot - 0.7) / 0.2, 0.0, 1.0))
        r_heading = self.w_heading * heading_shaped

        # ── 3. Alive bonus (now a flat tiebreaker, not a growing dominant term) ──
        r_alive = self.w_alive * (1.0 + self._step_count / max(self.alive_scale, 1.0))

        # ── 4. Orientation (upright) penalty ──
        if quat is not None and not np.any(np.isnan(quat)):
            pg = self._quat_rotate_inverse(quat, np.array([0, 0, -1.0]))
            orient_error = pg[0] ** 2 + pg[1] ** 2
        else:
            orient_error = 1.0
        r_orient = -self.w_orient * orient_error

        # ── 5. Optional debug printout for one-time sign verification ──
        if self.VERIFY_FORWARD_SIGN and (self._step_count % 30 == 0):
            wvx = float(world_v[0]) if world_v is not None else float("nan")
            wvy = float(world_v[1]) if world_v is not None else float("nan")
            print(f"[VERIFY] step={self._step_count:>4} "
                  f"body_vx={body_vx:+.4f}  world_v=({wvx:+.4f},{wvy:+.4f})  "
                  f"heading_dot={heading_dot:+.3f}  "
                  f"r_fwd={r_forward:+.3f} r_head={r_heading:+.3f} "
                  f"r_alive={r_alive:+.3f} r_orient={r_orient:+.3f}")

        return float(r_forward + r_heading + r_alive + r_orient)

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
