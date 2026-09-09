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


# ══════════════════════════════════════════════════════════════════════════
# Reference Gait V2 — Active-stance forward-stepping gait
# ══════════════════════════════════════════════════════════════════════════
class ReferenceGaitV2:
    """
    Active-stance forward-stepping gait for NAO.

    DESIGN PHILOSOPHY
    -----------------
    The original ReferenceGait is a stepping-in-place pattern where the
    swing leg oscillates around the home pose and the stance leg is static.
    Its fatal flaw: swinging the leg forward generates a reaction torque on
    the torso (Newton's third law), and with no active stance-leg push to
    counteract, the body slides BACKWARD over the static stance foot.

    This V2 gait fixes that by making the stance leg ACTIVE:

      1. Stance HipPitch ROTATES BACKWARD (becomes less negative) during the
         swing phase of the opposite leg. In joint space this means
         HipPitch going from -20deg toward -5deg. In world space this means
         the torso rotates forward around the stance hip — pushing the body
         forward over the support foot.

      2. Stance KneePitch EXTENDS SLIGHTLY during swing (40deg -> 32deg).
         This lifts the CoM slightly and helps project the body forward.

      3. Stance AnklePitch PLANTARFLEXES (ankle pitch becomes more negative,
         -20deg -> -28deg) for push-off, adding forward impulse.

      4. Swing leg lifts the knee significantly (+30deg from home) for
         adequate foot clearance, avoiding the scuffing issue.

      5. Swing foot LANDS AHEAD of its lift position (hip pitch ends at
         -25deg, which is 5deg more forward-flexed than home). This creates
         the geometric "step" that produces net forward progress per cycle.

      6. Double-support phases at each cycle boundary give the robot time
         to transfer weight laterally BEFORE lifting the swing leg.

      7. Symmetric hip yaw (kept at zero) to eliminate the asymmetric yaw
         drift that plagued V1.

    PHASE LAYOUT
    ------------
      0.00 - 0.15 : Double support, shift weight to LEFT  (lateral roll)
      0.15 - 0.50 : Single support on LEFT,  RIGHT leg swings
      0.50 - 0.65 : Double support, shift weight to RIGHT (lateral roll)
      0.65 - 1.00 : Single support on RIGHT, LEFT leg swings

    JOINT NAMING REMINDER (NAO convention, matches Aldebaran docs):
      HipPitch:   negative = thigh flexed forward (leg forward of vertical)
      KneePitch:  positive = knee bent (only direction allowed, non-negative)
      AnklePitch: negative = dorsiflexion (toes up),
                  more negative during push-off = plantarflexion inverted...
                  actually in NAO convention negative ankle-pitch while the
                  foot is on ground tilts the shin forward relative to foot,
                  which is what we want for push-off.

    All amplitudes here are INITIAL GUESSES. Expect to tune after observing
    the debug_gait_sim output. Parameters are tunable via __init__ args.
    """

    def __init__(
        self,
        gait_hz=0.05,               # 20-second cycle (twice as slow as iter 1)
        # Home pose (matches NaoWalkingEnv.LEG_HOME_POSE_DEG)
        hip_pitch_home_deg=-20.0,
        knee_pitch_home_deg=40.0,
        ankle_pitch_home_deg=-20.0,
        # ── Swing leg amplitudes ─────────────────────────────────────────
        # Iteration 2: cut roughly 2.5x from initial design after observing
        # that the larger motions caused both forward catapulting (iter 1)
        # and lateral over-commit (iter 2 first attempt). Real NAO walks use
        # small steps with subtle leg motion — we mimic that here.
        swing_hip_pitch_amp_deg=6.0,     # was 15 — small forward swing
        swing_hip_pitch_land_deg=2.0,    # was 5  — small step length
        swing_knee_lift_amp_deg=12.0,    # was 30 — minimal foot clearance
        swing_knee_land_offset_deg=2.0,  # was 5
        # ── Stance leg amplitudes ────────────────────────────────────────
        # Reduced to prevent over-aggressive torso rotation during push.
        stance_hip_push_amp_deg=5.0,     # was 15 — gentler body push
        stance_knee_extend_amp_deg=3.0,  # was 8
        stance_ankle_pushoff_amp_deg=3.0,  # was 8
        # ── Lateral weight transfer ──────────────────────────────────────
        # Smaller peak roll, AND gradual decay during single support
        # (see lateral_decay_during_swing below) so the CoM doesn't
        # over-commit to the stance side and tip the robot sideways.
        hip_roll_amp_deg=4.0,            # was 7 — less lateral over-commit
        lateral_decay_during_swing=True, # NEW: ramp roll down during swing
        # Phase timing (fractions of cycle)
        double_support_frac=0.15,        # 15% double support at each transition
    ):
        self.gait_hz = gait_hz

        # Convert all degrees to radians once
        self.hp_home   = math.radians(hip_pitch_home_deg)
        self.kp_home   = math.radians(knee_pitch_home_deg)
        self.ap_home   = math.radians(ankle_pitch_home_deg)

        self.sw_hip_amp   = math.radians(swing_hip_pitch_amp_deg)
        self.sw_hip_land  = math.radians(swing_hip_pitch_land_deg)
        self.sw_knee_lift = math.radians(swing_knee_lift_amp_deg)
        self.sw_knee_land = math.radians(swing_knee_land_offset_deg)

        self.st_hip_push     = math.radians(stance_hip_push_amp_deg)
        self.st_knee_extend  = math.radians(stance_knee_extend_amp_deg)
        self.st_ankle_push   = math.radians(stance_ankle_pushoff_amp_deg)

        self.hip_roll_amp = math.radians(hip_roll_amp_deg)
        self.lateral_decay_during_swing = lateral_decay_during_swing

        # Phase boundaries (hard-coded layout)
        self.ds_frac = double_support_frac
        # 0.00 → 0.15 : DS shift left
        # 0.15 → 0.50 : SS left, right swings
        # 0.50 → 0.65 : DS shift right
        # 0.65 → 1.00 : SS right, left swings
        self.p0 = 0.0
        self.p1 = self.ds_frac                       # 0.15
        self.p2 = 0.50
        self.p3 = 0.50 + self.ds_frac                # 0.65
        self.p4 = 1.0

    def get_phase(self, t):
        """Return phase in [0, 1) based on elapsed time."""
        return (t * self.gait_hz) % 1.0

    def get_phase_signal(self, phase):
        """Return (sin, cos) of phase for observation encoding."""
        angle = 2.0 * math.pi * phase
        return math.sin(angle), math.cos(angle)

    @staticmethod
    def _smooth_bump(u):
        """
        Smooth bump: 0 at u=0, 1 at u=0.5, 0 at u=1. Peak at mid-interval.
        Uses sin(pi*u) so the derivative is zero at endpoints (smooth start/stop).
        """
        if u < 0.0: u = 0.0
        if u > 1.0: u = 1.0
        return math.sin(math.pi * u)

    @staticmethod
    def _smooth_ramp(u):
        """
        Smooth ramp from 0 to 1: 0 at u=0, 0.5 at u=0.5, 1 at u=1.
        Uses (1 - cos(pi*u)) / 2 for zero derivative at endpoints.
        """
        if u < 0.0: u = 0.0
        if u > 1.0: u = 1.0
        return 0.5 * (1.0 - math.cos(math.pi * u))

    def _swing_leg_targets(self, u):
        """
        Compute swing-leg joint targets for swing progress u in [0, 1].
        u=0: leg about to lift (at home pose)
        u=0.5: mid-swing, knee lifted, hip at max forward flexion
        u=1: leg landed in FORWARD position (hip more negative than home,
             knee slightly bent for absorption)

        Returns (hip_pitch, knee_pitch, ankle_pitch) in radians.
        """
        # Hip pitch: starts at home, dips to home - sw_hip_amp at mid-swing,
        # then comes UP to home - sw_hip_land (which is less than sw_hip_amp,
        # so the foot lands FORWARD of home but not as far as mid-swing peak).
        # We build this as a piecewise-smooth curve.
        #
        # For 0 <= u <= 0.5: linear interp from home to (home - sw_hip_amp)
        # with a smooth ramp.
        # For 0.5 <= u <= 1: smooth interp from (home - sw_hip_amp) to
        # (home - sw_hip_land), landing forward.
        if u < 0.5:
            k = self._smooth_ramp(u / 0.5)  # 0 -> 1 over [0, 0.5]
            hip = self.hp_home - self.sw_hip_amp * k
        else:
            k = self._smooth_ramp((u - 0.5) / 0.5)  # 0 -> 1 over [0.5, 1]
            hip = (self.hp_home - self.sw_hip_amp) + \
                  (self.sw_hip_amp - self.sw_hip_land) * k

        # Knee: bends UP (large increase) for mid-swing clearance,
        # returns to nearly home with a small landing offset.
        # Bump is smooth: 0 at u=0, peaks at u=0.5, falls to sw_knee_land at u=1.
        if u < 0.5:
            k = self._smooth_ramp(u / 0.5)
            knee = self.kp_home + self.sw_knee_lift * k
        else:
            k = self._smooth_ramp((u - 0.5) / 0.5)
            knee = (self.kp_home + self.sw_knee_lift) - \
                   (self.sw_knee_lift - self.sw_knee_land) * k

        # Ankle: keep roughly at home. The foot stays parallel to ground-ish;
        # precise foot orientation control would need IK but this is close enough.
        # Small compensation: as hip goes forward, add a tiny negative
        # ankle correction to keep foot sole more parallel to ground.
        ankle = self.ap_home

        return hip, knee, ankle

    def _stance_leg_targets(self, u):
        """
        Compute stance-leg joint targets for stance progress u in [0, 1].
        u=0: stance leg just took the load (at home pose)
        u=0.5: mid-stance, maximum push: hip rotated backward, knee extended,
               ankle plantarflexed
        u=1: stance about to end, returning toward home for weight transfer

        THIS IS THE KEY FIX: the stance leg is ACTIVE. It pushes the body
        forward using hip rotation + knee extension + ankle push-off.

        Returns (hip_pitch, knee_pitch, ankle_pitch) in radians.
        """
        bump = self._smooth_bump(u)  # 0 at ends, 1 at middle

        # Stance hip rotates BACKWARD (less flexed). In NAO convention, this
        # means hip_pitch goes from -20 toward -5, i.e. becomes LESS NEGATIVE.
        hip = self.hp_home + self.st_hip_push * bump

        # Stance knee extends slightly (less bent) → helps lift CoM and drive
        # body forward. Knee pitch goes from 40 toward 32 (LESS POSITIVE).
        knee = self.kp_home - self.st_knee_extend * bump

        # Stance ankle plantarflexes for push-off. In NAO, ankle pitch going
        # MORE NEGATIVE while the foot is planted tilts the shin forward
        # relative to the foot — this pushes the robot forward.
        ankle = self.ap_home - self.st_ankle_push * bump

        return hip, knee, ankle

    def _landed_to_home_targets(self, u):
        """
        Smoothly interpolate from the swing-landed pose back to home pose.
        Used during double-support phases to bring the recently-landed leg
        back to home BEFORE it takes over as the active stance leg in the
        next phase. This ensures continuity at phase boundaries.

        u=0: leg is at landed pose (hip forward, knee slightly bent)
        u=1: leg is at home pose (-20°, +40°, -20°)
        """
        k = self._smooth_ramp(u)  # 0 → 1
        # Landed pose from end of swing:
        landed_hip   = self.hp_home - self.sw_hip_land     # e.g. -25°
        landed_knee  = self.kp_home + self.sw_knee_land    # e.g. +45°
        landed_ankle = self.ap_home                         # -20°

        hip   = landed_hip + (self.hp_home - landed_hip) * k
        knee  = landed_knee + (self.kp_home - landed_knee) * k
        ankle = landed_ankle + (self.ap_home - landed_ankle) * k
        return hip, knee, ankle

    def get_targets(self, phase):
        """
        Return target joint angles for 12 leg joints (LEG_JOINT_NAMES order).

        Order: [LHipYawPitch, LHipRoll, LHipPitch, LKneePitch, LAnklePitch, LAnkleRoll,
                RHipYawPitch, RHipRoll, RHipPitch, RKneePitch, RAnklePitch, RAnkleRoll]
        """
        # Initialize all leg joints at home pose
        L_hip_yaw  = 0.0
        L_hip_roll = 0.0
        L_hip      = self.hp_home
        L_knee     = self.kp_home
        L_ankle    = self.ap_home
        L_ank_roll = 0.0

        R_hip_yaw  = 0.0
        R_hip_roll = 0.0
        R_hip      = self.hp_home
        R_knee     = self.kp_home
        R_ankle    = self.ap_home
        R_ank_roll = 0.0

        if self.p0 <= phase < self.p1:
            # ─── PHASE 1: Double support, shift weight to LEFT ───
            # Right leg is at home (it hasn't done anything in the previous cycle
            # end — actually wait, at the end of the previous cycle left leg swung
            # so it's the LEFT leg that needs to return from landed → home).
            u = (phase - self.p0) / (self.p1 - self.p0)  # 0 → 1

            # LEFT leg was the last to swing (phase 0.65-1.00 in prev cycle),
            # so it is currently at landed pose. Interpolate back to home.
            L_hip, L_knee, L_ankle = self._landed_to_home_targets(u)
            # Right leg stays at home throughout this DS.

            # Lateral weight transfer: smoothly ramp hip roll from 0 to +amp.
            # When lateral_decay_during_swing=True, the previous swing phase
            # (Phase 4) ended at 0, so we ramp 0 → +amp here. When False, the
            # previous Phase 4 also ended at 0 (constant -amp during swing,
            # then Phase 4 holds at -amp until DS — wait, no, if we hold
            # -amp during swing, then DS starts from -amp, not 0). So this
            # ramp 0 → +amp is only correct in decay mode. For non-decay
            # mode we need to ramp from -amp to +amp.
            if self.lateral_decay_during_swing:
                k = self._smooth_ramp(u)  # 0 → 1
                roll = self.hip_roll_amp * k
            else:
                # Old behavior: previous swing held -amp, so ramp -amp → +amp
                k = self._smooth_ramp(u)
                roll = self.hip_roll_amp * (-1.0 + 2.0 * k)
            L_hip_roll =  roll
            R_hip_roll = -roll
            # Ankle rolls compensate so feet stay flat on ground
            L_ank_roll = -roll
            R_ank_roll =  roll

        elif self.p1 <= phase < self.p2:
            # ─── PHASE 2: Single support on LEFT, RIGHT leg swings forward ───
            u = (phase - self.p1) / (self.p2 - self.p1)  # 0 → 1

            # LEFT leg is the ACTIVE STANCE: pushes body forward
            L_hip, L_knee, L_ankle = self._stance_leg_targets(u)

            # RIGHT leg is swinging forward
            R_hip, R_knee, R_ankle = self._swing_leg_targets(u)

            # Lateral weight shift handling (Iter 2 fix):
            # Old behavior held +amp throughout swing → CoM over-committed
            # to left side, robot tipped left when right foot lifted.
            # New behavior decays roll from +amp at u=0 (max lean when
            # right foot lifts off, when we need maximum CoM support over
            # the left foot) to 0 at u=1 (returning toward center as the
            # swing foot is about to land and DS begins).
            if self.lateral_decay_during_swing:
                k = self._smooth_ramp(u)  # 0 → 1
                roll = self.hip_roll_amp * (1.0 - k)  # +amp → 0
            else:
                roll = self.hip_roll_amp  # constant +amp (old behavior)
            L_hip_roll =  roll
            R_hip_roll = -roll
            L_ank_roll = -roll
            R_ank_roll =  roll

        elif self.p2 <= phase < self.p3:
            # ─── PHASE 3: Double support, shift weight to RIGHT ───
            u = (phase - self.p2) / (self.p3 - self.p2)  # 0 → 1
            k = self._smooth_ramp(u)
            # Roll transitions from where Phase 2 ended to -amp (right lean):
            # - decay mode:    Phase 2 ends at 0, so ramp 0 → -amp
            # - non-decay mode:Phase 2 ends at +amp, so ramp +amp → -amp
            if self.lateral_decay_during_swing:
                roll = -self.hip_roll_amp * k  # 0 → -amp
            else:
                roll = self.hip_roll_amp * (1.0 - 2.0 * k)  # +amp → -amp
            L_hip_roll =  roll
            R_hip_roll = -roll
            L_ank_roll = -roll
            R_ank_roll =  roll

            # RIGHT leg was just the swing leg. Interpolate it from landed
            # pose BACK TO HOME so that stance phase begins cleanly at p3.
            # Left leg stays at home (it was the stance in the prev phase
            # and stance_targets returns to home at u=1, so it's continuous).
            R_hip, R_knee, R_ankle = self._landed_to_home_targets(u)

        elif self.p3 <= phase < self.p4:
            # ─── PHASE 4: Single support on RIGHT, LEFT leg swings forward ───
            u = (phase - self.p3) / (self.p4 - self.p3)  # 0 → 1

            # RIGHT leg is the ACTIVE STANCE
            R_hip, R_knee, R_ankle = self._stance_leg_targets(u)

            # LEFT leg is swinging forward
            L_hip, L_knee, L_ankle = self._swing_leg_targets(u)

            # Lateral decay (mirror of Phase 2): roll starts at -amp (max
            # right lean when left foot lifts off) and decays to 0 by end
            # of swing.
            if self.lateral_decay_during_swing:
                k = self._smooth_ramp(u)  # 0 → 1
                roll = -self.hip_roll_amp * (1.0 - k)  # -amp → 0
            else:
                roll = -self.hip_roll_amp  # constant -amp (old behavior)
            L_hip_roll =  roll
            R_hip_roll = -roll
            L_ank_roll = -roll
            R_ank_roll =  roll

        targets = np.array([
            L_hip_yaw, L_hip_roll, L_hip,
            L_knee,    L_ankle,    L_ank_roll,
            R_hip_yaw, R_hip_roll, R_hip,
            R_knee,    R_ankle,    R_ank_roll,
        ], dtype=np.float32)

        return targets


class ReferenceGaitV3:
    """
    Monotonic-stance walking gait — V3.

    DESIGN PHILOSOPHY (the key insight)
    ------------------------------------
    The V2 gait was stable but didn't move forward because its stance push
    was a SYMMETRIC bump: hip went -20° → -15° → -20° during stance. The
    push and the return canceled each other, so the body oscillated forward
    and back with near-zero net motion (~0.4 cm/cycle).

    V3 fixes this with a structurally different stance push: MONOTONIC.
    During stance, the hip ramps from home (-20°) toward extended (-12°) and
    STAYS THERE through the next double-support phase. The leg "resets" back
    to home only during its swing phase, when the foot is in the air and
    moving the joint doesn't push the body.

    The result: each stance phase contributes a NET forward translation of
    the body, and the resets happen without any backward impulse.

    PER-LEG CYCLE STRUCTURE
    -----------------------
    Each leg goes through 4 roles per cycle (offset by half-cycle from the
    other leg). The hip pitch trajectory for each leg over one cycle is:

    LEFT LEG:
      0.00-0.15  DS-shift-left       hold L_hip at home (-20°)
      0.15-0.50  L STANCE             L_hip ramps -20° → -12° (monotonic push)
      0.50-0.65  DS-shift-right       L_hip held at -12° (foot now behind body)
      0.65-1.00  L SWING (in air)     L_hip: -12° → -25° → -20° (lift then land at home)

    RIGHT LEG (mirrored, half-cycle offset):
      0.00-0.15  DS-shift-left        hold R_hip at home (-20°)
      0.15-0.50  R SWING (in air)     R_hip: -20° → -25° → -12° (lift then land forward)

      Wait — that doesn't match. Let me re-derive.

    Actually the cleanest way to think about it: at phase 0.00, BOTH legs are
    at home (-20°). The cycle then does:
      - L stance push during 0.15-0.50: L_hip -20° → -12° (forward push)
      - R swing during 0.15-0.50:       R_hip -20° → ??? (R is in the air, can do anything)
      - L holds at -12° during 0.50-0.65; R has just landed at ???
      - R stance push during 0.65-1.00: R_hip ??? → ???
      - L swing during 0.65-1.00:       L_hip -12° → ??? → -20° (returns to home for next cycle)

    For the cycle to be SELF-CONSISTENT (every cycle starts identically), at
    phase 0.0 both legs must be at home (-20°). This means R_hip ALSO must
    return to -20° at the end of its stance (which is at phase 1.0 = 0.0).

    So R stance must go from (its swing landing pose) → -20° at phase 1.0.
    What's R's swing landing pose? That's set by what R does during swing
    (phase 0.15-0.50). For R's stance to push forward, R's swing landing
    must be FORWARD-FLEXED relative to where R's stance ends. If stance ends
    at -20° (home), then swing must land at MORE NEGATIVE (e.g. -28°), so
    that R stance ramps -28° → -20° (8° forward push, same magnitude as L).

    So:
      0.00-0.15  DS-shift-left:   L=-20°, R=-20° (both at home, weight transfers left)
      0.15-0.50  L stance + R swing:
                 L: -20° → -12° (monotonic push)
                 R: -20° → -28° → -28° (lift and forward swing, land at -28° forward of home)
      0.50-0.65  DS-shift-right:  L=-12° (held), R=-28° (held)
      0.65-1.00  R stance + L swing:
                 R: -28° → -20° (monotonic push, 8° just like L)
                 L: -12° → -25° → -20° (lift and swing back to home for next cycle)
      1.00 = 0.00:    L=-20°, R=-20° ✓ self-consistent!

    Note the asymmetry: L stance pushes -20°→-12° (above home), while R
    stance pushes -28°→-20° (below home). This is because of where each
    leg "lives" in the cycle. The body translation is the same magnitude
    for both stances (8°), so we get symmetric body motion.

    Knee and ankle: keep simple.
      - Knee at home (40°) during stance and DS
      - Knee bumps to +52° during swing (12° lift, same as V2)
      - Ankle at home (-20°) throughout (no plantarflexion in V3 first cut)

    Hip roll: copied verbatim from V2 (lateral decay during swing). It worked.
    """

    def __init__(
        self,
        gait_hz=0.05,                  # 20-second cycle (same as V2 iter 2)
        # Home pose
        hip_pitch_home_deg=-20.0,
        knee_pitch_home_deg=40.0,
        ankle_pitch_home_deg=-20.0,
        # ── Stance: monotonic forward push amplitude ─────────────────────
        # The hip rotates by this much during stance.
        # iter 6 used 8° → fell during L stance push because the reaction
        # torque from hip rotation tipped the torso backward in single
        # support (the foot couldn't apply enough counter-torque to keep
        # the body upright while the lateral lean was also active).
        # iter 7: halved to 4° to roughly halve the reaction torque on the
        # torso. Trade-off: forward translation per cycle drops from ~2.6cm
        # to ~1.3cm. That's still well above the "1cm/cycle" goal and is a
        # fine base for an RL residual policy to build on.
        stance_push_amp_deg=4.0,
        # ── Swing: lift and forward placement ────────────────────────────
        # Swing knee lift amplitude (peak at mid-swing)
        swing_knee_lift_amp_deg=12.0,
        # Swing hip lift amplitude — how much further forward-flexed the hip
        # gets at peak swing (above the swing's own start/end positions).
        # This is the "leg lifts up and forward" motion.
        swing_hip_lift_amp_deg=5.0,
        # ── Lateral weight transfer ──────────────────────────────────────
        # V3 iter 2 update: Iter 1 used hip_roll_amp=4° (copied from V2) and
        # the robot fell during DS-shift-right because the body had drifted
        # ~3 cm leftward during L stance — the 4° lean wasn't enough to keep
        # the CoM directly over the L foot during single support, and the
        # accumulated leftward momentum overwhelmed the rightward DS command.
        #
        # The geometric requirement is: with NAO's CoM ~30 cm above the
        # ankles and the support foot ~5 cm to the side of the body
        # centerline, the body needs to lean by atan(5/30) ≈ 9.5° to put
        # the CoM directly over the support foot. We use 9° here, just
        # below this estimate, leaving a tiny margin.
        hip_roll_amp_deg=9.0,            # was 4.0 in iter 1
        # V3 iter 3 update: lateral_decay_during_swing was True in iter 1
        # and iter 2, but the data showed this was the literal cause of the
        # fall: the lean decayed from peak (at start of swing) to 0 (at end
        # of swing), which means by the time the robot was in deep single
        # support, the lean was already gone. Without the lean, the CoM
        # drifted off the support foot and the body tipped. The fix is to
        # HOLD the lean at full amplitude throughout the single support
        # phase, only changing it during the DS phases. Setting to False
        # activates the existing non-decay code path which does exactly this.
        lateral_decay_during_swing=False,  # was True in iter 1 and iter 2
        # Phase timing
        # V3 iter 2 update: extended from 0.15 → 0.25. With more lateral
        # momentum (from larger lean and more dynamic stance push), we need
        # MORE time during DS to transfer weight from one foot to the other.
        # 0.25 fraction = 5 seconds per DS phase (at 20s cycle).
        double_support_frac=0.25,        # was 0.15 in iter 1
        # V3 iter 4 update: smooth gait initialization.
        # In iter 3, the moment the gait started (after warmup), the steady-
        # state cycle's Phase 1 begins with hip roll at -9° (held from the
        # previous cycle's R stance end). But the warmup ends with hip roll
        # at 0°. This created a 9° instantaneous jump in the commanded value
        # at the very first gait step. With kp=300k on the hip joints, the
        # PD controller slammed the hips from 0° to ±9° in a single physics
        # step, perturbing the body laterally — visually appearing as the
        # legs "closing in really fast" before the gait properly started.
        # The fix: during the first gait_init_blend_steps calls to
        # get_targets, smoothly blend the hip roll command from 0 (warmup
        # state) toward its steady-state value. After this many calls, we
        # use the steady-state command directly. 60 calls = 2 seconds at
        # control_dt=1/30, which is the full DS-shift-left phase duration —
        # plenty of time for the body to smoothly transition into the gait.
        gait_init_blend_steps=60,
        # V3 iter 5 update: phase origin shift.
        # The iter 4 debug log revealed that starting at phase 0.0 (beginning
        # of DS-shift-left) causes the roll command to ramp from -amp to +amp
        # across the phase, crossing zero at phase 0.075. Combined with the
        # init blend multiplier (also ramping 0→1 across the same window),
        # the product produces a non-monotonic roll command that first pulls
        # the robot one way then the other — visually "legs close, then open".
        #
        # The fix: start the gait at phase 0.075, where the steady-state roll
        # is naturally zero and about to ramp up monotonically to +amp. The
        # robot enters the gait with commanded_roll=0 (matching its settled
        # pose with 0° roll) and then smoothly leans left over the remaining
        # 1.5s of DS-shift-left before the L stance push begins.
        #
        # On the pitch side, phase 0.075 is still inside DS-shift-left, so
        # all hip/knee/ankle commands equal home — matching the robot's
        # warmup pose exactly. No discontinuity on any joint at gait start.
        phase_offset=0.075,
    ):
        self.gait_hz = gait_hz
        self.phase_offset = float(phase_offset)

        # Convert all degrees to radians once
        self.hp_home   = math.radians(hip_pitch_home_deg)
        self.kp_home   = math.radians(knee_pitch_home_deg)
        self.ap_home   = math.radians(ankle_pitch_home_deg)

        self.stance_push   = math.radians(stance_push_amp_deg)
        self.sw_knee_lift  = math.radians(swing_knee_lift_amp_deg)
        self.sw_hip_lift   = math.radians(swing_hip_lift_amp_deg)

        self.hip_roll_amp = math.radians(hip_roll_amp_deg)
        self.lateral_decay_during_swing = lateral_decay_during_swing

        # V3 iter 4: gait init blend state
        self.gait_init_blend_steps = int(gait_init_blend_steps)
        self._calls_since_init = 0    # incremented each call to get_targets

        # Phase boundaries
        self.ds_frac = double_support_frac
        self.p0 = 0.0
        self.p1 = self.ds_frac                       # 0.15
        self.p2 = 0.50
        self.p3 = 0.50 + self.ds_frac                # 0.65
        self.p4 = 1.0

        # Pre-computed key hip values for clarity:
        # L leg goes between home and (home + stance_push) ["above home"]
        self.L_stance_start_hip = self.hp_home                    # -20°
        self.L_stance_end_hip   = self.hp_home + self.stance_push # -12°
        # R leg goes between (home - stance_push) and home ["below home"]
        self.R_swing_land_hip   = self.hp_home - self.stance_push # -28°
        self.R_stance_end_hip   = self.hp_home                    # -20°

    def get_phase(self, t):
        # V3 iter 5: shift the phase origin by phase_offset (default 0.075)
        # so that at t=0 we are at the zero-crossing of the DS-shift-left
        # roll ramp instead of at its -amp starting point. This lets the gait
        # enter smoothly from the robot's settled home pose with no
        # discontinuity on any commanded joint.
        return ((t * self.gait_hz) + self.phase_offset) % 1.0

    def get_phase_signal(self, phase):
        angle = 2.0 * math.pi * phase
        return math.sin(angle), math.cos(angle)

    def reset_init_blend(self):
        """
        Reset the gait init blend counter so the next get_targets call
        starts from the smoothly-blended initial state again.

        Called by NaoWalkingEnv.reset() at the start of each episode so
        that every episode begins with a smooth gait transition, not just
        the very first one.
        """
        self._calls_since_init = 0

    @staticmethod
    def _smooth_ramp(u):
        """Smooth ramp from 0 to 1 with zero derivative at endpoints."""
        if u < 0.0: u = 0.0
        if u > 1.0: u = 1.0
        return 0.5 * (1.0 - math.cos(math.pi * u))

    @staticmethod
    def _smooth_bump(u):
        """Smooth bump: 0 at u=0, 1 at u=0.5, 0 at u=1."""
        if u < 0.0: u = 0.0
        if u > 1.0: u = 1.0
        return math.sin(math.pi * u)

    def _swing_hip_arc(self, u, start_angle, end_angle, lift_amp):
        """
        Compute swing-leg hip pitch over swing progress u in [0, 1].

        At u=0: hip is at start_angle (where the stance ended)
        At u=0.5: hip is at MAX FORWARD FLEXION (more negative than both
                  start and end), where the leg is highest in the air
        At u=1: hip is at end_angle (the desired landing pose)

        The shape is: linearly interpolate from start to end, plus a smooth
        bump pulling the angle MORE NEGATIVE (forward) at mid-swing.
        Lift amplitude controls how much extra forward flexion the bump adds.
        """
        # Linear interpolation from start to end (smoothed)
        k = self._smooth_ramp(u)
        baseline = start_angle + (end_angle - start_angle) * k
        # Smooth bump pulling the hip more forward (more negative) at mid-swing
        bump = self._smooth_bump(u)
        return baseline - lift_amp * bump

    def _swing_knee_arc(self, u):
        """
        Compute swing-leg knee pitch over swing progress u.
        Returns to home at both endpoints, lifts at mid-swing.
        """
        bump = self._smooth_bump(u)
        return self.kp_home + self.sw_knee_lift * bump

    def get_targets(self, phase):
        """
        Return target joint angles for 12 leg joints (LEG_JOINT_NAMES order).

        Order: [LHipYawPitch, LHipRoll, LHipPitch, LKneePitch, LAnklePitch, LAnkleRoll,
                RHipYawPitch, RHipRoll, RHipPitch, RKneePitch, RAnklePitch, RAnkleRoll]
        """
        # Defaults
        L_hip_yaw  = 0.0
        L_hip_roll = 0.0
        L_hip      = self.hp_home
        L_knee     = self.kp_home
        L_ankle    = self.ap_home
        L_ank_roll = 0.0

        R_hip_yaw  = 0.0
        R_hip_roll = 0.0
        R_hip      = self.hp_home
        R_knee     = self.kp_home
        R_ankle    = self.ap_home
        R_ank_roll = 0.0

        # ─────────────────────────────────────────────────────────────────
        # Pitch trajectories per phase
        # ─────────────────────────────────────────────────────────────────
        if self.p0 <= phase < self.p1:
            # PHASE 1: DS-shift-left, both legs at home pose
            # L is about to begin its stance push
            # R is at home, having finished its previous stance at -20° (home)
            L_hip = self.hp_home
            R_hip = self.hp_home
            L_knee = self.kp_home
            R_knee = self.kp_home
            # V3 iter 6: ankles follow the "ankle = hip" rule for any leg
            # whose foot is on the ground. Both feet down here, both hips at
            # home (-20°), so both ankles equal hip_home = -20° = ap_home.
            # The defaults (set above the phase block) already give us this,
            # so no explicit assignment needed.
            L_ankle = L_hip
            R_ankle = R_hip

        elif self.p1 <= phase < self.p2:
            # PHASE 2: L stance + R swing
            u = (phase - self.p1) / (self.p2 - self.p1)  # 0 → 1
            k = self._smooth_ramp(u)

            # L stance: monotonic push from home to (home + stance_push)
            # i.e. from -20° to -12° (less forward-flexed = body forward)
            hip_delta = (self.L_stance_end_hip - self.L_stance_start_hip) * k
            L_hip = self.L_stance_start_hip + hip_delta
            L_knee = self.kp_home  # stance knee stays at home

            # V3 iter 6: stance ankle follows the "ankle = hip" rule. As L
            # hip rotates back during stance, L ankle rotates forward by the
            # same amount, keeping the L foot flat on the ground. This is
            # the iter 5 compensation idea, expressed via the simpler rule
            # that holds for any leg whose foot is in contact with the
            # ground (foot stays parallel to the ground iff ankle = hip,
            # given knee fixed at the home value).
            L_ankle = L_hip

            # R swing: hip goes from home (-20°) to swing_land (-28°),
            # passing through max forward flexion at mid-swing
            R_hip = self._swing_hip_arc(
                u,
                start_angle=self.hp_home,
                end_angle=self.R_swing_land_hip,
                lift_amp=self.sw_hip_lift,
            )
            R_knee = self._swing_knee_arc(u)

            # V3 iter 6: R swing ankle also tracks R hip. The R foot is in
            # the air during this phase, so the in-air orientation doesn't
            # affect ground contact, but tracking the hip ensures the foot
            # is correctly oriented (flat) at touchdown — when phase 2 ends
            # and phase 3 begins, R_hip = R_swing_land_hip = -28°, so
            # R_ankle must also = -28° at landing for a flat foot.
            R_ankle = R_hip

        elif self.p2 <= phase < self.p3:
            # PHASE 3: DS-shift-right
            # L holds at end of stance (-12°), foot behind body
            # R holds at end of swing (-28°), foot forward of body
            # Both feet on ground: L behind, R ahead → split stance
            L_hip = self.L_stance_end_hip   # -12°
            R_hip = self.R_swing_land_hip   # -28°
            L_knee = self.kp_home
            R_knee = self.kp_home
            # V3 iter 6: split-stance DS, both feet on ground. Both ankles
            # follow the "ankle = hip" rule, which keeps both feet flat in
            # this split configuration. L foot at L_hip = -12° (foot behind
            # body), R foot at R_hip = -28° (foot ahead of body), both flat.
            # NOTE: this is when R lands from its phase-2 swing, so R_ankle
            # transitioning to -28° here matches the swing landing target
            # set in phase 2 above (R_ankle = R_hip throughout swing).
            L_ankle = L_hip
            R_ankle = R_hip

        elif self.p3 <= phase < self.p4:
            # PHASE 4: R stance + L swing
            u = (phase - self.p3) / (self.p4 - self.p3)  # 0 → 1
            k = self._smooth_ramp(u)

            # R stance: monotonic push from swing-land (-28°) to home (-20°)
            # Same 8° push magnitude as L stance, just in a different range.
            hip_delta = (self.R_stance_end_hip - self.R_swing_land_hip) * k
            R_hip = self.R_swing_land_hip + hip_delta
            R_knee = self.kp_home

            # V3 iter 6: R stance ankle follows R hip via the "ankle = hip"
            # rule. R_ankle ramps from -28° (start of R stance, foot just
            # landed flat with R_hip = -28°) to -20° (end of R stance, foot
            # back to home configuration). This is the symmetric mirror of
            # the L stance compensation.
            R_ankle = R_hip

            # L swing: hip goes from end-of-L-stance (-12°) back to home (-20°),
            # passing through max forward flexion at mid-swing
            L_hip = self._swing_hip_arc(
                u,
                start_angle=self.L_stance_end_hip,  # -12°
                end_angle=self.hp_home,             # -20°
                lift_amp=self.sw_hip_lift,
            )
            L_knee = self._swing_knee_arc(u)

            # V3 iter 6: L swing ankle tracks L hip throughout the swing,
            # so the foot is correctly oriented at landing. At the end of
            # phase 4, L_hip = hp_home = -20°, so L_ankle also = -20° at
            # the cycle wrap into phase 1 of the next cycle.
            L_ankle = L_hip

        # ─────────────────────────────────────────────────────────────────
        # Hip roll: lateral weight transfer with decay-during-swing
        # (copied verbatim from V2 iter 2 — proven stable)
        # ─────────────────────────────────────────────────────────────────
        if self.p0 <= phase < self.p1:
            # PHASE 1: DS-shift-left, ramp roll from 0 → +amp
            u = (phase - self.p0) / (self.p1 - self.p0)
            if self.lateral_decay_during_swing:
                k = self._smooth_ramp(u)
                roll = self.hip_roll_amp * k
            else:
                k = self._smooth_ramp(u)
                roll = self.hip_roll_amp * (-1.0 + 2.0 * k)
        elif self.p1 <= phase < self.p2:
            # PHASE 2: SS-left, decay roll from +amp → 0
            u = (phase - self.p1) / (self.p2 - self.p1)
            if self.lateral_decay_during_swing:
                k = self._smooth_ramp(u)
                roll = self.hip_roll_amp * (1.0 - k)
            else:
                roll = self.hip_roll_amp
        elif self.p2 <= phase < self.p3:
            # PHASE 3: DS-shift-right, ramp roll to -amp
            u = (phase - self.p2) / (self.p3 - self.p2)
            k = self._smooth_ramp(u)
            if self.lateral_decay_during_swing:
                roll = -self.hip_roll_amp * k
            else:
                roll = self.hip_roll_amp * (1.0 - 2.0 * k)
        else:
            # PHASE 4: SS-right, decay roll from -amp → 0
            u = (phase - self.p3) / (self.p4 - self.p3)
            if self.lateral_decay_during_swing:
                k = self._smooth_ramp(u)
                roll = -self.hip_roll_amp * (1.0 - k)
            else:
                roll = -self.hip_roll_amp

        # ─────────────────────────────────────────────────────────────────
        # V3 iter 4: gait init blend (extended in iter 5 to cover ankles)
        # During the first gait_init_blend_steps calls, smoothly interpolate
        # the roll command and the ankle pitch commands from their warmup
        # values (roll=0, ankles=ap_home) toward whatever the steady-state
        # phase logic computed. This eliminates instantaneous jumps at the
        # start of every gait.
        #
        # iter 6: the ankle blend has been removed. With the corrected
        # "ankle = hip" rule, all ankles equal -20° (home) at the gait
        # start (phase 0.075, in DS-shift-left, both hips at home), so
        # there is no discontinuity to fade out. The roll blend is kept
        # as a defensive no-op in case phase_offset is changed later.
        #
        # After gait_init_blend_steps calls, blend_factor reaches 1.0 and
        # we use the steady-state values directly with no overhead.
        # ─────────────────────────────────────────────────────────────────
        if self._calls_since_init < self.gait_init_blend_steps:
            u = self._calls_since_init / float(self.gait_init_blend_steps)
            blend_factor = self._smooth_ramp(u)  # 0 → 1 with smooth endpoints
            roll = roll * blend_factor
        self._calls_since_init += 1

        L_hip_roll =  roll
        R_hip_roll = -roll
        L_ank_roll = -roll
        R_ank_roll =  roll

        targets = np.array([
            L_hip_yaw, L_hip_roll, L_hip,
            L_knee,    L_ankle,    L_ank_roll,
            R_hip_yaw, R_hip_roll, R_hip,
            R_knee,    R_ankle,    R_ank_roll,
        ], dtype=np.float32)

        return targets


class ReferenceGaitV4:
    """
    Persistent-landed-pose walking gait — V4.

    DESIGN PHILOSOPHY (the V3 lessons applied)
    -------------------------------------------
    V2 iter 2 was provably stable: thousands of steps without falling, body
    height oscillating in a tiny ±3mm range. But it produced ~0 forward
    motion because every joint trajectory was time-symmetric: the stance
    push went up and back down, the swing landed forward but was immediately
    pulled back to home. Net displacement: zero.

    V3 introduced a monotonic stance push to break the symmetry. It produced
    real forward translation in the data (iter 6/7 logs show the body
    actually moves forward during stance) but the push amplitudes large
    enough to cause meaningful forward motion (4-8°) also generated reaction
    torques that tipped the body backward in single support, AND the long
    single-support phases let lateral CoM drift accumulate fatally.

    V4 takes a different angle: keep V2's small, stable amplitudes, but
    eliminate the symmetry by having each leg ALTERNATE between two
    distinct steady-state positions (a "landed forward" position and a
    "retreated" position) instead of always returning to a single home.
    The cycle is fully periodic, but per leg, the sequence is:

      landed (-22°)
        → stance retreat to -24° over phase 2 (or 4)
        → DS hold at -24°
        → swing back forward to -22° (landed)
        → DS hold at -22°
        → next stance retreat...

    The joint moves only ±2° from a small range, the stance push is small
    enough (2°) that it should not generate body-tipping torque, and the
    overall lateral handling is identical to V2 iter 2 (proven stable).
    Forward translation per cycle is small (~1.4 cm geometric estimate)
    but real and consistent.

    PER-LEG STEADY-STATE TRAJECTORY (iter 2 corrected)
    ---------------------------------------------------
    Each leg cycles through:
      hip_landed  = MORE flexed (foot ahead of body) -- e.g. -24°
      hip_retreat = LESS flexed (body has translated forward over the foot)
                    -- e.g. -20° (= home, when stance_retreat == landed_offset)

    L leg trajectory over one cycle:
      Phase 1 [DS-shift-left]:    hold at hip_landed  (-24°, foot ahead of body)
      Phase 2 [L stance, R swing]:ramp hip_landed → hip_retreat (-24° → -20°)
                                  L hip becomes LESS flexed; the planted L
                                  foot stays in space; the body translates
                                  FORWARD over the L foot (~1.2 cm)
      Phase 3 [DS-shift-right]:   hold at hip_retreat (-20°, foot under body)
      Phase 4 [R stance, L swing]:swing hip_retreat → mid-swing peak → hip_landed
                                  (-20° → -26° → -24°, foot lifts ~1.9cm and
                                  moves forward in space)
      Cycle wrap → Phase 1: hold at hip_landed (-24°). ✓ continuous.

    R leg is mirror-offset by half cycle:
      Phase 1 [DS-shift-left]:    hold at hip_retreat (-20°)
      Phase 2 [L stance, R swing]:swing hip_retreat → mid → hip_landed
                                  (-20° → -26° → -24°)
      Phase 3 [DS-shift-right]:   hold at hip_landed (-24°)
      Phase 4 [R stance, L swing]:ramp hip_landed → hip_retreat (-24° → -20°)

    HOW FORWARD TRANSLATION HAPPENS
    --------------------------------
    Each "stance retreat" rotates the loaded leg's hip from -24° (more
    flexed) to -20° (home, less flexed). The thigh tilts BACKWARD relative
    to the body by 4°. With the foot planted by friction, the body must
    rotate forward to compensate, translating FORWARD over the foot by
    approximately:

      foot_x_at_hip(-24°) - foot_x_at_hip(-20°) ≈ 1.26 - 0.07 ≈ 1.19 cm

    Per cycle (2 stances): ~2.4 cm forward. Slow but well above the
    "≥1 cm/cycle" target. The swing leg meanwhile lifts the OTHER foot
    1.9 cm at peak swing and places it 1.3 cm forward of where it was,
    so the body has somewhere to step onto for the next half-cycle.

    iter 1 used the OPPOSITE semantics (hip_landed less flexed than
    hip_retreat), which made the stance "retreat" rotate the hip MORE
    flexed and translated the body BACKWARD. iter 1's data showed -2.6
    cm net X drift over 1300 steps, confirming the bug. iter 2 fixes
    this by inverting the formula: hip_retreat = hip_landed + retreat
    (instead of - retreat), so that retreat is LESS flexed than landed.

    ANKLE COMPENSATION
    ------------------
    Same rule as V3 iter 6+: ankle = hip whenever the foot is on the ground
    (and during swing too, so the foot is correctly oriented at landing).
    This keeps the foot flat throughout the cycle, avoiding the heel/toe
    pivoting that broke V3 iter 5.

    INIT BLEND
    ----------
    The V4 steady state has both legs in non-home positions (-22° and -24°),
    while the warmup pose has both legs at -20° (home). There is no phase
    in the cycle where both legs are at home, so we can't avoid this
    discontinuity with phase offsetting. Instead, an init blend (same
    mechanism as V3 iter 4) smoothly interpolates each commanded joint
    from its warmup value (home) toward its steady-state value over the
    first 60 calls (2 seconds at 30Hz). After that, the gait runs at full
    steady state.
    """

    def __init__(
        self,
        gait_hz=0.05,                        # 20-second cycle
        # Home pose (matches NaoWalkingEnv.LEG_HOME_POSE_DEG)
        hip_pitch_home_deg=-20.0,
        knee_pitch_home_deg=40.0,
        ankle_pitch_home_deg=-20.0,
        # ── V4 cycle structure ────────────────────────────────────────────
        # iter 1: 2°/2° offsets, 12° knee lift, 4° hip amp → STABLE but no
        #   visible foot lift (only 8mm clearance) and inverted formula
        #   caused -2.6cm X drift (backward bug).
        # iter 2: 4°/4° offsets, 24° knee lift, 4° hip amp, formula corrected
        #   → FELL sideways at step 460. Hip excursions caused diagonal
        #   stance + leftward drift.
        # iter 3: 2°/2° offsets, 24° knee lift, 4° hip amp → FELL FORWARD
        #   at step 240, exactly when R knee bend hit its peak (64°).
        #
        # Diagnosis (iter 3): the destabilization is NOT static CoM shift
        # (computed leg-CoM shift from knee bend is < 1cm whole-body, well
        # within the foot's support polygon). It's the REACTION TORQUE
        # generated by the swing leg's angular acceleration. By Newton's
        # third law, fast-bending knee → fast shin rotation → reaction
        # torque on the thigh → propagates through hip to body → body
        # tips forward over the L stance foot. iter 3 doubled the knee
        # angular velocity vs iter 1 (24° amplitude vs 12° over the same
        # swing duration), doubling the reaction torque, which exceeded
        # iter 1's tolerance margin.
        #
        # iter 4 (this version): find the largest knee lift that keeps the
        # *combined* swing-leg reaction torque at or below iter 1's level,
        # while still giving visibly more foot lift than iter 1's 8mm.
        # Two coupled changes:
        #   - knee_lift 24° → 16°: gives 1.18cm foot lift (50% more than
        #     iter 1's 0.83cm), at 1.33× iter 1's knee angular velocity.
        #   - sw_hip_amp 4° → 2°: cuts hip-related reaction torque in half
        #     (from 1× iter 1 to 0.5× iter 1). Has negligible effect on
        #     foot lift (~0.04cm) because the knee dominates lift geometry.
        # Combined estimated reaction torque: 0.33 × 1.0 (knee delta) +
        #   1.0 × 0.5 (hip delta saving) ≈ at-or-below iter 1's total.
        # iter 4: 2°/2° offsets, 16° knee lift, 2° hip amp → STABLE with
        #   visible foot lift (1.22cm) but ZERO net forward motion
        #   (+0.03 cm/cycle mean, oscillating near ±0.5 cm).
        #
        # Diagnosis of iter 4's zero-forward-motion: the L stance hip shows
        # a clear pattern in phase 2 where the actual angle moves OPPOSITE
        # to the command during the R swing window. At step 880 (mid-swing),
        # L_hip actual = -20.1° (0.5° ahead of -20.6° command). At step 940
        # (late swing), actual = -21.4° (1.4° behind -20.0° command). The
        # L hip moved 1.3° MORE FLEXED while the command went 0.6° LESS
        # FLEXED. The R swing leg's motion generates reaction torque on the
        # body that cancels the intended L stance retreat.
        #
        # With only 2° of commanded stance retreat, the cancellation fully
        # dominates and net body translation per cycle is essentially zero.
        # The hypothesis: larger commanded amplitude will push through the
        # disturbance and produce net motion.
        #
        # iter 5: stance amplitudes 2° → 3° (halfway between iter 4's zero-
        # motion and iter 2's 4° which made forward progress before failing
        # laterally). Everything else kept at iter 4's known-stable values.
        # Predicted forward: ~1.0 cm/cycle (if the "push through disturbance"
        # hypothesis holds). Stability: should be OK because 3° is still
        # below iter 2's 4° that triggered the lateral diagonal-stance
        # failure, and swing amplitudes are unchanged from stable iter 4.
        landed_offset_deg=3.0,    # iter 4 was 2.0 → iter 5 bump to 3.0
        stance_retreat_deg=3.0,   # iter 4 was 2.0 → iter 5 bump to 3.0
        # ── Swing arc ─────────────────────────────────────────────────────
        # iter 4: reduced from 4° to 2°. Has negligible effect on foot lift
        # (the knee bend dominates that geometry) but cuts hip-rotation
        # reaction torque in half, freeing up "reaction torque budget" for
        # the larger knee bend.
        swing_hip_amp_deg=2.0,
        # iter 4: precision-chosen to give visibly larger foot lift (1.18cm
        # at peak, 42% more than iter 1) while keeping the knee-related
        # reaction torque at 1.33× iter 1's value. Combined with the
        # halved hip amp above, total reaction torque should be ~iter 1's.
        swing_knee_lift_deg=16.0,
        # ── Lateral weight transfer ───────────────────────────────────────
        # Identical to V2 iter 2's stable configuration. Small lean, decay
        # during single support so the CoM doesn't over-commit.
        hip_roll_amp_deg=4.0,
        lateral_decay_during_swing=True,
        # ── Phase timing ──────────────────────────────────────────────────
        # iter 5 analysis: the stance pushes DO work (P2 delivers +1.3cm,
        # P4 delivers +1.1cm of real forward motion). BUT the non-stance
        # phases (P1 DS-shift-left, P3 DS-shift-right, cycle boundaries)
        # drag the body backward by ~2.5cm per cycle, canceling the
        # stance gains. Net forward motion ≈ 0.
        #
        # iter 6: shorten DS phases (0.15 → 0.08) to expand SS phases
        # (0.35 → 0.42 each). More active stance-push time, less hold-time
        # for backward drift to accumulate. Expected forward motion:
        # 0.3-1.0 cm/cycle (small but above noise).
        #
        # RISK: shorter DS means less time for lateral weight transfer.
        # V2 iter 2 used 0.15 because that was the "proven stable" value.
        # If this triggers lateral drift, we'll see it in the iter 6 log
        # and revert.
        double_support_frac=0.08,
        # ── Stance ankle lag compensation (iter 7) ────────────────────────
        # iter 6 analysis: during L stance in P2, the hip PD lags the command
        # by 1.5-2.2° (actual hip is more flexed than cmd due to body weight
        # loading). Because the ankle command was set equal to the HIP
        # COMMAND (L_ankle = L_hip), the actual ankle ended up at ~cmd, but
        # the actual hip was 2° MORE flexed. This made the shin tilt 2°
        # further forward than the ankle was prepared for → foot tipped
        # onto its heel (toe up by ~2°) → only heel in contact → low
        # friction → L foot SLID by ~11 mm during stance → most of the
        # geometric forward translation was lost to sliding.
        #
        # Fix: bias the stance-leg ankle command to be more plantarflexed
        # (more negative) than the commanded hip by this amount. When the
        # actual hip lags by ~this amount, the foot lies flat.
        #
        # Applied only to the STANCE leg in each phase (not to swing legs,
        # which aren't load-bearing and don't have hip lag).
        stance_ankle_lag_comp_deg=2.0,
        # ── Init blend ────────────────────────────────────────────────────
        # Smoothly interpolate from warmup pose (all legs at home) toward
        # the V4 steady-state asymmetric pose over the first N calls.
        gait_init_blend_steps=60,
    ):
        self.gait_hz = gait_hz

        # Convert home pose to radians once
        self.hp_home = math.radians(hip_pitch_home_deg)
        self.kp_home = math.radians(knee_pitch_home_deg)
        self.ap_home = math.radians(ankle_pitch_home_deg)

        # Cycle structure parameters in radians
        self.landed_offset = math.radians(landed_offset_deg)
        self.stance_retreat = math.radians(stance_retreat_deg)
        self.sw_hip_amp = math.radians(swing_hip_amp_deg)
        self.sw_knee_lift = math.radians(swing_knee_lift_deg)
        # iter 7: stance ankle lag compensation (applied to stance legs only)
        self.stance_ankle_lag_comp = math.radians(stance_ankle_lag_comp_deg)

        # Pre-computed steady-state hip values (iter 2 — corrected semantics):
        # hip_landed  = MORE flexed: foot is forward of body, where the leg
        #               lands at the end of a swing (e.g. -24°)
        # hip_retreat = LESS flexed: foot is closer to under-body, the body
        #               has translated forward over the foot during stance
        #               (e.g. -20° = home, when stance_retreat == landed_offset)
        # Stance trajectory: hip_landed → hip_retreat (less flexed → body forward)
        # Swing trajectory: hip_retreat → hip_landed (more flexed → foot moves forward)
        self.hip_landed = self.hp_home - self.landed_offset
        self.hip_retreat = self.hip_landed + self.stance_retreat

        self.hip_roll_amp = math.radians(hip_roll_amp_deg)
        self.lateral_decay_during_swing = lateral_decay_during_swing

        # Phase boundaries
        self.ds_frac = double_support_frac
        self.p0 = 0.0
        self.p1 = self.ds_frac                   # 0.15
        self.p2 = 0.50
        self.p3 = 0.50 + self.ds_frac            # 0.65
        self.p4 = 1.0

        # Init blend state
        self.gait_init_blend_steps = int(gait_init_blend_steps)
        self._calls_since_init = 0

    def get_phase(self, t):
        return (t * self.gait_hz) % 1.0

    def get_phase_signal(self, phase):
        angle = 2.0 * math.pi * phase
        return math.sin(angle), math.cos(angle)

    def reset_init_blend(self):
        """Reset init blend so the next call starts the smooth fade-in again."""
        self._calls_since_init = 0

    @staticmethod
    def _smooth_ramp(u):
        """Smooth ramp from 0 to 1 with zero derivative at endpoints."""
        if u < 0.0: u = 0.0
        if u > 1.0: u = 1.0
        return 0.5 * (1.0 - math.cos(math.pi * u))

    @staticmethod
    def _smooth_bump(u):
        """
        C¹-smooth bump: 0 at u=0, 1 at u=0.5, 0 at u=1, with zero derivative
        at both endpoints.

        Note: V2 and V3 used sin(πu), which is C⁰ at endpoints (slope = ±π
        at u=0, u=1). That created small velocity jumps at phase transitions
        where a swing arc started or ended at u≈0 or u≈1. V4 uses sin²(πu) =
        (1 - cos(2πu))/2 instead, which has slope 0 at both endpoints and
        joins smoothly with the held endpoint values during DS phases.
        """
        if u < 0.0: u = 0.0
        if u > 1.0: u = 1.0
        s = math.sin(math.pi * u)
        return s * s

    def _swing_hip_arc(self, u, start_angle, end_angle):
        """
        Compute swing-leg hip pitch over swing progress u in [0, 1].

        At u=0: hip is at start_angle (= retreated position from end of stance)
        At u=0.5: hip is at MAX FORWARD FLEXION = min(start, end) - sw_hip_amp
        At u=1: hip is at end_angle (= landed forward position)

        Linear interpolation between start and end, plus a smooth bump
        pulling the hip more forward (more negative) at mid-swing.
        """
        k = self._smooth_ramp(u)
        baseline = start_angle + (end_angle - start_angle) * k
        bump = self._smooth_bump(u)
        return baseline - self.sw_hip_amp * bump

    def _swing_knee_arc(self, u):
        """
        Compute swing-leg knee pitch over swing progress u.
        Returns to home at both endpoints, lifts at mid-swing.
        """
        bump = self._smooth_bump(u)
        return self.kp_home + self.sw_knee_lift * bump

    def get_targets(self, phase):
        """
        Return target joint angles for 12 leg joints (LEG_JOINT_NAMES order).

        Order: [LHipYawPitch, LHipRoll, LHipPitch, LKneePitch, LAnklePitch, LAnkleRoll,
                RHipYawPitch, RHipRoll, RHipPitch, RKneePitch, RAnklePitch, RAnkleRoll]
        """
        # Defaults — every joint at home
        L_hip_yaw  = 0.0
        L_hip_roll = 0.0
        L_hip      = self.hp_home
        L_knee     = self.kp_home
        L_ankle    = self.ap_home
        L_ank_roll = 0.0

        R_hip_yaw  = 0.0
        R_hip_roll = 0.0
        R_hip      = self.hp_home
        R_knee     = self.kp_home
        R_ankle    = self.ap_home
        R_ank_roll = 0.0

        # ─────────────────────────────────────────────────────────────────
        # Hip pitch / knee trajectories per phase
        # ─────────────────────────────────────────────────────────────────
        if self.p0 <= phase < self.p1:
            # PHASE 1: DS-shift-left
            # L just landed (from previous cycle's swing) at hip_landed
            # R just finished its stance at hip_retreat
            L_hip = self.hip_landed
            R_hip = self.hip_retreat
            L_knee = self.kp_home
            R_knee = self.kp_home

        elif self.p1 <= phase < self.p2:
            # PHASE 2: L stance + R swing
            u = (phase - self.p1) / (self.p2 - self.p1)  # 0 → 1
            k = self._smooth_ramp(u)

            # L stance: monotonic motion from hip_landed (more flexed,
            # foot ahead of body) to hip_retreat (less flexed, body has
            # translated forward over the foot). Example with iter 2
            # defaults: -24° → -20°. The 4° hip rotation while the L foot
            # is planted translates the body forward by ~1.2 cm.
            L_hip = self.hip_landed + (self.hip_retreat - self.hip_landed) * k
            L_knee = self.kp_home

            # R swing: hip from hip_retreat (start, less flexed)
            #          → mid-swing peak (more flexed, foot lifted)
            #          → hip_landed (end, more flexed, foot now ahead of body)
            # Example with iter 2 defaults: -20° → -26° → -24°.
            R_hip = self._swing_hip_arc(
                u,
                start_angle=self.hip_retreat,  # -20° (= home)
                end_angle=self.hip_landed,     # -24°
            )
            R_knee = self._swing_knee_arc(u)

        elif self.p2 <= phase < self.p3:
            # PHASE 3: DS-shift-right
            # L finished its stance at hip_retreat
            # R just landed at hip_landed
            L_hip = self.hip_retreat
            R_hip = self.hip_landed
            L_knee = self.kp_home
            R_knee = self.kp_home

        elif self.p3 <= phase < self.p4:
            # PHASE 4: R stance + L swing
            u = (phase - self.p3) / (self.p4 - self.p3)  # 0 → 1
            k = self._smooth_ramp(u)

            # R stance: monotonic motion from hip_landed to hip_retreat
            # (mirror of L stance in phase 2). Example: -24° → -20°.
            R_hip = self.hip_landed + (self.hip_retreat - self.hip_landed) * k
            R_knee = self.kp_home

            # L swing: hip_retreat (less flexed, where L was after phase-2
            #          stance) → mid-swing peak → hip_landed (more flexed,
            #          foot moved forward).
            # Example with iter 2 defaults: -20° → -26° → -24°.
            L_hip = self._swing_hip_arc(
                u,
                start_angle=self.hip_retreat,  # -24°
                end_angle=self.hip_landed,     # -22°
            )
            L_knee = self._swing_knee_arc(u)

        # ─────────────────────────────────────────────────────────────────
        # Ankles: ankle = hip rule (V3 iter 6+), with iter 7 stance lag
        # compensation.
        #
        # The flat-foot geometric rule is ankle_pitch = hip_pitch (when
        # knee is at its home value). This holds for the commanded angles
        # in an idealized world. But in simulation, the STANCE leg's hip
        # lags the command by ~2° because the body's weight pushes the
        # hip to be more flexed than commanded. If we set ankle_cmd =
        # hip_cmd, the actual ankle tracks cmd but the actual hip is 2°
        # more flexed → foot tilts toe-up → only heel contact → sliding.
        #
        # Iter 7 fix: set ankle_cmd = hip_cmd - lag_comp (more plantarflexed
        # by ~2°) UNIFORMLY for both legs in all phases. When a stance leg's
        # hip actually lags by 2°, its foot lies flat. When a swing leg is
        # in the air, its foot is commanded 2° toe-down, which causes a
        # toe-first landing (close to natural walking) rather than an
        # inverted heel-first landing.
        #
        # Applying compensation uniformly (rather than only to stance legs)
        # avoids discontinuities at the SS↔DS phase transitions, where a
        # per-leg stance/swing assignment would flip and create a sudden
        # 2° step change in the ankle command.
        # ─────────────────────────────────────────────────────────────────
        L_ankle = L_hip - self.stance_ankle_lag_comp
        R_ankle = R_hip - self.stance_ankle_lag_comp

        # ─────────────────────────────────────────────────────────────────
        # Hip roll: identical to V2 iter 2 stable configuration
        # (small lean, decay during single support)
        # ─────────────────────────────────────────────────────────────────
        if self.p0 <= phase < self.p1:
            # PHASE 1: DS-shift-left, ramp roll from 0 → +amp
            u = (phase - self.p0) / (self.p1 - self.p0)
            if self.lateral_decay_during_swing:
                k = self._smooth_ramp(u)
                roll = self.hip_roll_amp * k
            else:
                k = self._smooth_ramp(u)
                roll = self.hip_roll_amp * (-1.0 + 2.0 * k)
        elif self.p1 <= phase < self.p2:
            # PHASE 2: SS-left, decay roll from +amp → 0 (decay mode)
            u = (phase - self.p1) / (self.p2 - self.p1)
            if self.lateral_decay_during_swing:
                k = self._smooth_ramp(u)
                roll = self.hip_roll_amp * (1.0 - k)
            else:
                roll = self.hip_roll_amp
        elif self.p2 <= phase < self.p3:
            # PHASE 3: DS-shift-right, ramp roll to -amp
            u = (phase - self.p2) / (self.p3 - self.p2)
            k = self._smooth_ramp(u)
            if self.lateral_decay_during_swing:
                roll = -self.hip_roll_amp * k
            else:
                roll = self.hip_roll_amp * (1.0 - 2.0 * k)
        else:
            # PHASE 4: SS-right, decay roll from -amp → 0
            u = (phase - self.p3) / (self.p4 - self.p3)
            if self.lateral_decay_during_swing:
                k = self._smooth_ramp(u)
                roll = -self.hip_roll_amp * (1.0 - k)
            else:
                roll = -self.hip_roll_amp

        # ─────────────────────────────────────────────────────────────────
        # Init blend: fade from warmup pose (all legs at home, roll 0)
        # toward the V4 steady-state command over the first N calls.
        # The blend factor goes 0 → 1 with smooth endpoints, and is
        # applied per-joint as: cmd = home + (cmd - home) * blend_factor.
        # At call 0: cmd = home (matches warmup). At call >= N: cmd = cmd.
        # ─────────────────────────────────────────────────────────────────
        if self._calls_since_init < self.gait_init_blend_steps:
            u_blend = self._calls_since_init / float(self.gait_init_blend_steps)
            blend_factor = self._smooth_ramp(u_blend)
            roll = roll * blend_factor
            L_hip = self.hp_home + (L_hip - self.hp_home) * blend_factor
            R_hip = self.hp_home + (R_hip - self.hp_home) * blend_factor
            L_knee = self.kp_home + (L_knee - self.kp_home) * blend_factor
            R_knee = self.kp_home + (R_knee - self.kp_home) * blend_factor
            # Re-derive ankles using the iter 7 uniform stance-lag
            # compensation rule, with the compensation amount scaled by
            # blend_factor so that at call 0 ankles match warmup (no
            # compensation, ankle = hip = home) and at call N they match
            # the steady-state compensated value.
            comp_blended = self.stance_ankle_lag_comp * blend_factor
            L_ankle = L_hip - comp_blended
            R_ankle = R_hip - comp_blended
        self._calls_since_init += 1

        L_hip_roll =  roll
        R_hip_roll = -roll
        L_ank_roll = -roll
        R_ank_roll =  roll

        targets = np.array([
            L_hip_yaw, L_hip_roll, L_hip,
            L_knee,    L_ankle,    L_ank_roll,
            R_hip_yaw, R_hip_roll, R_hip,
            R_knee,    R_ankle,    R_ank_roll,
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
        action_scale=0.2,
        finger_kp=100.0,
        finger_kd=10.0,
        gait_hz=0.2,
        gait_warmup_steps=60,
        gait_lateral_only=False,
        gait_version="v1",   # "v1"=ReferenceGait, "v2"=ReferenceGaitV2, "v3"=ReferenceGaitV3, "v4"=ReferenceGaitV4
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
        self.gait_version      = gait_version

        self.w_forward      = reward_forward_vel
        self.w_alive        = reward_alive
        self.w_orient       = penalty_orientation
        self.alive_scale    = alive_scale
        self.term_height    = term_height
        self.term_orient_qw = term_orient_qw
        self.grace_period   = grace_period

        self.n_legs = len(self.LEG_JOINT_NAMES)
        if gait_version == "v4":
            # V4: persistent-landed-pose walking gait. Same gait_hz convention.
            if gait_hz == 0.2:
                self._gait = ReferenceGaitV4()  # use V4 class default (0.05)
            else:
                self._gait = ReferenceGaitV4(gait_hz=gait_hz)
            self.gait_hz = self._gait.gait_hz
        elif gait_version == "v3":
            # V3: monotonic-stance walking gait. Same gait_hz convention as V2
            # (treat V1 default 0.2 as "user didn't override" and use V3 default).
            if gait_hz == 0.2:
                self._gait = ReferenceGaitV3()  # use V3 class default (0.05)
            else:
                self._gait = ReferenceGaitV3(gait_hz=gait_hz)
            self.gait_hz = self._gait.gait_hz
        elif gait_version == "v2":
            # V2: active-stance forward-stepping gait. Iter 2 default cycle
            # is 0.05 Hz (20s). When the user passes the V1 default of 0.2,
            # we treat it as "not specified" and let V2's class default apply.
            # When the user explicitly passes any other value, honor it.
            if gait_hz == 0.2:  # V1 default — user didn't override
                self._gait = ReferenceGaitV2()  # use V2 class default (0.05)
            else:
                self._gait = ReferenceGaitV2(gait_hz=gait_hz)
            self.gait_hz = self._gait.gait_hz
            # Note: V2 does not support lateral_only diagnostic mode.
        else:
            self._gait = ReferenceGait(gait_hz=gait_hz, lateral_only=gait_lateral_only)
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
        if self.gait_version == "v4":
            print(f"[NaoWalkingEnv] ReferenceGaitV4 — persistent-landed-pose walking")
            print(f"[NaoWalkingEnv] Gait: {self.gait_hz}Hz (period={1.0/self.gait_hz:.1f}s), "
                  f"warmup: {self.gait_warmup_steps} steps")
            print(f"[NaoWalkingEnv] Hip landed: {math.degrees(self._gait.hip_landed):.1f}deg, "
                  f"Hip retreat: {math.degrees(self._gait.hip_retreat):.1f}deg "
                  f"(stance retreat: {math.degrees(self._gait.stance_retreat):.1f}deg)")
            print(f"[NaoWalkingEnv] Swing knee lift: {math.degrees(self._gait.sw_knee_lift):.1f}deg, "
                  f"Swing hip amp: {math.degrees(self._gait.sw_hip_amp):.1f}deg")
            print(f"[NaoWalkingEnv] Hip roll amp: {math.degrees(self._gait.hip_roll_amp):.1f}deg, "
                  f"Double support frac: {self._gait.ds_frac:.2f}, "
                  f"Init blend: {self._gait.gait_init_blend_steps} steps")
        elif self.gait_version == "v3":
            print(f"[NaoWalkingEnv] ReferenceGaitV3 — monotonic-stance walking")
            print(f"[NaoWalkingEnv] Gait: {self.gait_hz}Hz (period={1.0/self.gait_hz:.1f}s), "
                  f"warmup: {self.gait_warmup_steps} steps")
            print(f"[NaoWalkingEnv] Stance push amp: {math.degrees(self._gait.stance_push):.1f}deg "
                  f"(L: {math.degrees(self._gait.L_stance_start_hip):.1f}→{math.degrees(self._gait.L_stance_end_hip):.1f}, "
                  f"R: {math.degrees(self._gait.R_swing_land_hip):.1f}→{math.degrees(self._gait.R_stance_end_hip):.1f})")
            print(f"[NaoWalkingEnv] Swing knee lift: {math.degrees(self._gait.sw_knee_lift):.1f}deg, "
                  f"Swing hip lift: {math.degrees(self._gait.sw_hip_lift):.1f}deg")
            print(f"[NaoWalkingEnv] Hip roll amp: {math.degrees(self._gait.hip_roll_amp):.1f}deg, "
                  f"Double support frac: {self._gait.ds_frac:.2f}")
        elif self.gait_version == "v2":
            print(f"[NaoWalkingEnv] ReferenceGaitV2 — active-stance forward-stepping")
            print(f"[NaoWalkingEnv] Gait: {self.gait_hz}Hz (period={1.0/self.gait_hz:.1f}s), "
                  f"warmup: {self.gait_warmup_steps} steps")
            print(f"[NaoWalkingEnv] Swing hip amp: {math.degrees(self._gait.sw_hip_amp):.1f}deg, "
                  f"Swing knee lift: {math.degrees(self._gait.sw_knee_lift):.1f}deg, "
                  f"Swing land offset: {math.degrees(self._gait.sw_hip_land):.1f}deg")
            print(f"[NaoWalkingEnv] Stance hip push: {math.degrees(self._gait.st_hip_push):.1f}deg, "
                  f"Stance knee extend: {math.degrees(self._gait.st_knee_extend):.1f}deg, "
                  f"Stance ankle push-off: {math.degrees(self._gait.st_ankle_push):.1f}deg")
            print(f"[NaoWalkingEnv] Hip roll amp: {math.degrees(self._gait.hip_roll_amp):.1f}deg, "
                  f"Double support frac: {self._gait.ds_frac:.2f}")
        else:
            print(f"[NaoWalkingEnv] test_16 — alternating gait + ankle push-off{lat}")
            print(f"[NaoWalkingEnv] Gait: {self.gait_hz}Hz, warmup: {self.gait_warmup_steps} steps")
            print(f"[NaoWalkingEnv] Hip amp: {math.degrees(self._gait.hip_pitch_amp):.1f}deg, "
                  f"Knee amp: {math.degrees(self._gait.knee_pitch_amp):.1f}deg, "
                  f"Ankle push-off: {math.degrees(self._gait.ankle_pushoff_amp):.1f}deg")

    def _apply_gains(self):
        """
        Three-tier PD gains. Called after every world.reset().

        Tier 1 - Body/arm joints:           kp=35810, kd=500
        Tier 2 - Hips + Knees + AnklePitch: kp=300k,  kd=3000  (load-bearing)
        Tier 3 - Fingers/wrists:            kp=100,   kd=10

        NOTE: AnklePitch was moved to Tier 2 to fix a tracking-lag issue.
        With the ReferenceGaitV2 stance push (which commands the ankle to
        plantarflex while the hip and knee rotate), an 8.4x gain mismatch
        between hip/knee (300k) and ankle (35k) caused the ankle to lag,
        which kinematically rotated the torso BACKWARD instead of forward
        during the stance push — making the robot fall on its back. With
        the ankle at the same gain level, all three joints track in sync
        and the coordinated stance push moves the torso forward as intended.
        AnkleRoll is kept at Tier 1 because it doesn't need to track as
        aggressively (it's used for lateral balance, not forward propulsion).
        """
        high_gain_set = {
            "LHipYawPitch", "LHipRoll", "LHipPitch",
            "RHipYawPitch", "RHipRoll", "RHipPitch",
            "LKneePitch",   "RKneePitch",    # knees — hold crouch under load
            "LAnklePitch",  "RAnklePitch",   # ankles — must track stance push
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

        # V3 iter 4: reset the gait's init blend so each episode begins
        # with a smooth lateral transition (not just the very first one).
        # V1/V2 don't have this method, so guard with hasattr.
        if hasattr(self._gait, "reset_init_blend"):
            self._gait.reset_init_blend()

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
            gait_target = self._gait.get_targets(self._phase)

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
