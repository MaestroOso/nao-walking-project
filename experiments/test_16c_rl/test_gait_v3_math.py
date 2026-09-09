"""
Standalone sanity check for ReferenceGaitV3.

V3 is structurally different from V2:
  - Stance is MONOTONIC (no bump, no return)
  - Each leg's hip oscillates between two endpoints, with the swing reset
    happening in the air
  - At phase 0.0, both legs are at home pose (-20°)
  - L stance pushes -20° → -12°, R stance pushes -28° → -20°
  - Both stances produce the same body forward translation
"""
import sys, math
import numpy as np

# Stub gymnasium so we can import the file without Isaac Sim
class _StubBox:
    def __init__(self, *a, **k): pass
class _StubSpaces:
    Box = _StubBox
class _StubGym:
    class Env: pass
    spaces = _StubSpaces
sys.modules['gymnasium'] = _StubGym()
sys.modules['gymnasium.spaces'] = _StubSpaces

sys.path.insert(0, '.')
from nao_walking_env import ReferenceGaitV3

# Steady-state gait instance (no init blend) — used for tests 1-13.
# We disable the blend here because tests 11/11b check exact steady-state
# roll values, and the blend would scale them down on the early calls.
# A separate test (14) verifies the blend behavior on its own.
g = ReferenceGaitV3(gait_init_blend_steps=0)

LEG_NAMES = [
    "LHipYawPitch", "LHipRoll", "LHipPitch",
    "LKneePitch", "LAnklePitch", "LAnkleRoll",
    "RHipYawPitch", "RHipRoll", "RHipPitch",
    "RKneePitch", "RAnklePitch", "RAnkleRoll",
]

def show(phase, label):
    t = g.get_targets(phase)
    deg = [math.degrees(x) for x in t]
    print(f"\n=== phase={phase:.3f} — {label} ===")
    for i, name in enumerate(LEG_NAMES):
        print(f"  {name:<14} {deg[i]:+7.2f}°")

print("V3 GAIT — MONOTONIC STANCE WALKING")
print("="*60)
print("Home pose: HipPitch=-20°, KneePitch=+40°, AnklePitch=-20°")
print(f"Stance push amp: {math.degrees(g.stance_push):.1f}°")
print(f"L stance: {math.degrees(g.L_stance_start_hip):+.1f}° → "
      f"{math.degrees(g.L_stance_end_hip):+.1f}°")
print(f"R stance: {math.degrees(g.R_swing_land_hip):+.1f}° → "
      f"{math.degrees(g.R_stance_end_hip):+.1f}°")
print(f"Swing knee lift: {math.degrees(g.sw_knee_lift):.1f}°")
print(f"Swing hip lift:  {math.degrees(g.sw_hip_lift):.1f}°")
print(f"Hip roll amp:    {math.degrees(g.hip_roll_amp):.1f}°")
print(f"Double support:  {g.ds_frac:.2f} of cycle "
      f"({g.ds_frac * (1.0/g.gait_hz):.1f}s per DS phase at {g.gait_hz} Hz)")
print(f"Phase boundaries: p1={g.p1:.3f}, p2={g.p2:.3f}, p3={g.p3:.3f}")

# Compute key phases from the gait's actual boundaries (not hardcoded)
# so the test still works if ds_frac changes between iterations
P_DS1_START   = g.p0                              # 0.00
P_DS1_MID     = (g.p0 + g.p1) / 2                 # mid DS-shift-left
P_DS1_END     = g.p1                              # end DS-shift-left
P_SS1_MID     = (g.p1 + g.p2) / 2                 # mid L-stance / R-swing
P_SS1_END     = g.p2 - 0.001                      # near end of L-stance
P_DS2_START   = g.p2                              # start DS-shift-right
P_DS2_MID     = (g.p2 + g.p3) / 2                 # mid DS-shift-right
P_DS2_END     = g.p3                              # end DS-shift-right
P_SS2_MID     = (g.p3 + g.p4) / 2                 # mid R-stance / L-swing
P_END         = 0.999

# Key phases to check
show(P_DS1_START,  "cycle start (both at home)")
show(P_DS1_MID,    "mid DS-shift-left")
show(P_DS1_END,    "end DS, start of L-stance + R-swing")
show(P_SS1_MID,    "mid L-stance / mid R-swing")
show(P_SS1_END,    "end L-stance / end R-swing (near phase 0.5)")
show(P_DS2_START,  "start DS-shift-right")
show(P_DS2_MID,    "mid DS-shift-right")
show(P_DS2_END,    "end DS, start of R-stance + L-swing")
show(P_SS2_MID,    "mid R-stance / mid L-swing")
show(P_END,        "near end of cycle")

print("\n\n=== INVARIANT CHECKS ===")

# 1. Cycle wrap continuity: phase 0.0 should equal phase 1.0
print("\n[1. cycle wrap continuity]")
t0    = np.array(g.get_targets(0.0))
t_end = np.array(g.get_targets(0.9999))
diff_deg = np.degrees(np.abs(t0 - t_end))
max_wrap = float(np.max(diff_deg))
print(f"  Max joint diff at wrap: {max_wrap:.3f}°")
assert max_wrap < 1.0, f"FAIL: cycle wrap discontinuity {max_wrap:.2f}°"
print(f"  ✓ Cycle wraps cleanly")

# 2. At phase 0.0, BOTH legs are at home pose
print("\n[2. start-of-cycle home pose]")
t = g.get_targets(0.0)
L_hip   = math.degrees(t[2])
R_hip   = math.degrees(t[8])
L_knee  = math.degrees(t[3])
R_knee  = math.degrees(t[9])
print(f"  L_hip={L_hip:.2f}°  R_hip={R_hip:.2f}° (should both be -20)")
print(f"  L_knee={L_knee:.2f}° R_knee={R_knee:.2f}° (should both be 40)")
assert abs(L_hip - (-20.0)) < 0.5, f"FAIL: L_hip not at home"
assert abs(R_hip - (-20.0)) < 0.5, f"FAIL: R_hip not at home"
assert abs(L_knee - 40.0) < 0.5, f"FAIL: L_knee not at home"
assert abs(R_knee - 40.0) < 0.5, f"FAIL: R_knee not at home"
print(f"  ✓ Both legs at home at phase 0.0")

# 3. L stance is MONOTONIC: L_hip should never decrease during 0.15 → 0.50
print("\n[3. L stance monotonicity]")
phases_L_stance = np.linspace(g.p1, g.p2 - 0.001, 100)
L_hips = [math.degrees(g.get_targets(float(p))[2]) for p in phases_L_stance]
# Should be monotonically increasing (less negative = more extended)
prev = L_hips[0]
non_monotonic_count = 0
for v in L_hips[1:]:
    if v < prev - 0.01:  # allow tiny numerical noise
        non_monotonic_count += 1
    prev = v
print(f"  L_hip range during stance: {L_hips[0]:.2f}° → {L_hips[-1]:.2f}°")
print(f"  Non-monotonic samples: {non_monotonic_count}/100")
assert non_monotonic_count == 0, f"FAIL: L stance not monotonic"
assert L_hips[0] < L_hips[-1], f"FAIL: L stance going wrong direction"
assert abs(L_hips[0] - (-20.0)) < 0.5, f"FAIL: L stance start not -20°"
assert abs(L_hips[-1] - (-12.0)) < 0.5, f"FAIL: L stance end not -12°"
print(f"  ✓ L stance is monotonic forward push")

# 4. R stance is also monotonic: phase 0.65 → 1.0, R_hip from -28 → -20
print("\n[4. R stance monotonicity]")
phases_R_stance = np.linspace(g.p3, g.p4 - 0.001, 100)
R_hips = [math.degrees(g.get_targets(float(p))[8]) for p in phases_R_stance]
prev = R_hips[0]
non_monotonic_count = 0
for v in R_hips[1:]:
    if v < prev - 0.01:
        non_monotonic_count += 1
    prev = v
print(f"  R_hip range during stance: {R_hips[0]:.2f}° → {R_hips[-1]:.2f}°")
print(f"  Non-monotonic samples: {non_monotonic_count}/100")
assert non_monotonic_count == 0, f"FAIL: R stance not monotonic"
assert R_hips[0] < R_hips[-1], f"FAIL: R stance going wrong direction"
assert abs(R_hips[0] - (-28.0)) < 0.5, f"FAIL: R stance start not -28°"
assert abs(R_hips[-1] - (-20.0)) < 0.5, f"FAIL: R stance end not -20°"
print(f"  ✓ R stance is monotonic forward push")

# 5. Both stances produce the SAME body translation magnitude (8°)
print("\n[5. stance push symmetry]")
L_push = L_hips[-1] - L_hips[0]
R_push = R_hips[-1] - R_hips[0]
print(f"  L push: {L_push:.2f}°  R push: {R_push:.2f}°")
assert abs(L_push - R_push) < 0.5, f"FAIL: stance pushes asymmetric"
print(f"  ✓ Both stances push by same amount")

# 6. R swing (phase 0.15-0.50): R_hip starts at -20°, mid-swing has SOME extra
#    forward flexion (from sw_hip_lift), ends at -28° (landing position).
#    The hip lift is small (5°) because the KNEE handles foot clearance.
print("\n[6. R swing arc]")
R_swing_start = math.degrees(g.get_targets(P_DS1_END)[8])
R_swing_mid   = math.degrees(g.get_targets(P_SS1_MID)[8])
R_swing_end   = math.degrees(g.get_targets(P_SS1_END)[8])
print(f"  R_hip @ swing start: {R_swing_start:+.2f}° (should be ~-20)")
print(f"  R_hip @ swing mid:   {R_swing_mid:+.2f}° (should be more negative than start)")
print(f"  R_hip @ swing end:   {R_swing_end:+.2f}° (should be ~-28)")
assert R_swing_mid < R_swing_start - 1.0, "FAIL: R swing should lift (more negative) at mid"
assert R_swing_mid <= R_swing_end + 0.1, "FAIL: R swing mid should be at-least-as-forward as end"
assert abs(R_swing_end - (-28.0)) < 0.5, "FAIL: R swing landing wrong"
print(f"  ✓ R swing arc is correct")

# 7. L swing (phase 0.65-1.00): mirror — L_hip starts at -12°, lifts, ends at -20°
print("\n[7. L swing arc]")
L_swing_start = math.degrees(g.get_targets(P_DS2_END)[2])
L_swing_mid   = math.degrees(g.get_targets(P_SS2_MID)[2])
L_swing_end   = math.degrees(g.get_targets(P_END)[2])
print(f"  L_hip @ swing start: {L_swing_start:+.2f}° (should be ~-12)")
print(f"  L_hip @ swing mid:   {L_swing_mid:+.2f}° (should be more negative than both ends)")
print(f"  L_hip @ swing end:   {L_swing_end:+.2f}° (should be ~-20)")
assert abs(L_swing_start - (-12.0)) < 0.5, "FAIL: L swing start wrong"
assert abs(L_swing_end - (-20.0)) < 0.5, "FAIL: L swing end wrong"
assert L_swing_mid < L_swing_start - 1.0, "FAIL: L swing should lift at mid"
assert L_swing_mid <= L_swing_end + 0.1, "FAIL: L swing mid should be at-least-as-forward as end"
print(f"  ✓ L swing arc is correct")

# 8. Knee lift during swing
print("\n[8. swing knee lift]")
R_knee_swing_mid = math.degrees(g.get_targets(P_SS1_MID)[9])
L_knee_swing_mid = math.degrees(g.get_targets(P_SS2_MID)[3])
print(f"  R_knee @ R-swing mid: {R_knee_swing_mid:.2f}° (should be ~52, 12° above home)")
print(f"  L_knee @ L-swing mid: {L_knee_swing_mid:.2f}° (should be ~52)")
assert R_knee_swing_mid > 50.0, "FAIL: R_knee not lifting"
assert L_knee_swing_mid > 50.0, "FAIL: L_knee not lifting"
print(f"  ✓ Swing knees lift")

# 9. During DS-shift-right (0.50-0.65), L holds at -12° and R holds at -28°
#    (both feet on ground, split stance)
print("\n[9. DS-shift-right split stance]")
t = g.get_targets(P_DS2_MID)  # middle of DS-shift-right
L_hip_ds = math.degrees(t[2])
R_hip_ds = math.degrees(t[8])
print(f"  L_hip = {L_hip_ds:.2f}° (should be ~-12, held from L stance end)")
print(f"  R_hip = {R_hip_ds:.2f}° (should be ~-28, held from R swing end)")
assert abs(L_hip_ds - (-12.0)) < 0.5, "FAIL: L_hip not held during DS"
assert abs(R_hip_ds - (-28.0)) < 0.5, "FAIL: R_hip not held during DS"
print(f"  ✓ DS-shift-right has split stance (L behind body, R ahead)")

# 10. Hip yaws zero everywhere
print("\n[10. hip yaw = 0 everywhere]")
for p in np.linspace(0, 1, 50):
    t = g.get_targets(float(p))
    assert abs(t[0]) < 1e-9, f"FAIL: L_hip_yaw nonzero at {p:.2f}"
    assert abs(t[6]) < 1e-9, f"FAIL: R_hip_yaw nonzero at {p:.2f}"
print(f"  ✓ Hip yaws zero")

# 11. Lateral roll behavior during single support (V3 iter 3 update).
# With lateral_decay_during_swing=False, the roll is HELD at full amplitude
# throughout the entire single-support phase, NOT decayed to zero. This
# fixes the iter 2 failure where the body lost lateral support precisely
# when it needed it most.
print("\n[11. lateral roll held throughout single support]")
roll_swing_start = math.degrees(g.get_targets(P_DS1_END)[1])  # start of right swing
roll_swing_mid   = math.degrees(g.get_targets(P_SS1_MID)[1])  # mid of right swing
roll_swing_end   = math.degrees(g.get_targets(P_SS1_END)[1])  # end of right swing
expected_amp = math.degrees(g.hip_roll_amp)
print(f"  L_hip_roll @ R-swing start: {roll_swing_start:+.2f}° (should be ~+{expected_amp:.1f})")
print(f"  L_hip_roll @ R-swing mid:   {roll_swing_mid:+.2f}° (should be ~+{expected_amp:.1f})")
print(f"  L_hip_roll @ R-swing end:   {roll_swing_end:+.2f}°  (should be ~+{expected_amp:.1f})")

if g.lateral_decay_during_swing:
    # Old behavior: decays from peak to zero across swing
    assert roll_swing_start > expected_amp - 0.5, f"FAIL: roll should peak near +{expected_amp}°"
    assert abs(roll_swing_end) < 0.5, "FAIL: roll should decay to 0 by swing end"
    print(f"  ✓ Lateral decay works (decay mode)")
else:
    # New behavior: held at full amplitude throughout
    assert abs(roll_swing_start - expected_amp) < 0.5, \
        f"FAIL: roll at swing start should be +{expected_amp}°"
    assert abs(roll_swing_mid - expected_amp) < 0.5, \
        f"FAIL: roll at swing mid should still be +{expected_amp}°"
    assert abs(roll_swing_end - expected_amp) < 0.5, \
        f"FAIL: roll at swing end should still be +{expected_amp}°"
    print(f"  ✓ Lateral roll held at full amplitude throughout single support (no-decay mode)")

# 11b. Cross-check: during DS-shift-right, roll should ramp from +amp to -amp
print("\n[11b. DS-shift-right roll transition]")
roll_ds2_start = math.degrees(g.get_targets(P_DS2_START)[1])
roll_ds2_mid   = math.degrees(g.get_targets(P_DS2_MID)[1])
roll_ds2_end   = math.degrees(g.get_targets(P_DS2_END)[1])
print(f"  L_hip_roll @ DS2 start: {roll_ds2_start:+.2f}° (should be ~+{expected_amp:.1f})")
print(f"  L_hip_roll @ DS2 mid:   {roll_ds2_mid:+.2f}° (should be ~0)")
print(f"  L_hip_roll @ DS2 end:   {roll_ds2_end:+.2f}° (should be ~-{expected_amp:.1f})")
if not g.lateral_decay_during_swing:
    assert abs(roll_ds2_start - expected_amp) < 0.5, "FAIL: DS2 should start at +amp"
    assert abs(roll_ds2_mid) < 0.5, "FAIL: DS2 mid should be ~0"
    assert abs(roll_ds2_end - (-expected_amp)) < 0.5, "FAIL: DS2 end should be -amp"
    print(f"  ✓ DS-shift-right cleanly transitions from +amp to -amp")

# 12. Smoothness (no large jumps)
print("\n[12. smoothness]")
phases = np.linspace(0, 0.999, 500)
prev_t = None
max_jump = 0.0
max_info = ""
for p in phases:
    t = np.array(g.get_targets(float(p)))
    if prev_t is not None:
        diff = np.abs(t - prev_t)
        biggest = float(np.max(diff))
        if biggest > max_jump:
            max_jump = biggest
            idx = int(np.argmax(diff))
            max_info = f"phase={p:.3f}, joint={LEG_NAMES[idx]}"
    prev_t = t
max_jump_deg = math.degrees(max_jump)
print(f"  Max per-sample jump: {max_jump_deg:.3f}° ({max_info})")
assert max_jump_deg < 2.0, f"FAIL: discontinuity {max_jump_deg:.2f}°"
print(f"  ✓ Gait is smooth")

# 13. Forward progress estimate
print("\n[13. forward progress estimate]")
thigh_len = 0.100
# Each stance pushes the body by thigh × (sin(stance_end) - sin(stance_start))
import math
L_dx = thigh_len * (math.sin(math.radians(-12.0)) - math.sin(math.radians(-20.0)))
R_dx = thigh_len * (math.sin(math.radians(-20.0)) - math.sin(math.radians(-28.0)))
print(f"  L stance body translation: {L_dx*100:.2f} cm")
print(f"  R stance body translation: {R_dx*100:.2f} cm")
print(f"  Per cycle (2 stances):     {(L_dx + R_dx)*100:.2f} cm")
print(f"  At gait_hz=0.05 (20s):     {(L_dx + R_dx)/20*100:.3f} cm/s")
print(f"  Expected over 1300 steps (~2.07 cycles): "
      f"{(L_dx + R_dx)*2.07*100:.2f} cm")

# 14. Gait init blend (V3 iter 4 — fixes the gait-start lateral shock).
# Verify that on the first call, the roll is ~0 (matching the warmup state)
# rather than the steady-state -9° (which caused the leg-slamming behavior).
# Then verify that after blend_steps calls, the roll has reached the
# steady-state value.
print("\n[14. gait init blend]")
g_blend = ReferenceGaitV3()  # default gait_init_blend_steps=60
print(f"  Default blend steps: {g_blend.gait_init_blend_steps}")

# Steady-state hip roll at phase 0.0 (no-decay mode, end of prev R-stance):
# should be -hip_roll_amp = -9°. Our blended value should ramp from 0 to -9°.
expected_ss = -math.degrees(g_blend.hip_roll_amp)
print(f"  Steady-state L_hip_roll at phase 0.0: {expected_ss:+.2f}°")

# Call 0: blend factor = 0, so roll should be 0 regardless of phase
t = g_blend.get_targets(0.0)
roll_call_0 = math.degrees(t[1])
print(f"  L_hip_roll @ call 0:  {roll_call_0:+.4f}° (should be ~0)")
assert abs(roll_call_0) < 0.01, f"FAIL: first call should give roll≈0, got {roll_call_0}"

# Call ~30 (mid blend): should be partially blended (somewhere between 0 and -9)
for _ in range(29):
    g_blend.get_targets(0.0)
t = g_blend.get_targets(0.0)  # this is call 30
roll_call_30 = math.degrees(t[1])
print(f"  L_hip_roll @ call 30: {roll_call_30:+.4f}° (should be partial, ~{expected_ss/2:.1f})")
assert -9.0 < roll_call_30 < 0.0, \
    f"FAIL: call 30 should be partially blended, got {roll_call_30}"

# Call 60+: blend complete, should be at steady-state
for _ in range(30):
    g_blend.get_targets(0.0)
t = g_blend.get_targets(0.0)  # call 61
roll_call_61 = math.degrees(t[1])
print(f"  L_hip_roll @ call 61: {roll_call_61:+.4f}° (should be ~{expected_ss:+.2f})")
assert abs(roll_call_61 - expected_ss) < 0.1, \
    f"FAIL: call 61 should be at steady-state, got {roll_call_61}"

# Call 200: still steady-state
for _ in range(140):
    g_blend.get_targets(0.0)
t = g_blend.get_targets(0.0)  # call 201
roll_call_201 = math.degrees(t[1])
print(f"  L_hip_roll @ call 201: {roll_call_201:+.4f}° (should still be ~{expected_ss:+.2f})")
assert abs(roll_call_201 - expected_ss) < 0.1, \
    f"FAIL: call 201 should still be steady-state, got {roll_call_201}"

# reset_init_blend should restart the blend
g_blend.reset_init_blend()
t = g_blend.get_targets(0.0)
roll_after_reset = math.degrees(t[1])
print(f"  L_hip_roll @ first call after reset: {roll_after_reset:+.4f}° (should be ~0 again)")
assert abs(roll_after_reset) < 0.01, \
    f"FAIL: after reset, first call should give roll≈0, got {roll_after_reset}"

# Smoothness: the blended roll should never have a jump > some small threshold
# from one call to the next. This is what fixes the "leg slamming" behavior.
g_blend.reset_init_blend()
prev_roll = 0.0
max_jump = 0.0
for i in range(120):
    t = g_blend.get_targets(0.0)
    cur_roll = math.degrees(t[1])
    jump = abs(cur_roll - prev_roll)
    if jump > max_jump:
        max_jump = jump
    prev_roll = cur_roll
print(f"  Max per-call roll jump during blend (120 calls): {max_jump:.4f}°")
# Steady-state value of -9° reached over 60 calls: avg jump is 0.15°/call,
# but smooth_ramp gives a peak velocity around the middle. Allow up to 0.5°.
assert max_jump < 0.5, \
    f"FAIL: blend should be smooth, but max jump is {max_jump:.2f}°"

print(f"  ✓ Gait init blend works correctly")

print("\n\n✓ ALL V3 SANITY CHECKS PASSED")
