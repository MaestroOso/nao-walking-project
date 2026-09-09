# test_04_position

**Focus:** Pure physics forensics — no RL at all. Stepped away from
training entirely to answer "why does NAO fall regardless of pose or
reward tuning?"

**What we did:** Inspected the underlying USD scene graph directly,
because nothing about the problem was visible from the Python-level
joint APIs (`Articulation.get_joint_positions()` and similar). Many
scripts here represent successive attempts (`fix_standing.py` →
`fix_standing_v2.py`, `forensic_standing.py` → `forensic_v2.py`,
`add_collisions.py` → `_v2` → `_v3`) narrowing down the root cause.

**Found root causes:**
- Foot collision geometry was empty — the robot had no way to physically
  contact the ground at all.
- `RHipYawPitch` and all finger joints had zero drive stiffness.
- Torque limits on some joints couldn't resist gravity.
- USD `DriveAPI` gain edits didn't survive `world.reset()` unless applied
  through `controller.set_gains()` instead of setting the USD attribute
  directly.

**Outcome:** The robot still fell by the end of this experiment — but for
the first time, exactly *why* was known. This set up `test_05`'s damping
sweep to find the actual fix.

**Key learning:** URDF import bugs are invisible at the Python API level.
Broken collision geometry and zero-stiffness joints required USD-prim-level
inspection to find. If a robot behaves physically wrong in a way that
resists reward/pose tuning, check the USD scene graph before assuming it's
an RL problem.
