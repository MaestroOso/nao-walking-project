# NAO Robot Integration

The NAO URDF and mesh assets are **not stored in this repository** — they
come from external upstream sources and are assembled locally. This
avoids vendoring third-party git repositories (each of the sources below
is a full git checkout in its own right) directly into this project's
history.

## 1. Upstream sources

| Source | Repository | Commit used | Provides |
|---|---|---|---|
| URDF / robot description | [`ros-naoqi/nao_robot`](https://github.com/ros-naoqi/nao_robot) | `6747646` | NAO V5.0 URDF structure (links, joints, physical properties) |
| Meshes | [`ros-naoqi/nao_meshes`](https://github.com/ros-naoqi/nao_meshes) | `7c5b9f3` | Visual/collision mesh geometry |
| Meshes (alternate/supplementary) | [`jrl-umi3218/nao_description`](https://github.com/jrl-umi3218/nao_description) | `6f4e94b` | Additional mesh/description data used during assembly |

Clone them for reference (they're not committed here):

```bash
git clone https://github.com/ros-naoqi/nao_robot.git
git clone https://github.com/ros-naoqi/nao_meshes.git
git clone https://github.com/jrl-umi3218/nao_description.git

# Pin to the exact commits used during this project's setup:
git -C nao_robot checkout 6747646
git -C nao_meshes checkout 7c5b9f3
git -C nao_description checkout 6f4e94b
```

> The exact file-level provenance of every mesh in the assembled package
> below (which specific files came from which of the two mesh sources)
> wasn't independently re-verified while writing this guide — treat the
> table above as "these are the sources it was built from," not a
> per-file audit.

## 2. Assembling the Isaac-Sim-compatible package

The upstream URDF uses ROS-style `package://` mesh references, which
Isaac Sim's URDF importer doesn't resolve. The working setup converts
these to plain relative paths and lays out a self-contained package:

```
nao_isaac_sim/
├── nao.urdf              # working URDF, package:// paths → relative paths
├── nao_original.urdf     # unmodified upstream URDF, kept for reference
├── NaoWebots.urdf         # Webots-flavored variant (used later for gait capture cross-referencing)
└── meshes/V40/*.dae       # 41 mesh files
```

This assembled package is the one thing every experiment actually
depends on at runtime (referenced via a relative path from wherever the
experiment/training code lives). In this repo, it's the shared asset kept
under `walking-solutions/shared/robot/` rather than duplicated per
experiment.

## 3. Importing into Isaac Sim

```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.kit.commands
import isaacsim.asset.importer.urdf._urdf as urdf_module
from omni.isaac.core import World

world = World()
world.scene.add_default_ground_plane()

config = urdf_module.ImportConfig()
config.merge_fixed_joints = False
config.fix_base = False
config.import_inertia_tensor = True
config.distance_scale = 1.0

result = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path="walking-solutions/shared/robot/nao.urdf",
    import_config=config,
)
prim_path = result[1]   # e.g. "/NAOV40"

world.reset()
```

**Import config, and why each flag matters:**
- `merge_fixed_joints = False` — keep fixed joints as separate links rather
  than collapsing them; needed for finger/wrist joints to be individually
  addressable later.
- `fix_base = False` — the robot must be free-floating (not welded to the
  world), otherwise it can't fall or walk.
- `import_inertia_tensor = True` — use the URDF's actual inertia values
  rather than Isaac Sim's auto-computed approximation.

## 4. Verified structure after import

- **Total joints:** 83
- **Total links:** 84
- **Controllable DOF:** 42 (an early verification pass reported 25 — that
  was superseded once joint inspection covered the finger/thumb joints
  properly; 42 is the number the RL environment code actually uses, see
  `NUM_DOF` in `nao_constants.py`)
- Joint breakdown: 2 head, 5+5 arm, 6+6 leg (the 12 "critical" walking
  joints), 9+9 finger/hand.

## Known USD-level issues found during integration (not visible from Python)

These required inspecting the underlying USD scene graph directly —
nothing about them showed up from high-level joint APIs like
`Articulation.get_joint_positions()`:

- Foot collision geometry was initially empty (robot had no way to
  physically contact the ground).
- `RHipYawPitch` and all finger joints had zero drive stiffness.
- Torque limits on some joints couldn't resist gravity.
- USD `DriveAPI` gain edits didn't persist across `world.reset()` unless
  applied through `controller.set_gains()` instead of setting the USD
  attribute directly.

See `docs/experiment-history.md` (`test_04_position`, `test_05_damping_sweep`)
for how these were diagnosed and fixed — the fix that matters is
*correct damping being applied at all*, not any particular gain
magnitude (both low gains, Kp=50/Kd=5, and the URDF's original high
gains, Kp=35810/Kd=500, work once applied correctly).

Next: [Troubleshooting & Command Reference](04-troubleshooting.md).
