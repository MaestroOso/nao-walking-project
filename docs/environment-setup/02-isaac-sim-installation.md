# Isaac Sim Installation

Assumes the [prerequisite checks](01-system-requirements.md) have already
passed.

## 1. Create the virtual environment

```bash
cd nao-walking-project
python3.10 -m venv isaac_env
source isaac_env/bin/activate
pip install --upgrade pip
```

Always activate this environment before doing anything with Isaac Sim:

```bash
source isaac_env/bin/activate
```

## 2. Install Isaac Sim via pip

The Omniverse Launcher (GUI installer) is deprecated by NVIDIA. Use the
pip method instead — it's the current official approach, integrates
better with a normal Python workflow, and is a lighter install:

```bash
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

This installs 26 packages (25 `isaacsim-*` components + `omniverse-kit`),
including RL support, robot simulation tools, motion generation, and
ROS1/ROS2 integration. Expect ~5–10 GB download, ~20–30 minutes.

## 3. Verify the install

```bash
python -c "from isaacsim import SimulationApp; print('Isaac Sim Ready!')"
```

Then run an actual simulation smoke test — importing without error isn't
sufficient proof it works:

```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({})   # must be constructed before any other isaacsim/omni import

from omni.isaac.core import World
world = World()
world.scene.add_default_ground_plane()
world.reset()
for _ in range(60):
    world.step(render=True)

simulation_app.close()
```

If a ground plane renders and the process exits cleanly, the install is
good.

**Critical import order:** `SimulationApp` must always be constructed
before importing any other `isaacsim` or `omni` module. Getting this
backwards is a common source of confusing failures.

## 4. First-run behavior (expected, not a bug)

- **First run of any new scene:** slow — approx. 266 seconds observed,
  dominated by extension downloading/caching and shader compilation.
- **Subsequent runs:** ~10–30 seconds.
- Warnings to ignore: RTX SceneDB buffer-size adjustment messages,
  `omni.isaac.core.utils.nucleus` deprecation warnings — informational,
  not failures.

## Useful lookups

```bash
pip list | grep -E "(isaacsim|omni)"     # installed Isaac Sim packages
pip show isaacsim                         # package metadata
python -c "import isaacsim; print(isaacsim.__path__)"   # install location
```

Next: [NAO Robot Integration](03-nao-robot-integration.md).
