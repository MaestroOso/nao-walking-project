# Troubleshooting & Command Reference

## Environment activation

```bash
cd nao-walking-project
source isaac_env/bin/activate
```

```bash
deactivate                 # leave the virtual environment
echo $VIRTUAL_ENV          # confirm which environment is active
```

## Display / GUI issues

```bash
export DISPLAY=:0
xhost +local:
```

## Isaac Sim won't start / hangs

```bash
ps aux | grep isaac        # check for a stuck process
pkill -f isaac              # kill it
```

- First launch of any new scene is slow (extension caching, shader
  compilation) — ~4.5 minutes observed on first run. This is expected,
  not a hang. Subsequent runs are ~10–30 seconds.
- RTX SceneDB buffer-size warnings and `omni.isaac.core.utils.nucleus`
  deprecation warnings are informational — ignore them.

## Import order errors

`SimulationApp` must be constructed **before** any other `isaacsim` or
`omni` import:

```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({})
# only now import anything else isaacsim/omni-related
```

Getting this backwards produces confusing, seemingly unrelated errors.

## GCC/G++ version mismatch

If `gcc --version` and `g++ --version` don't both report the same major
version (11.x), the `update-alternatives` priorities are wrong — see
[01-system-requirements.md](01-system-requirements.md#why-these-specific-choices).

## Physics/robot falls immediately or explodes (NaN)

Not an installation problem — this is the actual RL/physics-tuning
problem the project spent its early experiments solving. See
`docs/experiment-history.md`, phase 1 (`test_01`–`test_05`) for the root
causes (empty foot collision, zero joint stiffness, damping not applying)
and the fix.

## Training is slow

```python
NaoWalkingEnv(headless=True, render_mode='human')   # no GUI rendering = 2-3x faster
```

Also reduce `n_steps` in the RL algorithm's hyperparameters if running
low on memory.

## Cleaning Python cache

```bash
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete
```

## Quick system checks

```bash
nvidia-smi                          # driver + GPU status
watch -n 1 nvidia-smi               # live GPU monitoring during training
free -h                              # RAM usage
df -h ~                              # disk space
ldd --version                        # GLIBC version
```

## Isaac Sim package lookups

```bash
pip list | grep -E "(isaacsim|omni)"
pip show isaacsim
python -c "import isaacsim; print(isaacsim.__path__)"
```

## Monitoring training (TensorBoard)

```bash
tensorboard --logdir training_logs/
# open http://localhost:6006
```
