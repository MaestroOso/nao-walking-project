# Environment Setup Guide

How to reproduce the local development environment for this project from
scratch: system prerequisites, Isaac Sim installation, and NAO robot
integration. This consolidates setup notes that were previously scattered
across `setup_log.txt`, `USEFUL_COMMANDS.md`, `phase1_completion_report.md`,
`session_tracker.md`, and `notes.md` in the working project folder.

## Contents

1. [System Requirements & Verified Configuration](01-system-requirements.md)
   — hardware/software baseline, what was actually verified working
2. [Isaac Sim Installation](02-isaac-sim-installation.md) — step-by-step
   install via pip, verification, first-run behavior
3. [NAO Robot Integration](03-nao-robot-integration.md) — where the URDF
   and mesh assets come from, how they were adapted for Isaac Sim, and how
   to import the robot
4. [Troubleshooting & Command Reference](04-troubleshooting.md) — common
   issues and their fixes, quick command lookup

## Quick start

```bash
cd nao-walking-project
python3.10 -m venv isaac_env
source isaac_env/bin/activate
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
python -c "from isaacsim import SimulationApp; print('Isaac Sim Ready!')"
```

See [02-isaac-sim-installation.md](02-isaac-sim-installation.md) for the
full sequence including prerequisite checks, and
[03-nao-robot-integration.md](03-nao-robot-integration.md) before trying to
import the robot — the URDF assets aren't in this repo and need to be
fetched/assembled first.

## Known discrepancy vs. `CLAUDE.md`

The system specs recorded here are what was **actually verified** during
setup (see the dated log this was reconstructed from). They differ from
the specs table in the repo's `CLAUDE.md`:

| | `CLAUDE.md` | Actually verified |
|---|---|---|
| OS | Ubuntu 22.04 | Ubuntu 24.04.3 LTS (Noble Numbat) |
| RAM | 32 GB | ~24 GB (23713 MB reported) |
| GPU | RTX 4060 | RTX 4060 **Laptop** |

Not resolved here — flagging it so the discrepancy doesn't get committed to
the academic report by accident. Worth reconciling `CLAUDE.md` in a
follow-up if the 24.04/laptop-GPU numbers are the correct ones.
