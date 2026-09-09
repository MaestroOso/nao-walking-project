# System Requirements & Verified Configuration

## Verified working configuration

This is the exact configuration Isaac Sim 4.5 was installed and verified
against, taken from the dated setup log (not the aspirational spec sheet —
see the discrepancy note in the guide's [README](README.md)).

| Component | Verified value |
|---|---|
| OS | Ubuntu 24.04.3 LTS (Noble Numbat), kernel 6.14.0-35-generic |
| GPU | NVIDIA GeForce RTX 4060 Laptop (8 GB VRAM) |
| NVIDIA Driver | 580.95.05 |
| CUDA | 13.0 |
| CPU | AMD Ryzen 7 7435HS (8 cores / 16 threads) |
| RAM | ~24 GB (23713 MB total) |
| Disk space used | ~25–30 GB (Isaac Sim + caches) |
| Python | 3.10.19 (dedicated virtual environment) |
| GCC / G++ | 11.5.0 |
| GLIBC | 2.39 |
| Isaac Sim | 4.5.0.0 |

## Why these specific choices

**Ubuntu 24.04 instead of the officially-supported 22.04.** Isaac Sim 4.5
officially targets 22.04. 24.04 was used anyway because the machine was
already configured with working NVIDIA drivers on it, and reinstalling the
OS purely for official support was judged not worth the time. Risk was
mitigated by explicitly verifying GLIBC compatibility (2.39, above the
2.34 minimum) and installing a GCC/G++ 11 compatibility layer (24.04 ships
GCC 13 by default). This worked, but it's an unsupported configuration —
if reproducing this setup, 22.04 is the safer default; 24.04 is a
documented fallback that's known to work with the compiler pin below.

**GCC/G++ 11 pinned via `update-alternatives`**, not the system default:

```bash
sudo apt install gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
gcc --version   # must report 11.x
g++ --version   # must report 11.x — a mismatched g++ (e.g. still 13) is a common miss
```

**Python 3.10 in a dedicated virtual environment (`isaac_env`)**, not
system Python or a shared one — Isaac Sim 4.5 requires 3.10 specifically
(later Isaac Sim versions move to 3.11), and isolation avoids dependency
conflicts with anything else on the machine.

**Isaac Sim 4.5 instead of 5.x** — chosen for Python 3.10 compatibility,
a more stable/documented release at the time, and because it's sufficient
for bipedal locomotion research; not chosen for any feature specific to
4.5.

## Pre-install verification checklist

Run these before starting the Isaac Sim install — catching a mismatch here
is much cheaper than debugging it mid-install:

```bash
nvidia-smi                  # confirm driver + CUDA version detected
gcc --version && g++ --version   # both must match (11.x, per above)
ldd --version                # GLIBC version, need >= 2.34
python3.10 --version
df -h ~                      # need ~30GB free
```
