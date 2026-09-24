# Installation

This project uses the public Isaac Lab source and Isaac Sim Python wheels. The
repository does not vendor the simulator or clone a task benchmark as a Python
dependency.

The [ManipArena Sim installation guide](https://github.com/maniparena/maniparena-sim/blob/main/docs/install.md) is the reference for the compatible Python and Sim layout. The standalone installer below installs only the Isaac Lab and Isaac Sim pieces needed by this repository.

## Pins

The installer uses the following versions, matching the verified Isaac Lab 6
runtime:

| Component | Pin |
| --- | --- |
| Python | `3.12` |
| Isaac Sim wheels | `isaacsim[all,extscache]==6.0.1.0` |
| Isaac Lab source | commit `af1bab4dc173ba69b08fab779c14ead61d13fd33` |
| Physics backend | `isaaclab_physx` from the same Isaac Lab checkout |

The Isaac Lab commit is checked out directly from
`https://github.com/isaac-sim/IsaacLab.git`. Isaac Sim wheels are resolved from
`https://pypi.nvidia.com`, with prereleases enabled because that is required by
the Sim 6 dependency set.

## Installer behavior

Run this from the repository root:

```bash
./scripts/install.sh
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
uv pip install -e .
```

The script creates `.venv`, clones or updates the pinned Isaac Lab checkout
under `.third_party/IsaacLab`, installs the Sim wheels, installs the Isaac Lab
core and PhysX packages in editable mode, and installs this project in
editable mode. It does not change a global Python installation. Set
`PINGPONG_VENV_DIR` or `PINGPONG_LAB_DIR` if those local directories need to be
placed elsewhere.

The NVIDIA Isaac Sim package is distributed under its own license and requires
accepting NVIDIA's license terms through the normal installation flow. The
license environment variables above only acknowledge that EULA to the local
runtime.

## Manual equivalent

The essential commands are:

```bash
uv venv --python 3.12 --seed .venv
uv pip install --python .venv/bin/python \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match --prerelease=allow \
  'isaacsim[all,extscache]==6.0.1.0'
uv pip install --python .venv/bin/python -e \
  .third_party/IsaacLab/source/isaaclab \
  -e .third_party/IsaacLab/source/isaaclab_physx
uv pip install --python .venv/bin/python -e .
```

The editable install is deliberate: the scene resolves the reviewed robot USD
relative to this source tree. A wheel install must arrange the same asset path
explicitly before constructing the scene.

## Verification

```bash
.venv/bin/python -m pingpong_rl.train --help
.venv/bin/python -m pingpong_rl.evaluate --help
.venv/bin/python scripts/smoke.py
.venv/bin/python scripts/smoke.py --cameras
.venv/bin/python -m pytest -q
```

`smoke.py` uses `--viz none` and a CUDA tensor action path for headless Sim 6.
Keyboard or interactive windows are outside the smoke test. If Isaac Lab logs
an inotify watch-limit warning while starting Kit, increase the host watch
limit or run headless with the existing limit; it is unrelated to the project
configuration.
