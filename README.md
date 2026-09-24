# PingPong RL

Standalone two-robot table-tennis reinforcement learning in Isaac Lab 6 and
PhysX. The repository contains the environment, the RGB-only policy interface,
the residual controller, and small train/evaluate entry points. It does not
import a benchmark runtime or require a benchmark repository at runtime.

The actor is restricted to deployment observations: RGB ball tracking,
proprioception, paddle poses obtained from robot kinematics, a calibrated task
target, confidence, and the previous action. Training may use a separate
privileged critic for physical state and contact diagnostics; those quantities
are never passed to the actor or the evaluation policy path.

## Requirements

- Ubuntu 22.04 or 24.04
- NVIDIA GPU and a current NVIDIA driver
- Python 3.12
- `uv` and `git`
- Isaac Sim `6.0.1.0` wheels and Isaac Lab at the pinned commit installed by
  [`scripts/install.sh`](scripts/install.sh)

The installer follows the public Isaac Lab 6 installation layout: Isaac Sim
comes from NVIDIA's Python index and Isaac Lab is installed from a pinned Git
checkout. The external simulator is intentionally kept outside this source
tree.

The upstream [ManipArena Sim repository](https://github.com/maniparena/maniparena-sim) and its [installation guide](https://github.com/maniparena/maniparena-sim/blob/main/docs/install.md) are used only as the local Isaac Lab bootstrap and public USD asset reference. This project does not import that repository's task registry or runtime.

## Install

```bash
git clone <this-repository>
cd <this-repository>
./scripts/install.sh
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
```

The script installs this package in editable mode so the local USD asset
remains available at the path used by the scene. If you create the environment
manually, run `uv pip install -e .` after installing Isaac Lab.

The robot USD is redistributed under the source repository's Apache 2.0
notice at the pinned upstream revision. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Verify

The fast command checks the Python entry points without starting the simulator:

```bash
python -m pingpong_rl.train --help
python -m pingpong_rl.evaluate --help
```

The Isaac Lab smoke test starts Sim headlessly, builds the table, ball, two
robot articulations, and PhysX scene, then performs reset and simulation steps:

```bash
python scripts/smoke.py
python scripts/smoke.py --cameras
```

CPU policy and geometry tests are run with:

```bash
python -m pytest -q
```

## Train and evaluate

```bash
python -m pingpong_rl.train --help
python -m pingpong_rl.evaluate --help
```

The command-line help is the source of truth for rollout length, seed,
checkpoint, camera, and device options. A checkpoint is an artifact of a
particular configuration; record its configuration and checksum with each
experiment.

## Layout

```text
assets/bimanual_robot/       public bimanual robot USD
pingpong_rl/                 scene, PhysX environment, controller, PPO, vision
scripts/install.sh           pinned Isaac Lab/Sim installer
scripts/smoke.py             headless reset/step smoke test
docs/INSTALL.md              installation details and version pins
docs/ALGORITHM.md            observations, control, reward, and evaluation
tests/                       CPU-side regression tests
```

## Scope

This release is a simulation and policy-engineering repository. It does not
claim physical-robot validation, sim-to-real safety, or aerodynamic spin
fidelity. Camera calibration, timing, collision margins, and hardware limits
must be checked independently before any physical experiment.

## License

Project source and documentation are Apache 2.0; see [`LICENSE`](LICENSE).
Third-party software and the redistributed robot USD retain their upstream
notices and terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
