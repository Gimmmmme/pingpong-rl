# Third-party notices

This file records the sources that are redistributed or required by the
project. Their licenses remain separate from the project license in `LICENSE`.

## Bimanual robot USD

`assets/bimanual_robot/bimanual_robot.usd` is copied from the public
`maniparena-sim` repository at commit
`bc40c5f7c0f58ca86a53a2ab1a45a8e5a4a0916b`:

<https://github.com/maniparena/maniparena-sim>

The upstream README at that revision states **Apache License 2.0**. The source
repository's asset blob has SHA-256
`74f3698c599df35ac38d8c853bcc1230aa5dfb7ca52381e8f7c2450454355e95`.
The attribution and upstream license statement are retained here because the
USD is redistributed in this repository. No other benchmark code is required
by the standalone package.

## Isaac Lab

Isaac Lab is installed by `scripts/install.sh` from
<https://github.com/isaac-sim/IsaacLab> at commit
`af1bab4dc173ba69b08fab779c14ead61d13fd33`. Isaac Lab source files state the
BSD 3-Clause license and retain the copyright notice for the Isaac Lab Project
Developers. The checkout is external to this repository and is not copied into
it.

## Isaac Sim

Isaac Sim Python wheels (`isaacsim[all,extscache]==6.0.1.0`) are downloaded
from NVIDIA's package index during installation. Isaac Sim is proprietary
software distributed under NVIDIA's license and EULA; its terms apply to the
installed wheels. The wheels are not redistributed here.

## Python packages

NumPy, PyTorch, OpenCV, and the other Python packages installed by the
project retain their respective upstream licenses. Their package metadata and
license files in the active environment are authoritative; this repository does
not replace those terms.
