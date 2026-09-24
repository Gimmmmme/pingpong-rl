#!/usr/bin/env bash
# Install the standalone Isaac Lab 6 runtime used by PingPong RL.
# This script intentionally does not install a benchmark framework.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PINGPONG_VENV_DIR:-${ROOT}/.venv}"
LAB_DIR="${PINGPONG_LAB_DIR:-${ROOT}/.third_party/IsaacLab}"
LAB_REPO="${PINGPONG_LAB_REPO:-https://github.com/isaac-sim/IsaacLab.git}"
LAB_COMMIT="af1bab4dc173ba69b08fab779c14ead61d13fd33"
SIM_SPEC="isaacsim[all,extscache]==6.0.1.0"

log() { printf '[pingpong-install] %s\n' "$*"; }
die() { printf '[pingpong-install] ERROR: %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || die "git is required"
command -v curl >/dev/null 2>&1 || die "curl is required"

if ! command -v uv >/dev/null 2>&1; then
  log "uv not found; installing it with the official installer"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if [[ -f "${HOME}/.local/bin/env" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/.local/bin/env"
  fi
  export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || die "uv is unavailable; install uv and rerun"

mkdir -p "$(dirname "${LAB_DIR}")"
if [[ ! -d "${LAB_DIR}/.git" && ! -f "${LAB_DIR}/.git" ]]; then
  log "cloning Isaac Lab source"
  git clone --filter=blob:none "${LAB_REPO}" "${LAB_DIR}"
fi

origin="$(git -C "${LAB_DIR}" remote get-url origin 2>/dev/null || true)"
if [[ -z "${origin}" ]]; then
  git -C "${LAB_DIR}" remote add origin "${LAB_REPO}"
elif [[ "${origin}" != "${LAB_REPO}" ]]; then
  die "existing LAB_DIR has a different origin: ${origin}"
fi
log "checking out Isaac Lab ${LAB_COMMIT}"
git -C "${LAB_DIR}" fetch --depth 1 origin "${LAB_COMMIT}"
git -C "${LAB_DIR}" checkout --detach "${LAB_COMMIT}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  log "creating Python 3.12 environment at ${VENV_DIR}"
  uv venv --python 3.12 --seed "${VENV_DIR}"
fi

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
PYTHON="${VENV_DIR}/bin/python"

pip_install() {
  uv pip install --python "${PYTHON}" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match --prerelease=allow "$@"
}

log "installing ${SIM_SPEC}"
pip_install "${SIM_SPEC}"
log "installing Isaac Lab core and PhysX backend"
pip_install -e "${LAB_DIR}/source/isaaclab" -e "${LAB_DIR}/source/isaaclab_physx"
log "installing PingPong RL in editable mode"
pip_install -e "${ROOT}"

cat <<EOF

PingPong RL runtime installed.

  source ${VENV_DIR}/bin/activate
  export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
  python -m pingpong_rl.train --help
  python scripts/smoke.py
EOF
