#!/usr/bin/env bash
# Launch Hunyuan3D-2 FastAPI server using its own venv.
#
# Picks the full v2-0 shape model (no texture) since we only need geometry.
# Override defaults with env vars: DRMSTEP_HUNYUAN_PORT, DRMSTEP_HUNYUAN_SUBFOLDER.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HY_DIR="${REPO_ROOT}/third_party/Hunyuan3D-2"
HY_VENV="${HY_DIR}/.venv"
PORT="${DRMSTEP_HUNYUAN_PORT:-8081}"
SUBFOLDER="${DRMSTEP_HUNYUAN_SUBFOLDER:-hunyuan3d-dit-v2-0}"
MODEL_PATH="${DRMSTEP_HUNYUAN_MODEL_PATH:-tencent/Hunyuan3D-2}"

if [[ ! -x "${HY_VENV}/bin/python" ]]; then
    echo "ERROR: Hunyuan3D venv not found at ${HY_VENV}" >&2
    echo "Bootstrap with: python -m venv ${HY_VENV} && source ${HY_VENV}/bin/activate && pip install -r ${HY_DIR}/requirements.txt" >&2
    exit 1
fi

cd "${REPO_ROOT}"
exec "${HY_VENV}/bin/python" services/hunyuan_server.py \
    --host 0.0.0.0 --port "${PORT}" \
    --model_path "${MODEL_PATH}" \
    --subfolder "${SUBFOLDER}"
