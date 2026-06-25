#!/usr/bin/env bash
# Drive the whole benchmark end-to-end.
#
# Assumes:
#  - Hunyuan3D-2 service already running (start with services/start_hunyuan.sh &)
#  - litellm proxy already running at http://localhost:4000
#  - CADGENBENCH_DATA_REPO or CADGENBENCH_DATA_DIR exported

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${REPO_ROOT}/runs/submission}"
PARALLEL="${DRMSTEP_PARALLEL:-1}"

echo ">> health check Hunyuan3D"
curl -sf "${DRMSTEP_HUNYUAN_URL:-http://localhost:8081}/" >/dev/null || {
    echo "Hunyuan3D not reachable; start it first." >&2; exit 1; }

echo ">> health check litellm"
curl -sf "${DRMSTEP_LITELLM_URL:-http://localhost:4000}/health" >/dev/null || {
    echo "litellm not reachable; start it first." >&2; exit 1; }

echo ">> run-bench"
drmstep run-bench --out "${OUT_DIR}" --parallel "${PARALLEL}" --verbose

echo ">> score"
drmstep score --submission "${OUT_DIR}"
