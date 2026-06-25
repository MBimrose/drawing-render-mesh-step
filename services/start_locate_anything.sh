#!/usr/bin/env bash
# LocateAnything-3B runs in-process inside drmstep — no service needed.
#
# This script is a placeholder for the future case where we want to lift it into
# a separate FastAPI worker (e.g. to free VRAM between calls). For now it just
# warms the HF cache.

set -euo pipefail

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("nvidia/LocateAnything-3B")
print("LocateAnything-3B cache warmed.")
PY
