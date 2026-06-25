#!/usr/bin/env bash
# Pre-download model weights so the first benchmark run isn't blocked on network.

set -euo pipefail

python - <<'PY'
from huggingface_hub import snapshot_download
for repo, allow in [
    ("nvidia/LocateAnything-3B", None),
    ("tencent/Hunyuan3D-2", ["hunyuan3d-dit-v2-0/*", "hunyuan3d-vae-v2-0/*", "*.json"]),
]:
    print(f"snapshot_download({repo})")
    snapshot_download(repo, allow_patterns=allow)
PY
