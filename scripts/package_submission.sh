#!/usr/bin/env bash
# Zip a submission dir per the cadgenbench HF Space contract.

set -euo pipefail

SUB_DIR="${1:?usage: package_submission.sh <submission_dir> [out.zip]}"
OUT_ZIP="${2:-submission.zip}"

if [[ ! -d "$SUB_DIR" ]]; then
    echo "no such dir: $SUB_DIR" >&2; exit 1
fi

META="$SUB_DIR/meta.json"
if [[ ! -f "$META" ]]; then
    cat > "$META" <<JSON
{
  "submission_name": "drawing-render-mesh-step",
  "method": "LocateAnything-3B + Qwen3-VL-235B view extraction -> Hunyuan3D-2 -> CADFit -> multi-dim Qwen3-VL scaling",
  "authors": ["MBimrose"]
}
JSON
fi

cd "$SUB_DIR"
zip -r "$OUT_ZIP" . -x "*/_work/*" "*/conversation.json" "_drmstep_run.json" "_batch_results.json"
echo "wrote $OUT_ZIP"
echo "contents:"
unzip -l "$OUT_ZIP" 2>/dev/null | head -20
