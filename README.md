# drawing-render-mesh-step

A pipeline that turns mechanical engineering drawings into CAD STEP files for the
[cadgenbench](https://github.com/huggingface/cadgenbench) benchmark.

```
input.png (multi-view drawing)
        │
   ┌────▼──── [1] LocateAnything-3B — crop the isometric view
   │
   ▼
   ┌──────── [2] Hunyuan3D-2 — image → textureless mesh (GLB → STL)
   │
   ▼
   ┌──────── [3] CADFit — mesh → parametric CadQuery program
   │
   ▼
   ┌──────── [4] Claude (via litellm) — read annotated dims, emit scale factors
   │
   ▼
   output.step (scaled, valid BREP)
```

For the editing task, `input.step` is fed through CADFit to recover a CadQuery
program, then Claude patches it per the edit instruction.

## Quick start

```bash
# 1. Init submodules
git clone --recurse-submodules https://github.com/MBimrose/drawing-render-mesh-step
cd drawing-render-mesh-step

# 2. Orchestrator env
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 3. Hunyuan3D-2 env (separate — it pins its own torch/diffusers)
python -m venv third_party/Hunyuan3D-2/.venv
source third_party/Hunyuan3D-2/.venv/bin/activate
pip install -r third_party/Hunyuan3D-2/requirements.txt
deactivate

# 4. CADFit env
python -m venv third_party/CADFit/.venv
source third_party/CADFit/.venv/bin/activate
pip install -r third_party/CADFit/requirements.txt
deactivate

# 5. Start Hunyuan3D-2 as a service (uses third_party/Hunyuan3D-2/.venv)
bash services/start_hunyuan.sh &

# 6. Point cadgenbench at the public input dataset on HF Hub
export CADGENBENCH_DATA_REPO=cadgenbench/cadgenbench-data

# 7. Run a single sample end-to-end
drmstep run-sample --sample <fixture_name> --out runs/smoke

# 8. Run the whole benchmark
drmstep run-bench --out runs/submission --parallel 4
drmstep score --submission runs/submission
bash scripts/package_submission.sh runs/submission submission.zip
```

## Components

| Step | Module | Backend |
|------|--------|---------|
| View extraction | `drmstep.view_extract` | `nvidia/LocateAnything-3B` (in-process) |
| Mesh generation | `drmstep.mesh_generate` | `tencent/Hunyuan3D-2` (FastAPI service @ `:8081`) |
| CAD fitting | `drmstep.cad_fit` | `ghadinehme/CADFit` (subprocess in its own venv) |
| Dimensional scaling | `drmstep.scaling` | Claude (Opus 4.7) via local litellm proxy @ `:4000` |
| Edit task | `drmstep.edit_pipeline` + `drmstep.code_patch` | Same CADFit + Claude pair |

## Licensing

- This orchestrator: MIT (see `LICENSE`).
- `third_party/CADFit` ships under **CC BY-NC 4.0** — non-commercial research use only.
  If you need commercial use, contact the CADFit authors.
- `third_party/Hunyuan3D-2` ships under the Tencent Hunyuan Community License.
- `nvidia/LocateAnything-3B` ships under its own NVIDIA OneWay Noncommercial License.

This repo is for benchmark/research use; it is **not** suitable for commercial deployment.
