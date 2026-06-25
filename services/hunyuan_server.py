"""Launcher for upstream Hunyuan3D-2 api_server with a configurable subfolder.

Upstream's api_server.py hard-codes the model variant (``hunyuan3d-dit-v2-mini-turbo``)
in ModelWorker's signature and exposes no CLI flag for it. We import the module
(which only declares classes and routes — no work at import time) and then run
our own copy of its ``__main__`` block with our chosen subfolder.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HY_DIR = REPO_ROOT / "third_party" / "Hunyuan3D-2"
sys.path.insert(0, str(HY_DIR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", default="tencent/Hunyuan3D-2")
    parser.add_argument("--subfolder", default="hunyuan3d-dit-v2-0")
    parser.add_argument("--tex_model_path", default="tencent/Hunyuan3D-2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-model-concurrency", dest="limit_concurrency",
                        type=int, default=5)
    parser.add_argument("--enable_tex", action="store_true")
    parser.add_argument("--enable_flashvdm", action="store_true",
                        help="Re-enable the FlashVDM fast-VAE path (off by default).")
    args = parser.parse_args()

    import api_server  # noqa: E402  loads upstream module without side effects
    import uvicorn

    if not args.enable_flashvdm:
        # Upstream's ModelWorker calls self.pipeline.enable_flashvdm(mc_algo='mc')
        # unconditionally. Stub it to a no-op so we run the full (non-flash) VAE
        # path for highest mesh fidelity.
        _orig_enable = api_server.Hunyuan3DDiTFlowMatchingPipeline.enable_flashvdm
        def _noop_enable_flashvdm(self, *a, **kw):
            print("[drmstep] FlashVDM disabled (use --enable_flashvdm to turn back on)")
        api_server.Hunyuan3DDiTFlowMatchingPipeline.enable_flashvdm = _noop_enable_flashvdm

    api_server.args = argparse.Namespace(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        tex_model_path=args.tex_model_path,
        device=args.device,
        limit_model_concurrency=args.limit_concurrency,
        enable_tex=args.enable_tex,
    )
    api_server.model_semaphore = asyncio.Semaphore(args.limit_concurrency)
    api_server.worker = api_server.ModelWorker(
        model_path=args.model_path,
        subfolder=args.subfolder,
        tex_model_path=args.tex_model_path,
        device=args.device,
        enable_tex=args.enable_tex,
    )
    uvicorn.run(api_server.app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
