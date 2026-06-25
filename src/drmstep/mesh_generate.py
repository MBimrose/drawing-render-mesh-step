"""Hunyuan3D-2 HTTP client. Talks to the FastAPI server in services/start_hunyuan.sh."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import httpx
import trimesh
from PIL import Image

from .config import Config

logger = logging.getLogger(__name__)


class MeshGenError(RuntimeError):
    pass


def _image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_to_mesh(image: Image.Image, out_stl: Path, config: Config) -> Path:
    """POST a PIL image to Hunyuan3D-2 /generate. Save the returned GLB as STL.

    Returns the STL path. Raises MeshGenError on transport or load failure.
    """
    payload = {
        "image": _image_to_b64(image),
        "num_inference_steps": config.hunyuan_num_inference_steps,
        "octree_resolution": config.hunyuan_octree_resolution,
        "guidance_scale": config.hunyuan_guidance_scale,
        "texture": False,
        "type": "glb",
    }
    url = f"{config.hunyuan_url}/generate"
    logger.info("POST %s (steps=%d, octree=%d)", url, payload["num_inference_steps"],
                payload["octree_resolution"])
    try:
        resp = httpx.post(url, json=payload, timeout=config.hunyuan_timeout_s)
    except httpx.HTTPError as exc:
        raise MeshGenError(f"hunyuan3d POST failed: {exc}") from exc
    if resp.status_code != 200:
        raise MeshGenError(f"hunyuan3d returned {resp.status_code}: {resp.text[:300]}")

    glb_bytes = resp.content
    try:
        scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", force="mesh")
    except Exception as exc:
        raise MeshGenError(f"glb decode failed: {exc}") from exc
    if not isinstance(scene, trimesh.Trimesh) or scene.is_empty:
        raise MeshGenError("hunyuan3d returned empty mesh")

    out_stl.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out_stl)
    logger.info("wrote %s (%d faces)", out_stl, len(scene.faces))
    return out_stl
