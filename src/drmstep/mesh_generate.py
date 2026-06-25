"""Hunyuan3D-2 HTTP client + watertight cleanup.

Talks to the FastAPI server launched by ``services/start_hunyuan.sh``. After
receiving the GLB, manifoldifies the mesh via pymeshlab's alpha-wrap so CADFit's
manifold3d-based IoU computation can actually compare it (raw Hunyuan output is
non-watertight, which collapses every CADFit candidate to IoU=0).
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import httpx
import pymeshlab
import trimesh
from PIL import Image

from .config import Config

logger = logging.getLogger(__name__)

# Alpha-wrap parameters tuned for Hunyuan3D-2 output (mesh normalized roughly to ±1).
# alpha_fraction = 0.01 produces ~15k watertight faces — coarse enough that CADFit's
# sketch extraction stays fast (a denser mesh has the same logical features but blows
# up runtime), while still preserving holes and fillets.
_ALPHA_WRAP_FRACTION = 0.01
_OFFSET_FRACTION = 0.003


class MeshGenError(RuntimeError):
    pass


def _image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _manifoldify(
    mesh: trimesh.Trimesh,
    alpha_fraction: float = _ALPHA_WRAP_FRACTION,
    offset_fraction: float = _OFFSET_FRACTION,
    keep_largest_body: bool = False,
) -> trimesh.Trimesh:
    """Run pymeshlab's alpha-wrap to produce a watertight version of ``mesh``.

    ``alpha_fraction`` controls smoothing: smaller → preserves fine detail (but
    can leave high-genus topology); larger → smoother surface, lower genus,
    fewer disconnected bodies. Default 0.01 is detail-preserving.

    ``keep_largest_body`` is useful for the simplified-mesh retry path: when
    Hunyuan3D produces multi-body output (e.g. thin sheet metal), keeping only
    the largest body removes spurious satellites that confuse CADFit.

    Falls back to the input mesh on any pymeshlab failure (e.g. degenerate input).
    """
    if mesh.is_watertight and not keep_largest_body:
        return mesh
    try:
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
        ms.generate_alpha_wrap(
            alpha_fraction=alpha_fraction,
            offset_fraction=offset_fraction,
        )
        cm = ms.current_mesh()
        wrapped = trimesh.Trimesh(vertices=cm.vertex_matrix(), faces=cm.face_matrix())

        if keep_largest_body:
            bodies = wrapped.split(only_watertight=True)
            if bodies:
                bodies = sorted(bodies, key=lambda b: b.volume, reverse=True)
                wrapped = bodies[0]

        if not wrapped.is_watertight:
            logger.warning("alpha-wrap output still non-watertight; using raw mesh")
            return mesh
        logger.info(
            "alpha-wrap(α=%.3f%s): %d → %d faces, watertight, vol=%.4f, euler=%d",
            alpha_fraction, ",largest" if keep_largest_body else "",
            len(mesh.faces), len(wrapped.faces), wrapped.volume, wrapped.euler_number,
        )
        return wrapped
    except Exception as exc:
        logger.warning("alpha-wrap failed (%s); using raw mesh", exc)
        return mesh


def simplify_for_retry(stl_path: Path, out_stl: Path) -> Path:
    """Re-simplify an existing mesh STL with a more aggressive alpha-wrap.

    Used when CADFit emits no operations on the detail-preserving wrap. Coarser
    alpha + largest-body-only typically drops genus enough for CADFit's
    extrude/revolve primitives to find a fit.
    """
    raw = trimesh.load(stl_path, force="mesh")
    aggressive = _manifoldify(
        raw,
        alpha_fraction=0.02,     # ~2x coarser
        offset_fraction=0.005,
        keep_largest_body=True,
    )
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    aggressive.export(out_stl)
    return out_stl


def image_to_mesh(image: Image.Image, out_stl: Path, config: Config) -> Path:
    """POST a PIL image to Hunyuan3D-2 /generate, alpha-wrap to watertight, save as STL."""
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

    sealed = _manifoldify(scene)
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    sealed.export(out_stl)
    logger.info(
        "wrote %s (%d faces, watertight=%s)", out_stl, len(sealed.faces), sealed.is_watertight,
    )
    return out_stl
