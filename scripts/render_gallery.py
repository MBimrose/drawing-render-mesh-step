"""Render a gallery of [drawing, iso_crop, mesh, output] for each sample.

For each sample in a batch output dir, produces a side-by-side PNG row:
  [input drawing] [iso crop] [Hunyuan mesh] [CADFit/STEP output]

Concatenates all rows into ``<out>/gallery.png``.

Usage:
    python scripts/render_gallery.py runs/batch --out renders/ --inputs <dataset>
    python scripts/render_gallery.py runs/smoke --out renders/ --inputs <dataset>

Set ``CADGENBENCH_DATA_REPO=HuggingAI4Engineering/cadgenbench-data`` to auto-resolve
the inputs dir via huggingface_hub.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

PANEL = (380, 380)
LABEL_H = 22
ROW_GAP = 4


def _resolve_inputs_dir(inputs: Path | None) -> Path | None:
    if inputs is not None:
        return inputs
    try:
        from cadgenbench.common.paths import data_inputs_dir
        return data_inputs_dir()
    except Exception:
        return None


def _load_mesh(path: Path) -> trimesh.Trimesh | None:
    try:
        return trimesh.load(path, force="mesh")
    except Exception:
        return None


def _render_mesh(mesh: trimesh.Trimesh | None, resolution=PANEL) -> Image.Image | None:
    """Render a mesh from a 3/4 angle. Tries pyrender via trimesh; falls back to
    a simple 2D silhouette if rendering fails (e.g. no GL display)."""
    if mesh is None or len(mesh.faces) == 0:
        return None
    try:
        scene = mesh.scene()
        # 3/4 isometric-ish view
        rot = trimesh.transformations.euler_matrix(
            np.radians(-25), np.radians(35), 0, axes="sxyz"
        )
        # Position camera so the whole mesh is visible
        bb = mesh.bounding_box.extents
        cam_dist = max(bb) * 2.4
        rot[:3, 3] = np.array([0, 0, cam_dist])
        scene.camera_transform = rot
        png = scene.save_image(resolution=resolution, visible=True)
        if png is None:
            raise RuntimeError("scene.save_image returned None")
        return Image.open(io.BytesIO(png))
    except Exception:
        return _matplotlib_render(mesh, resolution)


def _matplotlib_render(mesh: trimesh.Trimesh, resolution=PANEL) -> Image.Image | None:
    """Fallback CPU renderer using matplotlib's 3D triangle plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(resolution[0] / 100, resolution[1] / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        v, f = mesh.vertices, mesh.faces
        # Subsample large meshes
        if len(f) > 50_000:
            idx = np.random.default_rng(0).choice(len(f), 50_000, replace=False)
            f = f[idx]
        triangles = v[f]
        coll = Poly3DCollection(
            triangles, facecolor=(0.75, 0.78, 0.82), edgecolor=(0.2, 0.2, 0.2),
            linewidth=0.05, alpha=1.0,
        )
        ax.add_collection3d(coll)
        bb_min, bb_max = v.min(axis=0), v.max(axis=0)
        ctr = (bb_min + bb_max) / 2
        rng = (bb_max - bb_min).max() / 2 * 1.05
        ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
        ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
        ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
        ax.set_axis_off()
        ax.view_init(elev=25, azim=-35)
        ax.set_box_aspect((1, 1, 1))
        fig.tight_layout(pad=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    except Exception as exc:
        print(f"  matplotlib render failed: {exc}", file=sys.stderr)
        return None


def _render_step(step_path: Path, resolution=PANEL) -> Image.Image | None:
    try:
        import cadquery as cq
        shape = cq.importers.importStep(str(step_path))
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        bb = shape.val().BoundingBox()
        diag = (bb.xlen**2 + bb.ylen**2 + bb.zlen**2)**0.5
        defl = max(0.05, min(1.0, 0.005 * diag)) if diag > 0 else 0.1
        BRepMesh_IncrementalMesh(shape.val().wrapped, defl, False, 0.5, True).Perform()
        tmp = step_path.parent / "_tmp_render.stl"
        cq.exporters.export(shape, str(tmp))
        mesh = trimesh.load(tmp, force="mesh")
        tmp.unlink(missing_ok=True)
        return _render_mesh(mesh, resolution)
    except Exception as exc:
        print(f"  step render failed: {exc}", file=sys.stderr)
        return None


def _font():
    for p in [
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, 14)
            except Exception:
                pass
    return ImageFont.load_default()


def _panel(img: Image.Image | None, label: str, missing_note: str = "missing") -> Image.Image:
    panel = Image.new("RGB", (PANEL[0], PANEL[1] + LABEL_H), (255, 255, 255))
    if img is None:
        ph = Image.new("RGB", PANEL, (60, 60, 60))
        d = ImageDraw.Draw(ph)
        f = _font()
        try:
            tw = d.textlength(missing_note, font=f)
        except Exception:
            tw = len(missing_note) * 8
        d.text(((PANEL[0] - tw) / 2, PANEL[1] / 2 - 7), missing_note, fill=(180, 180, 180), font=f)
        panel.paste(ph, (0, LABEL_H))
    else:
        fitted = img.convert("RGB").resize(PANEL, Image.LANCZOS)
        panel.paste(fitted, (0, LABEL_H))
    d = ImageDraw.Draw(panel)
    d.rectangle([(0, 0), (panel.width, LABEL_H)], fill=(35, 35, 45))
    d.text((6, 3), label, fill=(255, 255, 255), font=_font())
    return panel


def render_sample(
    sample_name: str,
    sample_out_dir: Path,
    work_dir: Path | None,
    inputs_dir: Path | None,
    row_path: Path,
) -> bool:
    drawing_img = None
    if inputs_dir is not None:
        drawing_path = inputs_dir / sample_name / "input.png"
        if drawing_path.exists():
            drawing_img = Image.open(drawing_path)

    iso_img = None
    if work_dir and (work_dir / "iso_crop.png").exists():
        iso_img = Image.open(work_dir / "iso_crop.png")

    mesh_img = None
    if work_dir:
        for cand in ("mesh_simplified.stl", "mesh.stl"):
            mp = work_dir / cand
            if mp.exists():
                mesh = _load_mesh(mp)
                mesh_img = _render_mesh(mesh)
                if mesh_img is not None:
                    break

    output_img = None
    output_label = "output (none)"
    step_path = sample_out_dir / "output.step"
    stl_path = sample_out_dir / "output.stl"
    if step_path.exists():
        output_img = _render_step(step_path)
        output_label = "output (STEP)"
    elif stl_path.exists():
        mesh = _load_mesh(stl_path)
        output_img = _render_mesh(mesh)
        output_label = "output (STL fallback)"

    panels = [
        _panel(drawing_img, f"{sample_name} — drawing", missing_note="no drawing"),
        _panel(iso_img, "iso crop", missing_note="no iso crop"),
        _panel(mesh_img, "Hunyuan mesh"),
        _panel(output_img, output_label),
    ]
    if all(p.getbbox()[3] == LABEL_H + 6 for p in panels):  # heuristic; never true
        return False
    total_w = sum(p.width for p in panels) + ROW_GAP * (len(panels) - 1)
    total_h = max(p.height for p in panels)
    row = Image.new("RGB", (total_w, total_h), (20, 20, 25))
    x = 0
    for p in panels:
        row.paste(p, (x, 0))
        x += p.width + ROW_GAP
    row_path.parent.mkdir(parents=True, exist_ok=True)
    row.save(row_path)
    print(f"  {sample_name}: {row_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path, help="Directory with <sample>/output.* subdirs")
    parser.add_argument("--out", type=Path, default=Path("renders"))
    parser.add_argument("--work-root", type=Path, default=Path("runs/_work"))
    parser.add_argument("--inputs", type=Path, default=None,
                        help="Override inputs dir (defaults to cadgenbench's resolved dir)")
    args = parser.parse_args()

    inputs_dir = _resolve_inputs_dir(args.inputs)
    args.out.mkdir(parents=True, exist_ok=True)
    sample_dirs = sorted(d for d in args.batch_dir.iterdir()
                         if d.is_dir() and not d.name.startswith("_"))
    print(f"Rendering {len(sample_dirs)} samples → {args.out}/")

    rows = []
    for sd in sample_dirs:
        work = args.work_root / sd.name
        row_path = args.out / f"{sd.name}.png"
        if render_sample(sd.name, sd, work, inputs_dir, row_path):
            rows.append(row_path)

    if rows:
        imgs = [Image.open(r) for r in rows]
        total_h = sum(im.height + ROW_GAP for im in imgs)
        total_w = max(im.width for im in imgs)
        gallery = Image.new("RGB", (total_w, total_h), (20, 20, 25))
        y = 0
        for im in imgs:
            gallery.paste(im, (0, y))
            y += im.height + ROW_GAP
        gallery_path = args.out / "gallery.png"
        gallery.save(gallery_path)
        print(f"\nGallery: {gallery_path} ({gallery.width}x{gallery.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
