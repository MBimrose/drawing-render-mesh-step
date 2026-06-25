"""Render a browseable gallery of [drawing | crop] for each sample.

Side-by-side so you can confirm at a glance that the iso-crop matches the
drawing's actual isometric view. Produces one row per sample + a single big
gallery.png.

Usage:
    python scripts/render_crop_gallery.py iso_crops --out crop_gallery
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DRAWING_W = 600
CROP_W = 400
LABEL_H = 22
ROW_GAP = 4


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


def _fit_to_width(img: Image.Image, target_w: int, max_h: int) -> Image.Image:
    w, h = img.size
    new_h = int(h * (target_w / w))
    if new_h > max_h:
        target_w = int(w * (max_h / h))
        new_h = max_h
    return img.convert("RGB").resize((target_w, new_h), Image.LANCZOS)


def _panel(img: Image.Image, label: str, panel_w: int, panel_h: int) -> Image.Image:
    panel = Image.new("RGB", (panel_w, panel_h + LABEL_H), (255, 255, 255))
    fitted = _fit_to_width(img, panel_w, panel_h)
    panel.paste(fitted, ((panel_w - fitted.width) // 2,
                         LABEL_H + (panel_h - fitted.height) // 2))
    d = ImageDraw.Draw(panel)
    d.rectangle([(0, 0), (panel.width, LABEL_H)], fill=(35, 35, 45))
    d.text((6, 3), label, fill=(255, 255, 255), font=_font())
    return panel


def _row(sample: str, drawing_path: Path, crop_path: Path,
         drawing_w: int, crop_w: int, panel_h: int) -> Image.Image:
    drawing_img = Image.open(drawing_path)
    crop_img = Image.open(crop_path)
    d_panel = _panel(drawing_img, f"{sample} — drawing", drawing_w, panel_h)
    c_panel = _panel(crop_img, "iso crop", crop_w, panel_h)
    row = Image.new("RGB", (drawing_w + crop_w + ROW_GAP,
                            max(d_panel.height, c_panel.height)), (20, 20, 25))
    row.paste(d_panel, (0, 0))
    row.paste(c_panel, (drawing_w + ROW_GAP, 0))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crops_dir", type=Path, help="Directory with <sample>/{input.png,crop.png}")
    parser.add_argument("--out", type=Path, default=Path("crop_gallery"))
    parser.add_argument("--panel-h", type=int, default=380, help="Panel height in pixels")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sample_dirs = sorted([d for d in args.crops_dir.iterdir()
                          if d.is_dir() and (d / "crop.png").exists()],
                         key=lambda p: p.name)
    print(f"Rendering {len(sample_dirs)} rows...")

    rows = []
    for sd in sample_dirs:
        sample = sd.name
        row_path = args.out / f"{sample}.png"
        row_img = _row(sample, sd / "input.png", sd / "crop.png",
                       DRAWING_W, CROP_W, args.panel_h)
        row_img.save(row_path)
        rows.append(row_img)
        print(f"  {sample}")

    if rows:
        total_h = sum(r.height + ROW_GAP for r in rows)
        total_w = max(r.width for r in rows)
        gallery = Image.new("RGB", (total_w, total_h), (20, 20, 25))
        y = 0
        for r in rows:
            gallery.paste(r, (0, y))
            y += r.height + ROW_GAP
        gallery_path = args.out / "crop_gallery.png"
        gallery.save(gallery_path)
        print(f"\nGallery: {gallery_path} ({gallery.width}x{gallery.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
