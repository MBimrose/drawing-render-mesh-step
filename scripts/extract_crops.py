"""Extract isometric crops for all samples in the dataset.

Independent of the full generation pipeline — only runs the VLM view extractor
and saves the crop to disk. Useful for verifying crop quality across the whole
dataset before committing to long CADFit runs.

Output layout: iso_crops/<sample>/{input.png, crop.png, response.txt}

Usage:
    python scripts/extract_crops.py --inputs <dataset_dir> --out iso_crops/
    python scripts/extract_crops.py --inputs <dataset_dir> --out iso_crops/ --parallel 6
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from drmstep.config import load_config
from drmstep import view_vlm

logger = logging.getLogger("extract_crops")


def _resolve_inputs_dir(inputs: Path | None) -> Path:
    if inputs is not None:
        return inputs.resolve()
    try:
        from cadgenbench.common.paths import data_inputs_dir
        return data_inputs_dir()
    except Exception as exc:
        raise FileNotFoundError(f"--inputs not given and resolve failed: {exc}") from exc


def _process_sample(sample: str, inputs_dir: Path, out_dir: Path) -> tuple[str, bool, str]:
    """Extract iso crop for one sample. Returns (sample, ok, bbox_or_error)."""
    config = load_config()
    sample_inputs = inputs_dir / sample
    sample_out = out_dir / sample
    sample_out.mkdir(parents=True, exist_ok=True)

    # Find the drawing.
    input_png = sample_inputs / "input.png"
    if not input_png.exists():
        for cand in sample_inputs.glob("input.*"):
            if cand.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                input_png = cand
                break
    if not input_png.exists():
        return sample, False, "no input image"

    try:
        drawing = Image.open(input_png).convert("RGB")
        # Save a copy of the input for easy browsing.
        drawing.save(sample_out / "input.png")
    except Exception as exc:
        return sample, False, f"load failed: {exc}"

    try:
        res = view_vlm.extract_isometric_bbox(drawing, config)
    except Exception as exc:
        return sample, False, f"VLM error: {exc}"

    (sample_out / "response.txt").write_text(res.raw_response or "")

    if res.bbox_xyxy is None:
        return sample, False, "VLM returned no bbox"

    try:
        # Prefer the cleaned crop (non-part ink whited out) if available.
        crop = res.cleaned_crop if res.cleaned_crop is not None else drawing.crop(res.bbox_xyxy)
        crop.save(sample_out / "crop.png")
    except Exception as exc:
        return sample, False, f"crop failed: {exc}"

    bbox = res.bbox_xyxy
    return sample, True, f"bbox={bbox} size={crop.size}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=None,
                        help="Inputs dir (defaults to cadgenbench resolve)")
    parser.add_argument("--out", type=Path, default=Path("iso_crops"))
    parser.add_argument("--samples", type=str, default=None,
                        help="Comma-separated subset of fixtures")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=4,
                        help="Concurrent VLM calls (HTTP, GIL-released)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    inputs_dir = _resolve_inputs_dir(args.inputs)
    all_samples = sorted(p.name for p in inputs_dir.iterdir()
                         if (p / "description.yaml").exists())
    if args.samples:
        wanted = {s.strip() for s in args.samples.split(",")}
        all_samples = [s for s in all_samples if s in wanted]
    if args.limit:
        all_samples = all_samples[:args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Extracting iso_crops for {len(all_samples)} samples → {args.out}/")
    print(f"VLM parallel calls: {args.parallel}")

    t0 = time.time()
    results: list[tuple[str, bool, str]] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(_process_sample, s, inputs_dir, args.out): s
            for s in all_samples
        }
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                result = fut.result()
                ok = result[1]
                marker = "OK " if ok else "FAIL"
                print(f"  [{marker}] {s}: {result[2]}")
                results.append(result)
            except Exception as exc:
                print(f"  [FAIL] {s}: CRASHED: {exc}")
                results.append((s, False, str(exc)))

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_ok}/{len(results)} succeeded in {time.time()-t0:.1f}s")

    # Build an index page that lists all crops
    index_lines = ["# Iso Crops Index\n"]
    for s, ok, msg in sorted(results, key=lambda x: x[0]):
        status = "✓" if ok else "✗"
        index_lines.append(f"- **{s}** {status} — {msg}")
    (args.out / "INDEX.md").write_text("\n".join(index_lines))
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
