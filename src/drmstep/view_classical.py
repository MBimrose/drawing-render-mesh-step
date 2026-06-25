"""Classical-CV isometric-view extraction for engineering drawings.

Pipeline:
  1. Threshold the page to a binary ink mask.
  2. Split into rectangular view regions by whitespace gaps (row + column projections).
  3. Score each region by the fraction of detected Hough lines that are NOT axis-aligned.
     The isometric pictorial has the most diagonals; orthographic views are mostly
     horizontal/vertical; title blocks rank in between but are filtered by area+location.
  4. Return the highest-scoring region's bounding box.

Deterministic, no VLM or GPU; runs in ~50–200ms per page.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Tuning knobs (fractions of image dim unless noted).
_INK_THRESHOLD = 200          # 8-bit; anything darker than this is ink
_WHITESPACE_ROW_FRAC = 0.005  # row counts as whitespace if <0.5% pixels are ink
_MIN_GAP_FRAC = 0.015         # whitespace gap must be ≥1.5% of page dim to split
_MIN_VIEW_AREA_FRAC = 0.01    # ignore regions smaller than 1% of page area
_MAX_VIEW_AREA_FRAC = 0.6     # ignore the page-frame "region" (almost full page)
_AXIS_DEG_TOL = 8.0           # lines within ±8° of axis count as axis-aligned
_TITLE_BLOCK_BOTTOM_FRAC = 0.75  # views below this Y are likely title-block content


def _binarize(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("L"))
    _, binary = cv2.threshold(arr, _INK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    return binary  # uint8: 255 where ink, 0 where paper


def _find_gaps(projection: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """Return [(start, end)] spans of "filled" rows/cols separated by ≥min_gap of zeros."""
    is_filled = projection > 0
    spans: list[tuple[int, int]] = []
    in_span = False
    start = 0
    gap_run = 0
    for i, filled in enumerate(is_filled):
        if filled:
            if not in_span:
                in_span = True
                start = i
            gap_run = 0
        else:
            if in_span:
                gap_run += 1
                if gap_run >= min_gap:
                    spans.append((start, i - gap_run + 1))
                    in_span = False
                    gap_run = 0
    if in_span:
        spans.append((start, len(is_filled)))
    return spans


def _split_into_views(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Split the page into view-region bboxes via morphological opening + connected components.

    CAD drawings have dimension/leader lines threading between views, so plain whitespace
    projection never sees clean gaps. Instead:
      1. Remove the page frame.
      2. Open (erode → dilate) with a small kernel to wipe out thin dimension lines,
         arrows, and stray text while keeping the bulky view geometry intact.
      3. Dilate aggressively to bond each view's leftover ink into one blob.
      4. Find connected components; each large one is a view bounding box.

    Returns: list of (x1, y1, x2, y2) in pixel coords. The original ``binary`` is what
    we crop from later — the morphology is only used to localize the boxes.
    """
    h, w = binary.shape

    # Strip the page frame.
    margin_y = int(h * 0.02)
    margin_x = int(w * 0.02)
    inner = binary.copy()
    inner[:margin_y, :] = 0; inner[-margin_y:, :] = 0
    inner[:, :margin_x] = 0; inner[:, -margin_x:] = 0

    # Erode-then-dilate (opening) with a kernel sized to delete dimension-line/text
    # widths (~2–3 px) while preserving view interiors.
    open_kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(inner, cv2.MORPH_OPEN, open_kernel, iterations=2)

    # Aggressively dilate to bond a view's surviving ink into one connected component.
    # Kernel size ~1.5% of min(h,w) — closes most intra-view gaps without merging views.
    blob_size = max(15, int(min(h, w) * 0.015))
    blob_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (blob_size, blob_size))
    bonded = cv2.dilate(opened, blob_kernel, iterations=2)

    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(bonded, connectivity=8)

    boxes: list[tuple[int, int, int, int]] = []
    page_area = float(h * w)
    for i in range(1, num_labels):  # skip background label 0
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = bw * bh
        if area / page_area < 0.005:
            continue
        boxes.append((int(x), int(y), int(x + bw), int(y + bh)))
    return boxes


def _diagonal_score(binary_region: np.ndarray) -> float:
    """Return the fraction of Hough line segments in the region that are NOT axis-aligned.

    Axis-aligned = within ±_AXIS_DEG_TOL of horizontal or vertical.
    """
    rh, rw = binary_region.shape
    if min(rh, rw) < 30:
        return 0.0
    min_line_len = max(20, int(min(rh, rw) * 0.05))
    lines = cv2.HoughLinesP(
        binary_region, rho=1, theta=np.pi / 180, threshold=30,
        minLineLength=min_line_len, maxLineGap=5,
    )
    if lines is None or len(lines) == 0:
        return 0.0
    total = 0
    diagonal = 0
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if length < min_line_len:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle > 90:
            angle = 180 - angle
        is_axis = angle < _AXIS_DEG_TOL or abs(angle - 90) < _AXIS_DEG_TOL
        total += 1
        if not is_axis:
            diagonal += 1
    if total == 0:
        return 0.0
    return diagonal / total


def find_isometric_bbox(image: Image.Image) -> Optional[tuple[int, int, int, int]]:
    """Return the bbox of the most-likely isometric view, or None if the heuristic fails.

    The chosen view maximizes ``diagonal_score * sqrt(area_fraction)`` — it prefers
    regions that have many diagonal strokes AND are large enough to be a real view
    rather than a callout cluster.
    """
    binary = _binarize(image)
    h, w = binary.shape
    page_area = float(h * w)

    boxes = _split_into_views(binary)
    if not boxes:
        return None

    best: Optional[tuple[float, tuple[int, int, int, int]]] = None
    for box in boxes:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        area_frac = (bw * bh) / page_area
        if area_frac < _MIN_VIEW_AREA_FRAC or area_frac > _MAX_VIEW_AREA_FRAC:
            continue
        # Skip likely title-block / signature regions in the bottom-right.
        cy_frac = ((y1 + y2) / 2) / h
        if cy_frac > _TITLE_BLOCK_BOTTOM_FRAC:
            continue
        region = binary[y1:y2, x1:x2]
        diag = _diagonal_score(region)
        if diag <= 0:
            continue
        score = diag * (area_frac ** 0.5)
        logger.debug(
            "candidate box=(%d,%d,%d,%d) area_frac=%.3f diag=%.3f score=%.4f",
            x1, y1, x2, y2, area_frac, diag, score,
        )
        if best is None or score > best[0]:
            best = (score, box)

    if best is None:
        return None
    return best[1]
