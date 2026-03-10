"""
Coordinate mapping utilities for the split-detect-map-merge pipeline.

Converts detection bounding boxes between:
  • sub-image local pixel coordinates
  • Qwen-VL 0-1000 normalised coordinates
  • global (original image) pixel coordinates
"""

from typing import List, Optional, Tuple

from . import Detection


# ──────────────────────────────────────────────────────────────────────
# Qwen-VL normalisation helpers
# ──────────────────────────────────────────────────────────────────────

def normalize_bbox_qwen(
    bbox: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    """
    Convert pixel coords → Qwen-VL 0-1000 normalised coords.
    Used when generating SFT target labels.

    Args:
        bbox: (x1, y1, x2, y2) in pixel coordinates.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        (nx1, ny1, nx2, ny2) each in [0, 1000].
    """
    x1, y1, x2, y2 = bbox
    nx1 = int(round(x1 / img_w * 1000))
    ny1 = int(round(y1 / img_h * 1000))
    nx2 = int(round(x2 / img_w * 1000))
    ny2 = int(round(y2 / img_h * 1000))
    return (
        max(0, min(nx1, 1000)),
        max(0, min(ny1, 1000)),
        max(0, min(nx2, 1000)),
        max(0, min(ny2, 1000)),
    )


def denormalize_bbox_qwen(
    bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    """
    Convert Qwen-VL 0-1000 normalised coords → pixel coords.
    Used when parsing model output.

    Args:
        bbox: (nx1, ny1, nx2, ny2) each in [0, 1000].
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        (x1, y1, x2, y2) in pixel coordinates.
    """
    nx1, ny1, nx2, ny2 = bbox
    x1 = nx1 / 1000.0 * img_w
    y1 = ny1 / 1000.0 * img_h
    x2 = nx2 / 1000.0 * img_w
    y2 = ny2 / 1000.0 * img_h
    return (x1, y1, x2, y2)


# ──────────────────────────────────────────────────────────────────────
# Bbox clipping
# ──────────────────────────────────────────────────────────────────────

def clip_bbox(
    bbox: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    """Clip bbox to image boundaries.  Needed after mapping."""
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(x1), float(img_w)))
    y1 = max(0.0, min(float(y1), float(img_h)))
    x2 = max(0.0, min(float(x2), float(img_w)))
    y2 = max(0.0, min(float(y2), float(img_h)))
    return (x1, y1, x2, y2)


# ──────────────────────────────────────────────────────────────────────
# Global ↔ local mapping
# ──────────────────────────────────────────────────────────────────────

def map_to_global(
    detections: List[Detection],
    offset: Tuple[int, int],
    sub_size: Tuple[int, int],
    model_input_size: Optional[Tuple[int, int]] = None,
) -> List[Detection]:
    """
    Convert detections from sub-image local coords to global image coords.

    Handles two cases:

    1. *model_input_size is None*: detections are already in pixel coords
       of the sub-image → simply shift by offset.
    2. *model_input_size is set*: detections are in the model's normalised
       space (e.g. Qwen-VL 0-1000).  We first denormalise to sub-image
       pixel coords, then shift by offset.

    Args:
        detections: List of Detection objects with local bboxes.
        offset: (x_off, y_off) of the sub-image origin in global coords.
        sub_size: (w, h) of the sub-image in pixels.
        model_input_size: If the model resizes input to a fixed resolution,
            supply (model_w, model_h).  For Qwen-VL use (1000, 1000).

    Returns:
        New Detection list with bboxes in global pixel coordinates.
    """
    x_off, y_off = offset
    sub_w, sub_h = sub_size
    mapped: List[Detection] = []

    for det in detections:
        x1, y1, x2, y2 = det.bbox

        if model_input_size is not None:
            # Denormalise from model space → sub-image pixel space
            mw, mh = model_input_size
            x1 = x1 / mw * sub_w
            y1 = y1 / mh * sub_h
            x2 = x2 / mw * sub_w
            y2 = y2 / mh * sub_h

        # Shift to global coords
        global_bbox = (x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off)

        mapped.append(Detection(
            bbox=global_bbox,
            label=det.label,
            confidence=det.confidence,
            depth=det.depth,
            quadrant_id=det.quadrant_id,
        ))

    return mapped


def map_to_local(
    detections: List[Detection],
    offset: Tuple[int, int],
    sub_size: Tuple[int, int],
    model_input_size: Optional[Tuple[int, int]] = None,
) -> List[Detection]:
    """
    Inverse of :func:`map_to_global`.  Used when building SFT data.

    Takes detections in global pixel coords and maps them into the
    local coordinate system of the sub-image (optionally normalised
    to the model's input space).

    Args:
        detections: List of Detection objects with global bboxes.
        offset: (x_off, y_off) of the sub-image origin in global coords.
        sub_size: (w, h) of the sub-image in pixels.
        model_input_size: If set, normalise from sub-image pixel space to
            the model's input resolution.

    Returns:
        New Detection list with bboxes in local (or model-normalised) coords.
    """
    x_off, y_off = offset
    sub_w, sub_h = sub_size
    mapped: List[Detection] = []

    for det in detections:
        x1, y1, x2, y2 = det.bbox

        # Shift from global to sub-image pixel coords
        lx1 = x1 - x_off
        ly1 = y1 - y_off
        lx2 = x2 - x_off
        ly2 = y2 - y_off

        # Clip to sub-image boundaries
        lx1 = max(0.0, min(float(lx1), float(sub_w)))
        ly1 = max(0.0, min(float(ly1), float(sub_h)))
        lx2 = max(0.0, min(float(lx2), float(sub_w)))
        ly2 = max(0.0, min(float(ly2), float(sub_h)))

        # Skip if the box doesn't intersect the sub-image
        if lx2 <= lx1 or ly2 <= ly1:
            continue

        if model_input_size is not None:
            mw, mh = model_input_size
            lx1 = lx1 / sub_w * mw
            ly1 = ly1 / sub_h * mh
            lx2 = lx2 / sub_w * mw
            ly2 = ly2 / sub_h * mh

        mapped.append(Detection(
            bbox=(lx1, ly1, lx2, ly2),
            label=det.label,
            confidence=det.confidence,
            depth=det.depth,
            quadrant_id=det.quadrant_id,
        ))

    return mapped
