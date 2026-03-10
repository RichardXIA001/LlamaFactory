"""
Detection merging utilities for the split-detect-map-merge pipeline.

Provides IoU computation, class-wise NMS (with optional depth preference),
soft-NMS, and boundary-artefact removal.
"""

import math
from collections import defaultdict
from typing import List, Tuple

from . import Detection


def compute_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Standard IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def merge_detections(
    detections: List[Detection],
    iou_threshold: float = 0.5,
    prefer_deeper: bool = True,
) -> List[Detection]:
    """
    Class-wise NMS with depth preference.

    Strategy:
        1. Group detections by label.
        2. Within each group, sort by (depth DESC, confidence DESC) so
           deeper-level, high-confidence detections are kept first.
        3. Standard greedy NMS within each group.
        4. Return surviving detections.

    Args:
        detections: All detections (from multiple sub-images + full image).
        iou_threshold: Suppression threshold.
        prefer_deeper: If True, deeper-level detections are preferred
            (they typically capture finer detail).

    Returns:
        Surviving detections after NMS.
    """
    if not detections:
        return []

    # Group by label (case-insensitive)
    groups: dict[str, List[Detection]] = defaultdict(list)
    for det in detections:
        groups[det.label.lower()].append(det)

    survivors: List[Detection] = []

    for _label, dets in groups.items():
        # Sort: deeper first (desc), then higher confidence (desc)
        if prefer_deeper:
            dets.sort(key=lambda d: (d.depth, d.confidence), reverse=True)
        else:
            dets.sort(key=lambda d: d.confidence, reverse=True)

        keep: List[Detection] = []
        suppressed = [False] * len(dets)

        for i, det_i in enumerate(dets):
            if suppressed[i]:
                continue
            keep.append(det_i)
            for j in range(i + 1, len(dets)):
                if suppressed[j]:
                    continue
                if compute_iou(det_i.bbox, dets[j].bbox) >= iou_threshold:
                    suppressed[j] = True

        survivors.extend(keep)

    return survivors


def soft_nms(
    detections: List[Detection],
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.05,
) -> List[Detection]:
    """
    Soft-NMS variant that decays confidence instead of hard suppression.

    For every selected detection *M*, every remaining detection *b_i* with
    IoU(*M*, *b_i*) > 0 has its confidence decayed by::

        score *= exp(- iou^2 / sigma)

    Detections whose score drops below ``score_threshold`` are removed.

    Args:
        detections: Input detections.
        iou_threshold: Not used for hard suppression; provided for API
            symmetry.  Decay starts at any IoU > 0.
        sigma: Gaussian decay bandwidth.
        score_threshold: Minimum confidence to keep.

    Returns:
        Surviving detections with updated confidence scores.
    """
    if not detections:
        return []

    # Work on copies so we don't mutate the originals
    remaining = [
        Detection(
            bbox=d.bbox,
            label=d.label,
            confidence=d.confidence,
            depth=d.depth,
            quadrant_id=d.quadrant_id,
        )
        for d in detections
    ]

    kept: List[Detection] = []

    while remaining:
        # Pick the detection with the highest confidence
        best_idx = max(range(len(remaining)), key=lambda i: remaining[i].confidence)
        best = remaining.pop(best_idx)
        kept.append(best)

        new_remaining: List[Detection] = []
        for det in remaining:
            iou = compute_iou(best.bbox, det.bbox)
            if iou > 0:
                decay = math.exp(-(iou ** 2) / sigma)
                det = Detection(
                    bbox=det.bbox,
                    label=det.label,
                    confidence=det.confidence * decay,
                    depth=det.depth,
                    quadrant_id=det.quadrant_id,
                )
            if det.confidence >= score_threshold:
                new_remaining.append(det)
        remaining = new_remaining

    return kept


def remove_boundary_artifacts(
    detections: List[Detection],
    image_size: Tuple[int, int],
    margin_ratio: float = 0.02,
) -> List[Detection]:
    """
    Remove detections that are extremely thin slivers at image edges.

    These are often artefacts from objects cut at quadrant boundaries.
    A bbox with width or height < ``margin_ratio * image_dim`` **and**
    touching the image boundary is removed.

    Args:
        detections: Detection list (global coords).
        image_size: (width, height) of the original image.
        margin_ratio: Threshold ratio.  Boxes thinner than this fraction
            of the image dimension are candidates for removal.

    Returns:
        Filtered detection list.
    """
    img_w, img_h = image_size
    min_w = margin_ratio * img_w
    min_h = margin_ratio * img_h

    kept: List[Detection] = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        bw = x2 - x1
        bh = y2 - y1

        # Check if box is a thin sliver at the edge
        at_left = x1 <= 1
        at_right = x2 >= img_w - 1
        at_top = y1 <= 1
        at_bottom = y2 >= img_h - 1

        # Remove only if it's thin AND touching an edge
        if bw < min_w and (at_left or at_right):
            continue
        if bh < min_h and (at_top or at_bottom):
            continue

        kept.append(det)

    return kept
