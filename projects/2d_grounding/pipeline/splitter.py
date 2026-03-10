"""
Image splitting utilities for the split-detect-map-merge pipeline.

Splits an image into quadrants (2×2 grid) with configurable overlap
so that objects near boundaries appear in at least one sub-image intact.
"""

from typing import Dict, Tuple

import numpy as np


def get_quadrant_boundaries(
    img_w: int,
    img_h: int,
    overlap_ratio: float = 0.1,
) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Pure math version — returns (x1, y1, x2, y2) for each quadrant
    without actually cropping.  Useful for visualization and testing.

    The image is split at the centre, and each cut edge is padded by
    ``overlap_ratio * dimension`` pixels so neighbouring quadrants
    share a strip along the boundary.

    Args:
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        overlap_ratio: Fraction of width/height to pad on each cut edge.

    Returns:
        Dict mapping quadrant id (1-4) to (x1, y1, x2, y2):
            1 → top-left, 2 → top-right,
            3 → bottom-left, 4 → bottom-right.
    """
    mid_x = img_w // 2
    mid_y = img_h // 2

    pad_x = int(img_w * overlap_ratio)
    pad_y = int(img_h * overlap_ratio)

    boundaries = {
        1: (0, 0, min(mid_x + pad_x, img_w), min(mid_y + pad_y, img_h)),        # top-left
        2: (max(mid_x - pad_x, 0), 0, img_w, min(mid_y + pad_y, img_h)),        # top-right
        3: (0, max(mid_y - pad_y, 0), min(mid_x + pad_x, img_w), img_h),        # bottom-left
        4: (max(mid_x - pad_x, 0), max(mid_y - pad_y, 0), img_w, img_h),        # bottom-right
    }
    return boundaries


def split_image_into_quadrants(
    image: np.ndarray,
    overlap_ratio: float = 0.1,
) -> Dict[int, Dict]:
    """
    Split image into 4 quadrants with overlap padding.

    Args:
        image: (H, W, 3) numpy array.
        overlap_ratio: Fraction of width/height to pad on each cut edge.

    Returns:
        Dict mapping quadrant id (1-4) to::

            {
                "image": np.ndarray,          # cropped sub-image
                "offset": (x_offset, y_offset),  # top-left corner in global coords
                "size": (w, h),               # sub-image dimensions
            }
    """
    h, w = image.shape[:2]
    boundaries = get_quadrant_boundaries(w, h, overlap_ratio)

    quadrants: Dict[int, Dict] = {}
    for qid, (x1, y1, x2, y2) in boundaries.items():
        crop = image[y1:y2, x1:x2].copy()
        quadrants[qid] = {
            "image": crop,
            "offset": (x1, y1),
            "size": (x2 - x1, y2 - y1),
        }
    return quadrants
