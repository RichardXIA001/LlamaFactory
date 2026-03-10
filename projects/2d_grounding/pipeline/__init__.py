"""
Split-Detect-Map-Merge pipeline for enhanced VLM object detection.

The pipeline works by:
1. Splitting an image into sub-regions (quadrants with overlap)
2. Running VLM detection on each sub-region independently
3. Mapping sub-region detections back to global image coordinates
4. Merging overlapping detections via NMS

This enhances detection of small/dense objects that VLMs may miss
when processing the full image at once.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Detection:
    """A single detection with bounding box, label, confidence, and depth metadata.

    Attributes:
        bbox: (x1, y1, x2, y2) in pixel coordinates.
        label: Class label string.
        confidence: Detection confidence score in [0, 1].
        depth: Depth level where this detection was produced.
               0 = full image, 1 = first-level split, etc.
        quadrant_id: Which sub-image produced this detection (0 = full image).
    """
    bbox: Tuple[float, float, float, float]
    label: str = ""
    confidence: float = 1.0
    depth: int = 0
    quadrant_id: int = 0
