"""
Split → Detect → Map → Merge pipeline for enhanced VLM detection.

This module provides the high-level :func:`split_detect_merge` function
that orchestrates the full pipeline:

1. **Split** the input image into overlapping quadrants.
2. **Detect** objects in each quadrant (and optionally in the full image)
   using a VLM.
3. **Map** each sub-image's detections back to global coordinates.
4. **Merge** all detections via NMS to remove duplicates from overlap
   regions.

Usage (standalone)::

    python -m pipeline.detect \\
        --image /path/to/image.jpg \\
        --model Qwen/Qwen3-VL-4B-Instruct \\
        --output /path/to/output.jpg
"""

import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from . import Detection
from .mapper import clip_bbox, map_to_global
from .merger import merge_detections, remove_boundary_artifacts
from .splitter import split_image_into_quadrants


# ──────────────────────────────────────────────────────────────────────
# Helper: default VLM detect function wrapper
# ──────────────────────────────────────────────────────────────────────

def _default_detect_fn(
    image: np.ndarray,
    model: Any,
    processor: Any,
    device: Any,
    prompt: str,
    max_new_tokens: int = 8192,
) -> List[Detection]:
    """
    Default detection function that wraps the existing Qwen-VL inference
    and parsing utilities.

    Args:
        image: (H, W, 3) numpy RGB array.
        model: Loaded Qwen-VL model.
        processor: Loaded processor.
        device: Torch device string.
        prompt: Detection prompt.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        List of Detection objects with bboxes in **pixel** coordinates
        of the supplied image.
    """
    # Lazy imports so the module can be loaded without torch installed
    # (e.g. for unit-testing the geometry functions alone).
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model.inference import inference_local_qwen3vl
    from utils.parse_utils import parse_bounding_boxes

    h, w = image.shape[:2]
    pil_img = Image.fromarray(image)

    # Save to a temporary in-memory path that the inference API can consume
    import io
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    pil_img.save(tmp, format="JPEG")
    tmp_path = tmp.name
    tmp.close()

    try:
        response = inference_local_qwen3vl(
            model=model,
            processor=processor,
            device=device,
            img_urls=[tmp_path],
            prompts=[prompt],
            max_new_tokens=max_new_tokens,
            verbose=False,
        )
        if isinstance(response, list):
            response = response[0]

        boxes = parse_bounding_boxes(response, w, h)
    finally:
        import os
        os.unlink(tmp_path)

    detections: List[Detection] = []
    for box in boxes:
        detections.append(Detection(
            bbox=(box["x1"], box["y1"], box["x2"], box["y2"]),
            label=box.get("label", ""),
            confidence=1.0,
            depth=0,
            quadrant_id=0,
        ))
    return detections


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────

def split_detect_merge(
    image: np.ndarray,
    detect_fn: Callable[..., List[Detection]],
    overlap_ratio: float = 0.1,
    iou_threshold: float = 0.5,
    include_full_image: bool = True,
    remove_artifacts: bool = True,
    artifact_margin: float = 0.02,
    prefer_deeper: bool = True,
    verbose: bool = True,
) -> List[Detection]:
    """
    Run the full split → detect → map → merge pipeline.

    Args:
        image: (H, W, 3) numpy RGB array — the original full-size image.
        detect_fn: A callable ``(image_np) -> List[Detection]`` that runs
            object detection on a single image array and returns detections
            in the pixel coordinate system of that array.
        overlap_ratio: Overlap between adjacent quadrants (fraction).
        iou_threshold: IoU threshold for NMS merging.
        include_full_image: If True, also run detection on the full image
            at depth 0 and include those detections in the merge.
        remove_artifacts: If True, remove thin boundary slivers after
            merging.
        artifact_margin: Margin ratio for artefact removal.
        prefer_deeper: If True, prefer sub-image detections over full-image
            ones during NMS (they tend to capture more detail).
        verbose: Print progress messages.

    Returns:
        Merged list of Detection objects in global pixel coordinates.
    """
    h, w = image.shape[:2]
    all_detections: List[Detection] = []
    t0 = time.time()

    # ── Step 0 (optional): detect on full image ────────────────────
    if include_full_image:
        if verbose:
            print("[Pipeline] Detecting on full image …")
        full_dets = detect_fn(image)
        # Tag depth=0
        for det in full_dets:
            det.depth = 0
            det.quadrant_id = 0
        all_detections.extend(full_dets)
        if verbose:
            print(f"[Pipeline]   → {len(full_dets)} detections from full image")

    # ── Step 1: split ──────────────────────────────────────────────
    if verbose:
        print(f"[Pipeline] Splitting image ({w}×{h}) into quadrants "
              f"(overlap={overlap_ratio:.0%}) …")
    quadrants = split_image_into_quadrants(image, overlap_ratio)

    # ── Step 2 & 3: detect per quadrant, then map to global ───────
    for qid, qinfo in quadrants.items():
        sub_img = qinfo["image"]
        offset = qinfo["offset"]
        sub_size = qinfo["size"]

        if verbose:
            print(f"[Pipeline] Quadrant {qid}: size={sub_size}, offset={offset}")

        # Detect in sub-image coordinates
        sub_dets = detect_fn(sub_img)

        # Tag depth and quadrant
        for det in sub_dets:
            det.depth = 1
            det.quadrant_id = qid

        # Map to global coordinates
        global_dets = map_to_global(sub_dets, offset, sub_size)

        # Clip to original image boundaries
        clipped: List[Detection] = []
        for det in global_dets:
            cb = clip_bbox(det.bbox, w, h)
            if cb[2] > cb[0] and cb[3] > cb[1]:  # valid box
                clipped.append(Detection(
                    bbox=cb,
                    label=det.label,
                    confidence=det.confidence,
                    depth=det.depth,
                    quadrant_id=det.quadrant_id,
                ))
        all_detections.extend(clipped)

        if verbose:
            print(f"[Pipeline]   → {len(sub_dets)} detections "
                  f"({len(clipped)} after clipping)")

    # ── Step 4: merge ──────────────────────────────────────────────
    if verbose:
        print(f"[Pipeline] Merging {len(all_detections)} total detections "
              f"(IoU threshold={iou_threshold}) …")

    merged = merge_detections(
        all_detections,
        iou_threshold=iou_threshold,
        prefer_deeper=prefer_deeper,
    )

    if verbose:
        print(f"[Pipeline]   → {len(merged)} detections after NMS")

    # ── Step 5 (optional): remove boundary artefacts ───────────────
    if remove_artifacts:
        before = len(merged)
        merged = remove_boundary_artifacts(merged, (w, h), artifact_margin)
        if verbose and len(merged) < before:
            print(f"[Pipeline]   → removed {before - len(merged)} "
                  f"boundary artefacts → {len(merged)} final detections")

    elapsed = time.time() - t0
    if verbose:
        print(f"[Pipeline] Done in {elapsed:.2f}s — "
              f"{len(merged)} final detections")

    return merged


# ──────────────────────────────────────────────────────────────────────
# Convenience wrapper that handles model loading
# ──────────────────────────────────────────────────────────────────────

def run_pipeline(
    image_path: str,
    model: Any = None,
    processor: Any = None,
    device: Any = None,
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
    lora_path: Optional[str] = None,
    is_lora: bool = False,
    prompt: Optional[str] = None,
    overlap_ratio: float = 0.1,
    iou_threshold: float = 0.5,
    include_full_image: bool = True,
    max_new_tokens: int = 8192,
    verbose: bool = True,
) -> Tuple[List[Detection], np.ndarray]:
    """
    Convenience entry-point: load image → (optionally load model) →
    run pipeline → return detections and the original image array.

    Args:
        image_path: Path to the input image.
        model / processor / device: Pre-loaded model components.  If None
            they are loaded from ``model_name``.
        model_name: HuggingFace / ModelScope model id.
        lora_path: Optional LoRA adapter path.
        is_lora: Whether a LoRA adapter is being used.
        prompt: Detection prompt (a sensible default is used if None).
        overlap_ratio: Quadrant overlap.
        iou_threshold: NMS threshold.
        include_full_image: Also detect on the full image?
        max_new_tokens: Max tokens for VLM generation.
        verbose: Print progress.

    Returns:
        (detections, image_np): final merged detections and the loaded
        image as a numpy array.
    """
    prompt = prompt or (
        "Detect and locate every individual object in this image. "
        "Report bounding box coordinates for all items in JSON format."
    )

    # Load image
    img_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(img_pil)

    # Load model if needed
    if model is None:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from model.loader import load_local_qwen3vl_model
        model, processor, device = load_local_qwen3vl_model(
            model_name=model_name,
            lora_path=lora_path,
            is_lora=is_lora,
            device="auto",
        )

    # Build detect_fn closure
    def detect_fn(img_arr: np.ndarray) -> List[Detection]:
        return _default_detect_fn(
            img_arr, model, processor, device, prompt, max_new_tokens,
        )

    detections = split_detect_merge(
        image=image_np,
        detect_fn=detect_fn,
        overlap_ratio=overlap_ratio,
        iou_threshold=iou_threshold,
        include_full_image=include_full_image,
        verbose=verbose,
    )

    return detections, image_np


# ──────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Split-Detect-Map-Merge pipeline for enhanced VLM detection",
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct",
                        help="Model name or path")
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--is_lora", action="store_true")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--overlap", type=float, default=0.1)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--no_full_image", action="store_true",
                        help="Skip detection on the full image")
    parser.add_argument("--output", default=None,
                        help="Path to save visualised result image")
    parser.add_argument("--max_new_tokens", type=int, default=8192)

    args = parser.parse_args()

    detections, image_np = run_pipeline(
        image_path=args.image,
        model_name=args.model,
        lora_path=args.lora_path,
        is_lora=args.is_lora,
        prompt=args.prompt,
        overlap_ratio=args.overlap,
        iou_threshold=args.iou_threshold,
        include_full_image=not args.no_full_image,
        max_new_tokens=args.max_new_tokens,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Detected {len(detections)} objects")
    print(f"{'='*60}")
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        print(f"  [{i:3d}] label={det.label!r:30s}  "
              f"bbox=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})  "
              f"depth={det.depth}  quad={det.quadrant_id}")

    # Visualise if output path given
    if args.output:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils.draw_utils import draw_labeled_boxes

        pil_img = Image.fromarray(image_np)
        boxes = [
            {
                "x1": d.bbox[0], "y1": d.bbox[1],
                "x2": d.bbox[2], "y2": d.bbox[3],
                "label": d.label,
            }
            for d in detections
        ]
        draw_labeled_boxes(pil_img, boxes, output_path=args.output)
        print(f"\nVisualised output saved to: {args.output}")


if __name__ == "__main__":
    main()
