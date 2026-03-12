"""
Test the split-detect-merge inference pipeline WITHOUT a trained model.

Mocks the VLM to simulate different split/detect behaviors, then runs
the full pipeline (split → crop → map → merge → evaluate) to verify
correctness.

Three test modes:
  1. always_split:  depth 0 returns <SPLIT>, depth 1+ returns detections
  2. never_split:   all depths return detections (baseline, no splitting)
  3. mixed:         some quadrants split, others detect

Uses real images and real ground truth so you can see actual AP numbers
and verify the coordinate mapping is correct.

Usage:
    python test_split_pipeline.py \
        --image /path/to/image.jpg \
        --low_ap_json map_under20_images.json \
        --output_dir ./test_pipeline_output

    # Or test on multiple images:
    python test_split_pipeline.py \
        --images_dir /path/to/images \
        --low_ap_json map_under20_images.json \
        --image_prefix train_ \
        --max_images 5 \
        --output_dir ./test_pipeline_output
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Also add the project root (2d_grounding/) in case this script lives in a subdirectory like eval/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import Detection
from pipeline.splitter import split_image_into_quadrants, get_quadrant_boundaries
from pipeline.mapper import map_to_global, clip_bbox
from pipeline.merger import merge_detections, remove_boundary_artifacts, compute_iou
from utils.image_utils import load_image
from utils.parse_utils import parse_bounding_boxes
from utils.draw_utils import draw_labeled_boxes


# ====================================================================
# AP computation (standalone, no torch needed)
# ====================================================================

def compute_ap(preds, gts, iou_thr=0.5):
    """preds/gts: list of dicts with x1,y1,x2,y2."""
    if not gts:
        return {"ap": 1.0 if not preds else 0.0, "recall": 1.0, "precision": 1.0, "matched": 0}
    if not preds:
        return {"ap": 0.0, "recall": 0.0, "precision": 0.0, "matched": 0}

    gt_matched = [False] * len(gts)
    tp, fp = [], []
    for p in preds:
        pb = (p["x1"], p["y1"], p["x2"], p["y2"])
        best_iou, best_i = 0, -1
        for gi, g in enumerate(gts):
            if gt_matched[gi]:
                continue
            gb = (g["x1"], g["y1"], g["x2"], g["y2"])
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou; best_i = gi
        if best_iou >= iou_thr and best_i >= 0:
            tp.append(1); fp.append(0); gt_matched[best_i] = True
        else:
            tp.append(0); fp.append(1)

    tc = np.cumsum(tp).astype(float)
    fc = np.cumsum(fp).astype(float)
    rec = tc / len(gts)
    prec = tc / (tc + fc)
    for i in range(len(prec)-1, 0, -1):
        prec[i-1] = max(prec[i-1], prec[i])
    ap, pr = 0.0, 0.0
    for i in range(len(rec)):
        ap += (rec[i] - pr) * prec[i]; pr = rec[i]
    return {"ap": float(ap), "recall": float(rec[-1]), "precision": float(prec[-1]),
            "matched": int(sum(gt_matched))}


# ====================================================================
# Mock VLM that returns <SPLIT> or fake detections from GT
# ====================================================================

class MockVLM:
    """Simulates VLM output for testing the pipeline.

    For detect-nodes: looks up GT boxes that fall in the crop region and
    returns them as if the VLM detected them (with optional noise/dropout).

    For split-nodes: returns the split_token string.
    """

    def __init__(
        self,
        gt_boxes: List[Dict],
        img_w: int,
        img_h: int,
        split_token: str = "<SPLIT>",
        mode: str = "always_split",  # always_split, never_split, mixed
        detection_recall: float = 0.85,
        bbox_noise_std: float = 5.0,
        mixed_split_quadrants: Optional[List[int]] = None,
    ):
        self.gt_boxes = gt_boxes
        self.img_w = img_w
        self.img_h = img_h
        self.split_token = split_token
        self.mode = mode
        self.detection_recall = detection_recall
        self.bbox_noise_std = bbox_noise_std
        self.mixed_split_quadrants = mixed_split_quadrants or [1, 2]
        self.call_log = []

    def __call__(self, crop_np: np.ndarray, region: Tuple[float, ...], depth: int,
                 quadrant_id: int) -> str:
        """Return a mock VLM response string."""
        rx1, ry1, rx2, ry2 = region
        rw, rh = rx2 - rx1, ry2 - ry1
        h_crop, w_crop = crop_np.shape[:2]

        self.call_log.append({
            "depth": depth, "quadrant_id": quadrant_id,
            "region": [rx1, ry1, rx2, ry2],
            "crop_size": [w_crop, h_crop],
        })

        # Decide split or detect
        should_split = False
        if self.mode == "always_split" and depth == 0:
            should_split = True
        elif self.mode == "mixed" and depth == 0:
            should_split = True
        elif self.mode == "mixed" and depth == 1:
            should_split = quadrant_id in self.mixed_split_quadrants

        if should_split:
            return self.split_token

        # Generate mock detections from GT boxes in this region
        dets = []
        for gt in self.gt_boxes:
            cx = (gt["x1"] + gt["x2"]) / 2
            cy = (gt["y1"] + gt["y2"]) / 2
            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                continue
            # Recall dropout
            if np.random.random() > self.detection_recall:
                continue
            # Map GT to crop-local pixel coords, then to 0-1000 Qwen space
            lx1 = (gt["x1"] - rx1) / rw * 1000
            ly1 = (gt["y1"] - ry1) / rh * 1000
            lx2 = (gt["x2"] - rx1) / rw * 1000
            ly2 = (gt["y2"] - ry1) / rh * 1000
            # Add noise
            lx1 += np.random.normal(0, self.bbox_noise_std)
            ly1 += np.random.normal(0, self.bbox_noise_std)
            lx2 += np.random.normal(0, self.bbox_noise_std)
            ly2 += np.random.normal(0, self.bbox_noise_std)
            # Clip to 0-1000
            lx1 = max(0, min(1000, int(round(lx1))))
            ly1 = max(0, min(1000, int(round(ly1))))
            lx2 = max(0, min(1000, int(round(lx2))))
            ly2 = max(0, min(1000, int(round(ly2))))
            if lx2 > lx1 and ly2 > ly1:
                dets.append({"bbox_2d": [lx1, ly1, lx2, ly2], "label": "object"})

        # Format as the VLM would output
        json_str = json.dumps(dets, indent=2)
        return f"```json\n{json_str}\n```"


# ====================================================================
# Pipeline runner (mirrors inference_with_split logic)
# ====================================================================

def run_split_pipeline(
    image_np: np.ndarray,
    mock_vlm: MockVLM,
    parse_fn: Callable,
    split_token: str = "<SPLIT>",
    max_depth: int = 2,
    overlap_ratio: float = 0.1,
    iou_threshold: float = 0.5,
    prefer_deeper: bool = True,
    remove_artifacts: bool = True,
) -> Dict[str, Any]:
    """Run the full split-detect-merge pipeline with a mock VLM."""
    img_h, img_w = image_np.shape[:2]
    num_calls = 0

    def _recurse(crop_np, region, depth, quad_id, qpath):
        nonlocal num_calls
        rx1, ry1, rx2, ry2 = region
        rw, rh = int(rx2 - rx1), int(ry2 - ry1)
        h_crop, w_crop = crop_np.shape[:2]

        response = mock_vlm(crop_np, region, depth, quad_id)
        num_calls += 1

        tree_node = {
            "depth": depth, "quadrant_id": quad_id, "quadrant_path": qpath,
            "region": [rx1, ry1, rx2, ry2], "action": "detect",
        }

        if split_token in response and depth < max_depth:
            tree_node["action"] = "split"
            tree_node["children"] = {}

            quadrants = split_image_into_quadrants(crop_np, overlap_ratio)
            all_child_dets = []

            for child_qid, qinfo in quadrants.items():
                child_crop = qinfo["image"]
                ox, oy = qinfo["offset"]
                cw, ch = qinfo["size"]
                child_region = (rx1 + ox, ry1 + oy, rx1 + ox + cw, ry1 + oy + ch)
                child_path = f"{qpath}-{child_qid}"

                child_dets, child_tree = _recurse(
                    child_crop, child_region, depth + 1, child_qid, child_path,
                )
                tree_node["children"][child_qid] = child_tree
                all_child_dets.extend(child_dets)

            return all_child_dets, tree_node
        else:
            boxes = parse_fn(response, w_crop, h_crop)
            dets_raw = [
                Detection(
                    bbox=(b["x1"], b["y1"], b["x2"], b["y2"]),
                    label=b.get("label", "object"), confidence=1.0,
                    depth=depth, quadrant_id=quad_id,
                )
                for b in boxes
            ]
            global_dets = map_to_global(dets_raw, offset=(int(rx1), int(ry1)), sub_size=(rw, rh))
            clipped = []
            for det in global_dets:
                cb = clip_bbox(det.bbox, img_w, img_h)
                if cb[2] > cb[0] and cb[3] > cb[1]:
                    clipped.append(Detection(
                        bbox=cb, label=det.label, confidence=det.confidence,
                        depth=det.depth, quadrant_id=det.quadrant_id,
                    ))
            tree_node["num_detections"] = len(clipped)
            return clipped, tree_node

    full_region = (0.0, 0.0, float(img_w), float(img_h))
    all_dets, split_tree = _recurse(image_np, full_region, 0, 0, "0")

    merged = merge_detections(all_dets, iou_threshold=iou_threshold, prefer_deeper=prefer_deeper)
    if remove_artifacts:
        merged = remove_boundary_artifacts(merged, (img_w, img_h))

    result_boxes = [
        {"x1": d.bbox[0], "y1": d.bbox[1], "x2": d.bbox[2], "y2": d.bbox[3], "label": d.label}
        for d in merged
    ]
    return {
        "detections": result_boxes,
        "raw_detections_before_nms": len(all_dets),
        "split_tree": split_tree,
        "num_vlm_calls": num_calls,
    }


# ====================================================================
# Tree printer
# ====================================================================

def print_tree(tree, indent=0):
    prefix = "  " * indent
    action = tree.get("action", "?")
    qpath = tree.get("quadrant_path", "?")
    region = tree.get("region", [])
    region_str = f"[{int(region[0])},{int(region[1])},{int(region[2])},{int(region[3])}]" if region else "[]"

    if action == "split":
        print(f"{prefix}[d{tree['depth']}] {qpath} {region_str}  SPLIT")
        for qid in sorted(tree.get("children", {}).keys()):
            print_tree(tree["children"][qid], indent + 1)
    else:
        n = tree.get("num_detections", "?")
        print(f"{prefix}[d{tree['depth']}] {qpath} {region_str}  DETECT -> {n} dets")


# ====================================================================
# Visualization
# ====================================================================

def visualize_result(image_path, detections, gt_boxes, output_path, split_tree=None):
    """Draw detections (blue) and GT (green) on the image."""
    from utils.draw_utils import load_font
    from PIL import ImageDraw as _ImageDraw
    img = load_image(image_path, convert_mode="RGB")
    draw = _ImageDraw.Draw(img)
    font = load_font(12)

    # GT in green
    for gt in gt_boxes:
        draw.rectangle([gt["x1"], gt["y1"], gt["x2"], gt["y2"]], outline=(0, 255, 0), width=1)

    # Detections in blue
    for det in detections:
        draw.rectangle([int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])],
                       outline=(0, 100, 255), width=2)

    # Draw split boundaries if available
    if split_tree:
        _draw_split_boundaries(draw, split_tree, font)

    img.save(output_path)
    print(f"  Visualization saved: {output_path}")


def _draw_split_boundaries(draw, tree, font, depth_colors=None):
    if depth_colors is None:
        depth_colors = {0: (255, 255, 0), 1: (0, 200, 255), 2: (255, 100, 255)}

    region = tree.get("region", [])
    if not region:
        return

    depth = tree.get("depth", 0)
    color = depth_colors.get(depth, (200, 200, 200))
    x1, y1, x2, y2 = [int(v) for v in region]

    if tree.get("action") == "split":
        # Draw dashed boundary
        for i in range(x1, x2, 20):
            draw.line([(i, y1), (min(i+10, x2), y1)], fill=color, width=2)
            draw.line([(i, y2), (min(i+10, x2), y2)], fill=color, width=2)
        for i in range(y1, y2, 20):
            draw.line([(x1, i), (x1, min(i+10, y2))], fill=color, width=2)
            draw.line([(x2, i), (x2, min(i+10, y2))], fill=color, width=2)

        # Label
        action_color = (255, 50, 50)
        tag = f"d{depth} SPLIT"
        draw.rectangle([x1+2, y1+2+depth*18, x1+len(tag)*7+6, y1+16+depth*18], fill=action_color)
        draw.text((x1+4, y1+3+depth*18), tag, fill=(255, 255, 255), font=font)

        for child_tree in tree.get("children", {}).values():
            _draw_split_boundaries(draw, child_tree, font, depth_colors)


# ====================================================================
# Main test runner
# ====================================================================

def test_single_image(
    image_path: str,
    gt_boxes: List[Dict],
    mode: str = "always_split",
    max_depth: int = 2,
    overlap_ratio: float = 0.1,
    iou_threshold: float = 0.5,
    detection_recall: float = 0.85,
    bbox_noise_std: float = 5.0,
    output_dir: Optional[str] = None,
):
    """Run the pipeline test on a single image in a given mode."""
    image_np = np.array(load_image(image_path, convert_mode="RGB"))
    img_h, img_w = image_np.shape[:2]
    image_name = Path(image_path).stem

    print(f"\n{'='*60}")
    print(f"Image: {Path(image_path).name}  ({img_w}x{img_h})")
    print(f"Mode: {mode}  max_depth={max_depth}  overlap={overlap_ratio}")
    print(f"GT boxes: {len(gt_boxes)}")

    mock = MockVLM(
        gt_boxes=gt_boxes, img_w=img_w, img_h=img_h,
        mode=mode, detection_recall=detection_recall,
        bbox_noise_std=bbox_noise_std,
    )

    t0 = time.time()
    result = run_split_pipeline(
        image_np=image_np, mock_vlm=mock, parse_fn=parse_bounding_boxes,
        max_depth=max_depth, overlap_ratio=overlap_ratio,
        iou_threshold=iou_threshold,
    )
    elapsed = time.time() - t0

    # Print tree
    print(f"\nRecursion tree:")
    print_tree(result["split_tree"])

    # Print stats
    dets = result["detections"]
    print(f"\nResults:")
    print(f"  VLM calls:              {result['num_vlm_calls']}")
    print(f"  Raw dets (before NMS):  {result['raw_detections_before_nms']}")
    print(f"  Final dets (after NMS): {len(dets)}")
    print(f"  Time:                   {elapsed:.3f}s")

    # Compute AP
    ap_result = compute_ap(dets, gt_boxes, iou_threshold)
    print(f"  AP@{iou_threshold}:               {ap_result['ap']:.4f}")
    print(f"  Precision:              {ap_result['precision']:.4f}")
    print(f"  Recall:                 {ap_result['recall']:.4f}")
    print(f"  Matched:                {ap_result['matched']} / {len(gt_boxes)}")

    # VLM call breakdown
    print(f"\nVLM call log:")
    for call in mock.call_log:
        r = call["region"]
        print(f"  d{call['depth']} q{call['quadrant_id']}  "
              f"region=[{int(r[0])},{int(r[1])},{int(r[2])},{int(r[3])}]  "
              f"crop={call['crop_size']}")

    # Visualize
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        vis_path = os.path.join(output_dir, f"{image_name}_{mode}.jpg")
        visualize_result(image_path, dets, gt_boxes, vis_path, result["split_tree"])

    return {
        "mode": mode, "detections": dets, "split_tree": result["split_tree"],
        "num_vlm_calls": result["num_vlm_calls"], "ap": ap_result,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test the split-detect-merge pipeline with mocked VLM",
    )
    parser.add_argument("--image", default=None, help="Single image path")
    parser.add_argument("--images_dir", default=None, help="Directory of images")
    parser.add_argument("--image_prefix", default="train_")
    parser.add_argument("--max_images", type=int, default=3)
    parser.add_argument("--low_ap_json", required=True,
                        help="Path to map_under20_images.json (for GT boxes)")
    parser.add_argument("--output_dir", default="./test_pipeline_output")
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--overlap_ratio", type=float, default=0.1)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--detection_recall", type=float, default=0.85,
                        help="Simulated VLM recall (fraction of GT boxes detected)")
    parser.add_argument("--bbox_noise", type=float, default=5.0,
                        help="Simulated bbox noise in 0-1000 Qwen coords")
    args = parser.parse_args()

    # Load GT
    with open(args.low_ap_json) as f:
        low_ap_data = json.load(f)
    gt_by_image = {e["image"]: e["ground_truth"] for e in low_ap_data}

    # Collect images to test
    if args.image:
        image_paths = [args.image]
    elif args.images_dir:
        images_dir = Path(args.images_dir)
        image_paths = [str(p) for p in sorted(images_dir.glob(f"{args.image_prefix}*.jpg"))[:args.max_images]]
    else:
        # Use images from low_ap_json
        available = [e["image"] for e in low_ap_data[:args.max_images]]
        # Try to find them
        for candidate_dir in [Path("."), Path("./images"), Path("/root/Codes/data/SKU110K_fixed/images")]:
            if candidate_dir.exists():
                found = [str(candidate_dir / name) for name in available if (candidate_dir / name).exists()]
                if found:
                    image_paths = found[:args.max_images]
                    break
        else:
            print("ERROR: No images found. Use --image or --images_dir.")
            return

    if not image_paths:
        print("ERROR: No images found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Run all three modes on each image
    modes = ["never_split", "always_split", "mixed"]
    all_results = {}

    for img_path in image_paths:
        img_name = Path(img_path).name
        gt = gt_by_image.get(img_name, [])
        if not gt:
            print(f"\nSkipping {img_name}: no GT boxes found in low_ap_json")
            continue

        all_results[img_name] = {}
        for mode in modes:
            np.random.seed(42)  # Reproducible noise
            result = test_single_image(
                image_path=img_path, gt_boxes=gt, mode=mode,
                max_depth=args.max_depth, overlap_ratio=args.overlap_ratio,
                iou_threshold=args.iou_threshold,
                detection_recall=args.detection_recall,
                bbox_noise_std=args.bbox_noise,
                output_dir=args.output_dir,
            )
            all_results[img_name][mode] = {
                "ap": result["ap"]["ap"],
                "recall": result["ap"]["recall"],
                "precision": result["ap"]["precision"],
                "num_dets": len(result["detections"]),
                "num_vlm_calls": result["num_vlm_calls"],
            }

    # Summary comparison
    print(f"\n\n{'='*70}")
    print("SUMMARY: Mode Comparison")
    print(f"{'='*70}")
    print(f"{'Image':<25s} {'Mode':<15s} {'AP':>8s} {'Recall':>8s} {'Dets':>6s} {'Calls':>6s}")
    print("-" * 70)
    for img_name, modes_data in all_results.items():
        for mode, data in modes_data.items():
            print(f"{img_name:<25s} {mode:<15s} {data['ap']:>8.4f} {data['recall']:>8.4f} "
                  f"{data['num_dets']:>6d} {data['num_vlm_calls']:>6d}")
        print()

    # Save summary
    summary_path = os.path.join(args.output_dir, "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    print(f"\nVisualizations saved to: {args.output_dir}/")
    print("  Green boxes = Ground Truth")
    print("  Blue boxes = Detections (after NMS)")
    print("  Dashed lines = Split boundaries (color-coded by depth)")


if __name__ == "__main__":
    main()