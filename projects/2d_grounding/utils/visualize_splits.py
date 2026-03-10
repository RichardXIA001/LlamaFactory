"""
Visualize recursive split-detect results from the RFT pipeline.

Reads the same YAML config used for rejection sampling generation,
locates the SFT output JSON in ``output_dir``, and produces:

1. **{image}_overview.jpg** — full image with quadrant boundaries
   (color-coded by depth), SPLIT/DETECT labels, detections, and GT.
2. **{image}_panels.jpg**  — grid of per-crop panels with local
   detections and GT.
3. **{image}_tree.txt**    — text recursion tree.

Uses existing utilities:
  - utils.draw_utils.draw_labeled_boxes, load_font
  - utils.image_utils.load_image

Usage:
    python -m utils.visualize_splits configs/rft_test_single.yaml

    # Or visualize only a specific image:
    python -m utils.visualize_splits configs/rft_full.yaml --image_name train_1026.jpg
"""

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Handle imports for both module and standalone script usage
try:
    from .draw_utils import draw_labeled_boxes, load_font
    from .image_utils import load_image
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.draw_utils import draw_labeled_boxes, load_font
    from utils.image_utils import load_image


# ====================================================================
# Color scheme
# ====================================================================

DEPTH_COLORS = {
    0: (255, 255, 0),    # yellow  — full image
    1: (0, 200, 255),    # cyan    — depth 1
    2: (255, 100, 255),  # magenta — depth 2
    3: (255, 160, 0),    # orange  — depth 3
}

SPLIT_COLOR = (255, 50, 50)      # red
DETECT_COLOR = (50, 200, 50)     # green
GT_COLOR = (0, 255, 0)           # green
MERGED_COLOR = (0, 150, 255)     # blue


def _get_depth_color(depth: int) -> Tuple[int, int, int]:
    return DEPTH_COLORS.get(depth, (200, 200, 200))


# ====================================================================
# Drawing helpers
# ====================================================================

def draw_dashed_rect(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[float, float, float, float],
    color: Tuple[int, int, int],
    width: int = 3,
    dash_len: int = 15,
    gap_len: int = 10,
) -> None:
    """Draw a dashed rectangle."""
    x1, y1, x2, y2 = [int(v) for v in bbox]

    def dashed_line(start, end, is_horizontal):
        sx, sy = start
        ex, ey = end
        length = abs(ex - sx) if is_horizontal else abs(ey - sy)
        dx = (1 if ex > sx else -1) if is_horizontal else 0
        dy = (1 if ey > sy else -1) if not is_horizontal else 0
        pos = 0
        while pos < length:
            seg_end = min(pos + dash_len, length)
            px1 = sx + dx * pos
            py1 = sy + dy * pos
            px2 = sx + dx * seg_end
            py2 = sy + dy * seg_end
            draw.line([(px1, py1), (px2, py2)], fill=color, width=width)
            pos += dash_len + gap_len

    dashed_line((x1, y1), (x2, y1), is_horizontal=True)
    dashed_line((x1, y2), (x2, y2), is_horizontal=True)
    dashed_line((x1, y1), (x1, y2), is_horizontal=False)
    dashed_line((x2, y1), (x2, y2), is_horizontal=False)


def draw_label_tag(
    draw: ImageDraw.ImageDraw,
    text: str,
    position: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    """Draw a text tag with a colored background."""
    x, y = position
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([x, y, x + tw + 8, y + th + 6], fill=bg_color)
    draw.text((x + 4, y + 3), text, fill=(255, 255, 255), font=font)


# ====================================================================
# Full-image overview
# ====================================================================

def visualize_full_image(
    image: Image.Image,
    sft_examples: List[Dict],
    gt_boxes: Optional[List[Dict]] = None,
    output_path: Optional[str] = None,
    draw_gt: bool = True,
    draw_detections: bool = True,
    line_width: int = 2,
) -> Image.Image:
    """Full image with quadrant boundaries, SPLIT/DETECT tags, detections, GT."""
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    font_small = load_font(14)
    img_w, img_h = vis.size

    # GT boxes (background layer)
    if draw_gt and gt_boxes:
        for box in gt_boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            draw.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=1)

    # Collect detections from detect-nodes and map to global coords
    all_det_boxes = []
    for ex in sft_examples:
        if ex["target"] == "<SPLIT>":
            continue
        crop = ex["crop_region"]
        cx1, cy1, cx2, cy2 = crop
        cw, ch = cx2 - cx1, cy2 - cy1
        try:
            dets = json.loads(ex["target"])
        except (json.JSONDecodeError, TypeError):
            continue
        for det in dets:
            b = det.get("bbox", det.get("bbox_2d", []))
            if len(b) < 4:
                continue
            gx1 = cx1 + b[0] / 1000.0 * cw
            gy1 = cy1 + b[1] / 1000.0 * ch
            gx2 = cx1 + b[2] / 1000.0 * cw
            gy2 = cy1 + b[3] / 1000.0 * ch
            all_det_boxes.append({"x1": gx1, "y1": gy1, "x2": gx2, "y2": gy2})

    # Draw detections
    if draw_detections:
        for box in all_det_boxes:
            x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
            draw.rectangle([x1, y1, x2, y2], outline=MERGED_COLOR, width=line_width)

    # Draw quadrant boundaries + labels (sorted by depth, deeper on top)
    for ex in sorted(sft_examples, key=lambda e: e["depth"]):
        crop = ex["crop_region"]
        depth = ex["depth"]
        is_split = ex["target"] == "<SPLIT>"
        qpath = ex.get("quadrant_path", ex.get("quadrant_id", "?"))

        color = _get_depth_color(depth)
        draw_dashed_rect(draw, crop, color, width=max(2, 4 - depth))

        action_str = "SPLIT" if is_split else "DETECT"
        tag_color = SPLIT_COLOR if is_split else DETECT_COLOR
        tag_text = f"d{depth} {qpath} {action_str}"
        tag_x = int(crop[0]) + 4
        tag_y = int(crop[1]) + 4 + depth * 24
        draw_label_tag(draw, tag_text, (tag_x, tag_y), tag_color, font_small)

    # Legend
    legend_y = img_h - 100
    legend_x = 10
    draw.rectangle([legend_x, legend_y, legend_x + 320, img_h - 5], fill=(0, 0, 0, 180))
    if draw_gt and gt_boxes:
        draw.rectangle([legend_x + 5, legend_y + 5, legend_x + 25, legend_y + 15],
                        outline=GT_COLOR, width=2)
        draw.text((legend_x + 30, legend_y + 3), "Ground Truth", fill=GT_COLOR, font=font_small)
    if draw_detections:
        draw.rectangle([legend_x + 5, legend_y + 22, legend_x + 25, legend_y + 32],
                        outline=MERGED_COLOR, width=2)
        draw.text((legend_x + 30, legend_y + 20), "Detections", fill=MERGED_COLOR, font=font_small)
    draw_label_tag(draw, "SPLIT", (legend_x + 5, legend_y + 40), SPLIT_COLOR, font_small)
    draw_label_tag(draw, "DETECT", (legend_x + 85, legend_y + 40), DETECT_COLOR, font_small)
    for d in range(max(e["depth"] for e in sft_examples) + 1):
        c = _get_depth_color(d)
        dx = legend_x + 5 + d * 80
        draw.rectangle([dx, legend_y + 65, dx + 15, legend_y + 75], outline=c, width=2)
        draw.text((dx + 20, legend_y + 63), f"depth {d}", fill=c, font=font_small)

    if output_path:
        vis.save(output_path)
        print(f"  Saved full-image overview: {output_path}")
    return vis


# ====================================================================
# Per-crop panel grid
# ====================================================================

def visualize_crop_panels(
    image: Image.Image,
    sft_examples: List[Dict],
    gt_boxes: Optional[List[Dict]] = None,
    output_path: Optional[str] = None,
    max_panel_size: int = 600,
    line_width: int = 2,
) -> Image.Image:
    """Grid of crop panels — one per recursion node."""
    font = load_font(16)
    font_header = load_font(20)
    header_h = 32

    panels = []
    for ex in sorted(sft_examples, key=lambda e: (e["depth"], str(e.get("quadrant_path", "")))):
        crop = ex["crop_region"]
        cx1, cy1, cx2, cy2 = [int(v) for v in crop]
        cw, ch = cx2 - cx1, cy2 - cy1
        if cw <= 0 or ch <= 0:
            continue

        crop_img = image.crop((cx1, cy1, cx2, cy2)).copy()
        scale = min(max_panel_size / cw, max_panel_size / ch, 1.0)
        new_w, new_h = int(cw * scale), int(ch * scale)
        if scale < 1.0:
            crop_img = crop_img.resize((new_w, new_h), Image.LANCZOS)

        panel = Image.new("RGB", (new_w, new_h + header_h), (40, 40, 40))
        panel.paste(crop_img, (0, header_h))
        pdraw = ImageDraw.Draw(panel)

        # Header
        is_split = ex["target"] == "<SPLIT>"
        qpath = ex.get("quadrant_path", ex.get("quadrant_id", "?"))
        header_color = SPLIT_COLOR if is_split else DETECT_COLOR
        pdraw.rectangle([0, 0, new_w, header_h], fill=header_color)
        pdraw.text((6, 6), f"d{ex['depth']} {qpath}  {'SPLIT' if is_split else 'DETECT'}",
                   fill=(255, 255, 255), font=font_header)

        # Local GT boxes
        if gt_boxes:
            for gt in gt_boxes:
                gcx = (gt["x1"] + gt["x2"]) / 2
                gcy = (gt["y1"] + gt["y2"]) / 2
                if not (cx1 <= gcx <= cx2 and cy1 <= gcy <= cy2):
                    continue
                lx1 = int((gt["x1"] - cx1) * scale)
                ly1 = int((gt["y1"] - cy1) * scale) + header_h
                lx2 = int((gt["x2"] - cx1) * scale)
                ly2 = int((gt["y2"] - cy1) * scale) + header_h
                pdraw.rectangle([lx1, ly1, lx2, ly2], outline=GT_COLOR, width=1)

        # Local detections (detect-nodes only)
        if not is_split:
            try:
                dets = json.loads(ex["target"])
            except (json.JSONDecodeError, TypeError):
                dets = []
            for det in dets:
                b = det.get("bbox", det.get("bbox_2d", []))
                if len(b) < 4:
                    continue
                lx1 = int(b[0] / 1000.0 * new_w)
                ly1 = int(b[1] / 1000.0 * new_h) + header_h
                lx2 = int(b[2] / 1000.0 * new_w)
                ly2 = int(b[3] / 1000.0 * new_h) + header_h
                pdraw.rectangle([lx1, ly1, lx2, ly2], outline=MERGED_COLOR, width=line_width)
            pdraw.text((6, header_h + 4), f"{len(dets)} detections",
                       fill=(255, 255, 100), font=font)

        panels.append(panel)

    if not panels:
        return Image.new("RGB", (400, 200), (0, 0, 0))

    # Grid layout
    cols = min(len(panels), 4)
    rows = math.ceil(len(panels) / cols)
    max_pw = max(p.width for p in panels)
    max_ph = max(p.height for p in panels)
    pad = 6
    grid = Image.new("RGB", (cols * (max_pw + pad) + pad, rows * (max_ph + pad) + pad), (30, 30, 30))
    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        grid.paste(panel, (pad + c * (max_pw + pad), pad + r * (max_ph + pad)))

    if output_path:
        grid.save(output_path)
        print(f"  Saved crop panels: {output_path}")
    return grid


# ====================================================================
# Recursion tree text
# ====================================================================

def build_recursion_tree(sft_examples: List[Dict]) -> str:
    """Build a text recursion tree."""
    lines = []
    for ex in sorted(sft_examples, key=lambda e: (e["depth"], str(e.get("quadrant_path", "")))):
        depth = ex["depth"]
        is_split = ex["target"] == "<SPLIT>"
        qpath = ex.get("quadrant_path", ex.get("quadrant_id", "?"))
        crop = ex["crop_region"]
        cw, ch = int(crop[2] - crop[0]), int(crop[3] - crop[1])

        indent = "  " * depth
        if is_split:
            detail = "-> recurse into 4 quadrants"
        else:
            try:
                n = len(json.loads(ex["target"]))
            except Exception:
                n = "?"
            detail = f"-> {n} detections"

        lines.append(f"{indent}[d{depth}] {qpath}  ({cw}x{ch})  "
                     f"{'SPLIT' if is_split else 'DETECT'}  {detail}")
    return "\n".join(lines)


# ====================================================================
# Process one image
# ====================================================================

def visualize_single_image_splits(
    image_name: str,
    sft_examples: List[Dict],
    images_dir: str,
    gt_boxes: Optional[List[Dict]],
    output_dir: str,
    draw_gt: bool = True,
) -> None:
    """Generate all visualizations for one image."""
    img_path = os.path.join(images_dir, image_name)
    if not os.path.exists(img_path):
        print(f"  WARNING: image not found: {img_path}")
        return

    image = load_image(img_path, convert_mode="RGB")
    stem = Path(image_name).stem

    print(f"\n{'='*60}")
    print(f"Image: {image_name}  ({image.size[0]}x{image.size[1]})")
    print(f"SFT examples: {len(sft_examples)}")

    tree_str = build_recursion_tree(sft_examples)
    print(f"\nRecursion tree:\n{tree_str}")

    # Save tree text
    with open(os.path.join(output_dir, f"{stem}_tree.txt"), "w") as f:
        f.write(f"Image: {image_name}\n\n{tree_str}\n")

    # Full-image overview
    visualize_full_image(
        image=image,
        sft_examples=sft_examples,
        gt_boxes=gt_boxes,
        output_path=os.path.join(output_dir, f"{stem}_overview.jpg"),
        draw_gt=draw_gt,
    )

    # Crop panels
    visualize_crop_panels(
        image=image,
        sft_examples=sft_examples,
        gt_boxes=gt_boxes,
        output_path=os.path.join(output_dir, f"{stem}_panels.jpg"),
    )


# ====================================================================
# Config loading + main
# ====================================================================

def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Visualize recursive split-detect results from RFT pipeline",
    )
    parser.add_argument("config", help="Path to the same YAML config used for rejection sampling")
    parser.add_argument("--image_name", default=None,
                        help="Visualize only this image (e.g. train_1026.jpg). "
                             "If not set, visualizes all images in the SFT JSON.")
    parser.add_argument("--no_gt", action="store_true",
                        help="Don't draw ground truth boxes")
    args = parser.parse_args()

    # --- Load config ---
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg.get("output_dir", "./rft_results")
    images_dir = cfg.get("images_dir", "")
    low_ap_json = cfg.get("low_ap_json")
    test_image = cfg.get("test_image")

    # --- Locate the SFT output JSON ---
    # Convention: test_single_image_sft.json for single-image mode,
    #             sft_rft_training_data.json for full pipeline mode
    sft_json_path = None
    for candidate in [
        os.path.join(output_dir, "test_single_image_sft.json"),
        os.path.join(output_dir, "sft_rft_training_data.json"),
    ]:
        if os.path.exists(candidate):
            sft_json_path = candidate
            break

    if sft_json_path is None:
        print(f"ERROR: No SFT output JSON found in {output_dir}")
        print(f"  Looked for: test_single_image_sft.json, sft_rft_training_data.json")
        print(f"  Run rejection sampling first with this config.")
        return

    # --- For single-image mode, resolve images_dir from test_image path ---
    if test_image and not images_dir:
        images_dir = str(Path(test_image).parent)

    if not images_dir:
        print("ERROR: Cannot determine images directory. "
              "Set 'images_dir' or 'test_image' in the config.")
        return

    # --- Load SFT data ---
    with open(sft_json_path) as f:
        sft_data = json.load(f)
    print(f"Loaded {len(sft_data)} SFT examples from {sft_json_path}")

    by_image: Dict[str, List[Dict]] = defaultdict(list)
    for ex in sft_data:
        by_image[ex["image_name"]].append(ex)
    print(f"Images: {len(by_image)}")

    # --- Load GT (optional) ---
    gt_by_image: Dict[str, List[Dict]] = {}
    if low_ap_json and not args.no_gt and os.path.exists(low_ap_json):
        with open(low_ap_json) as f:
            for entry in json.load(f):
                gt_by_image[entry["image"]] = entry.get("ground_truth", [])
        print(f"Loaded GT for {len(gt_by_image)} images")

    # --- Determine which images to visualize ---
    if args.image_name:
        if args.image_name not in by_image:
            print(f"ERROR: '{args.image_name}' not found in SFT data.")
            print(f"Available: {list(by_image.keys())[:10]}")
            return
        image_names = [args.image_name]
    else:
        image_names = list(by_image.keys())

    # --- Output dir for visualizations ---
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # --- Visualize ---
    for name in image_names:
        gt = gt_by_image.get(name)
        visualize_single_image_splits(
            image_name=name,
            sft_examples=by_image[name],
            images_dir=images_dir,
            gt_boxes=gt,
            output_dir=vis_dir,
            draw_gt=not args.no_gt and gt is not None,
        )

    print(f"\n{'='*60}")
    print(f"All done! Outputs saved to: {vis_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()