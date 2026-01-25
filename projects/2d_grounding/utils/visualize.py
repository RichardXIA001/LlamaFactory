# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Visualization utilities for 2D grounding detection results.

This module provides functions to visualize bounding boxes from detection results
on images, supporting both predictions and ground truth annotations.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# Handle imports for both module and standalone script usage
try:
    from .draw_utils import draw_labeled_boxes
    from .image_utils import load_image
except ImportError:
    # If running as standalone script, add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.draw_utils import draw_labeled_boxes
    from utils.image_utils import load_image


def visualize_detection_results(
    results_json_path: str,
    images_dir: str,
    output_dir: str,
    draw_predictions: bool = True,
    draw_ground_truth: bool = True,
    prediction_color: Tuple[int, int, int] = (255, 0, 0),  # Red
    ground_truth_color: Tuple[int, int, int] = (0, 255, 0),  # Green
    line_width: int = 3,
    show_labels: bool = True,
    output_suffix: str = "_detected",
    verbose: bool = True,
) -> List[str]:
    """
    Visualize detection results by drawing bounding boxes on images.

    Args:
        results_json_path: Path to JSON file containing detection results
        images_dir: Directory containing source images
        output_dir: Directory to save visualized images
        draw_predictions: Whether to draw prediction boxes (default: True)
        draw_ground_truth: Whether to draw ground truth boxes (default: True)
        prediction_color: RGB color for prediction boxes (default: Red)
        ground_truth_color: RGB color for ground truth boxes (default: Green)
        line_width: Width of bounding box lines (default: 3)
        show_labels: Whether to show labels on boxes (default: True)
        output_suffix: Suffix to add to output image filenames (default: "_detected")
        verbose: Print progress messages (default: True)

    Returns:
        List of paths to saved visualization images

    Example:
        >>> results = visualize_detection_results(
        ...     results_json_path="eval_results/detailed_results.json",
        ...     images_dir="/path/to/images",
        ...     output_dir="/path/to/output",
        ...     draw_predictions=True,
        ...     draw_ground_truth=True
        ... )
    """
    # Load detection results
    results_json_path = Path(results_json_path)
    if not results_json_path.exists():
        raise FileNotFoundError(f"Results JSON not found: {results_json_path}")

    with open(results_json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not isinstance(results, list):
        raise ValueError(f"Expected list in JSON file, got {type(results)}")

    # Setup directories
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    saved_images = []

    if verbose:
        print(f"Processing {len(results)} images...")

    for idx, result in enumerate(results, 1):
        image_name = result.get("image", "")
        if not image_name:
            if verbose:
                print(f"Warning: Skipping result {idx} - no image name")
            continue

        # Load source image
        image_path = images_dir / image_name
        if not image_path.exists():
            if verbose:
                print(f"Warning: Image not found: {image_path}")
            continue

        try:
            image = load_image(str(image_path))
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load image {image_name}: {e}")
            continue

        # Prepare boxes to draw
        boxes_to_draw = []

        # Add predictions if requested
        if draw_predictions:
            predictions = result.get("predictions", [])
            for pred in predictions:
                box = pred.copy()
                if not show_labels:
                    box["label"] = ""  # Remove label if not showing
                boxes_to_draw.append(box)

        # Add ground truth if requested
        if draw_ground_truth:
            ground_truth = result.get("ground_truth", [])
            for gt in ground_truth:
                box = gt.copy()
                if not show_labels:
                    box["label"] = ""  # Remove label if not showing
                boxes_to_draw.append(box)

        # Draw boxes on image
        # Create a copy to avoid modifying the original
        vis_image = image.copy()

        # Draw predictions first (so ground truth appears on top if overlapping)
        if draw_predictions:
            pred_boxes = result.get("predictions", [])
            if pred_boxes:
                # Prepare prediction boxes
                pred_boxes_prep = []
                for pred in pred_boxes:
                    pred_copy = pred.copy()
                    if not show_labels:
                        pred_copy["label"] = ""
                    pred_boxes_prep.append(pred_copy)

                # Draw predictions with custom color
                # draw_labeled_boxes cycles through colors, so we provide the same color for all
                pred_colors = [prediction_color] * len(pred_boxes_prep)
                draw_labeled_boxes(
                    vis_image,
                    pred_boxes_prep,
                    output_path=None,  # Don't save yet
                    colors=pred_colors,
                    line_width=line_width,
                    show_if_no_output=False,
                )

        # Draw ground truth on top
        if draw_ground_truth:
            gt_boxes = result.get("ground_truth", [])
            if gt_boxes:
                # Prepare ground truth boxes
                gt_boxes_prep = []
                for gt in gt_boxes:
                    gt_copy = gt.copy()
                    if not show_labels:
                        gt_copy["label"] = ""
                    gt_boxes_prep.append(gt_copy)

                # Draw ground truth with custom color
                gt_colors = [ground_truth_color] * len(gt_boxes_prep)
                draw_labeled_boxes(
                    vis_image,
                    gt_boxes_prep,
                    output_path=None,  # Don't save yet
                    colors=gt_colors,
                    line_width=line_width,
                    show_if_no_output=False,
                )

        # Save visualized image
        output_filename = Path(image_name).stem + output_suffix + Path(image_name).suffix
        output_path = output_dir / output_filename
        vis_image.save(output_path)
        saved_images.append(str(output_path))

        if verbose and (idx <= 10 or idx % 10 == 0):
            num_pred = result.get("num_predictions", 0)
            num_gt = result.get("num_ground_truth", 0)
            print(f"  [{idx}/{len(results)}] {image_name} -> {output_filename} "
                  f"(pred: {num_pred}, gt: {num_gt})")

    if verbose:
        print(f"\nVisualization complete! Saved {len(saved_images)} images to: {output_dir}")

    return saved_images


def visualize_single_image(
    image_path: str,
    predictions: List[Dict[str, Any]],
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str] = None,
    prediction_color: Tuple[int, int, int] = (255, 0, 0),  # Red
    ground_truth_color: Tuple[int, int, int] = (0, 255, 0),  # Green
    line_width: int = 3,
    show_labels: bool = True,
) -> Image.Image:
    """
    Visualize bounding boxes on a single image.

    Args:
        image_path: Path to source image
        predictions: List of prediction boxes (dicts with x1, y1, x2, y2, label)
        ground_truth: Optional list of ground truth boxes
        output_path: Optional path to save the visualized image
        prediction_color: RGB color for prediction boxes (default: Red)
        ground_truth_color: RGB color for ground truth boxes (default: Green)
        line_width: Width of bounding box lines (default: 3)
        show_labels: Whether to show labels on boxes (default: True)

    Returns:
        PIL Image with bounding boxes drawn

    Example:
        >>> image = visualize_single_image(
        ...     image_path="test_0.jpg",
        ...     predictions=[{"x1": 10, "y1": 10, "x2": 50, "y2": 50, "label": "product"}],
        ...     ground_truth=[{"x1": 12, "y1": 12, "x2": 52, "y2": 52, "label": "product"}],
        ...     output_path="output.jpg"
        ... )
    """
    # Load image
    image = load_image(image_path)
    vis_image = image.copy()

    # Draw predictions
    if predictions:
        pred_boxes = [p.copy() for p in predictions]
        if not show_labels:
            for box in pred_boxes:
                box["label"] = ""

        draw_labeled_boxes(
            vis_image,
            pred_boxes,
            output_path=None,
            colors=[prediction_color] * len(pred_boxes),
            line_width=line_width,
            show_if_no_output=False,
        )

    # Draw ground truth on top
    if ground_truth:
        gt_boxes = [gt.copy() for gt in ground_truth]
        if not show_labels:
            for box in gt_boxes:
                box["label"] = ""

        draw_labeled_boxes(
            vis_image,
            gt_boxes,
            output_path=None,
            colors=[ground_truth_color] * len(gt_boxes),
            line_width=line_width,
            show_if_no_output=False,
        )

    # Save if output path provided
    if output_path:
        vis_image.save(output_path)

    return vis_image


if __name__ == "__main__":
    """
    Example usage: Visualize detection results from evaluation.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize 2D grounding detection results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_json",
        type=str,
        default="/root/Codes/LlamaFactory/2d_grounding/eval_results/detailed_results.json",
        help="Path to detailed results JSON file",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/root/Codes/data/SKU110K_fixed/images",
        help="Directory containing source images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/Codes/data/SKU110K_fixed/detections/test_qwen_4b_zero_shot/img",
        help="Directory to save visualized images",
    )
    parser.add_argument(
        "--draw_predictions",
        action="store_true",
        default=True,
        help="Draw prediction boxes",
    )
    parser.add_argument(
        "--no_predictions",
        action="store_true",
        help="Don't draw prediction boxes",
    )
    parser.add_argument(
        "--draw_ground_truth",
        action="store_true",
        default=True,
        help="Draw ground truth boxes",
    )
    parser.add_argument(
        "--no_ground_truth",
        action="store_true",
        help="Don't draw ground truth boxes",
    )
    parser.add_argument(
        "--prediction_color",
        type=str,
        default="255,0,0",
        help="RGB color for predictions (format: R,G,B)",
    )
    parser.add_argument(
        "--ground_truth_color",
        type=str,
        default="0,255,0",
        help="RGB color for ground truth (format: R,G,B)",
    )
    parser.add_argument(
        "--line_width",
        type=int,
        default=3,
        help="Width of bounding box lines",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="Don't show labels on boxes",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_detected",
        help="Suffix to add to output image filenames",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress messages",
    )

    args = parser.parse_args()

    # Parse colors
    def parse_color(color_str: str) -> Tuple[int, int, int]:
        try:
            r, g, b = map(int, color_str.split(","))
            return (r, g, b)
        except ValueError:
            raise ValueError(f"Invalid color format: {color_str}. Expected R,G,B")

    prediction_color = parse_color(args.prediction_color)
    ground_truth_color = parse_color(args.ground_truth_color)

    # Determine what to draw
    draw_predictions = args.draw_predictions and not args.no_predictions
    draw_ground_truth = args.draw_ground_truth and not args.no_ground_truth

    print("=" * 70)
    print("2D Grounding Detection Visualization")
    print("=" * 70)
    print(f"Results JSON: {args.results_json}")
    print(f"Images directory: {args.images_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Draw predictions: {draw_predictions} (color: {prediction_color})")
    print(f"Draw ground truth: {draw_ground_truth} (color: {ground_truth_color})")
    print(f"Line width: {args.line_width}")
    print(f"Show labels: {not args.no_labels}")
    print("=" * 70)

    # Run visualization
    saved_images = visualize_detection_results(
        results_json_path=args.results_json,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        draw_predictions=draw_predictions,
        draw_ground_truth=draw_ground_truth,
        prediction_color=prediction_color,
        ground_truth_color=ground_truth_color,
        line_width=args.line_width,
        show_labels=not args.no_labels,
        output_suffix=args.output_suffix,
        verbose=args.verbose,
    )

    print("\n" + "=" * 70)
    print(f"Visualization completed! Saved {len(saved_images)} images.")
    print("=" * 70)
