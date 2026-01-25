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
Evaluation metrics for 2D grounding tasks.

This module provides functions for computing IoU (Intersection over Union)
and mAP (mean Average Precision) metrics for object detection and grounding tasks.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def calculate_iou(box1: Dict[str, Any], box2: Dict[str, Any]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box with keys 'x1', 'y1', 'x2', 'y2'
        box2: Second bounding box with keys 'x1', 'y1', 'x2', 'y2'

    Returns:
        IoU value between 0 and 1. Returns 0 if boxes don't overlap.

    Example:
        >>> box1 = {"x1": 10, "y1": 10, "x2": 50, "y2": 50}
        >>> box2 = {"x1": 30, "y1": 30, "x2": 70, "y2": 70}
        >>> iou = calculate_iou(box1, box2)
    """
    # Calculate intersection coordinates
    x_left = max(box1["x1"], box2["x1"])
    y_top = max(box1["y1"], box2["y1"])
    x_right = min(box1["x2"], box2["x2"])
    y_bottom = min(box1["y2"], box2["y2"])

    # Check if boxes don't overlap
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate areas of both boxes
    box1_area = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    box2_area = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def calculate_iou_matrix(
    pred_boxes: List[Dict[str, Any]], gt_boxes: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Calculate IoU matrix between predicted and ground truth boxes.

    Args:
        pred_boxes: List of predicted bounding boxes
        gt_boxes: List of ground truth bounding boxes

    Returns:
        Numpy array of shape (len(pred_boxes), len(gt_boxes)) with IoU values
    """
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = calculate_iou(pred_box, gt_box)
    return iou_matrix


def match_boxes(
    pred_boxes: List[Dict[str, Any]],
    gt_boxes: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    label_match: bool = True,
) -> Tuple[List[int], List[int], List[float]]:
    """
    Match predicted boxes to ground truth boxes using IoU threshold.

    Uses greedy matching: each prediction is matched to the highest IoU ground truth
    that hasn't been matched yet and exceeds the IoU threshold.

    Args:
        pred_boxes: List of predicted bounding boxes with 'label' key
        gt_boxes: List of ground truth bounding boxes with 'label' key
        iou_threshold: Minimum IoU for a valid match (default: 0.5)
        label_match: If True, only match boxes with the same label (default: True)

    Returns:
        Tuple of (matched_pred_indices, matched_gt_indices, iou_scores)
        - matched_pred_indices: List of prediction indices that were matched
        - matched_gt_indices: List of corresponding ground truth indices
        - iou_scores: List of IoU scores for each match
    """
    if not pred_boxes or not gt_boxes:
        return [], [], []

    # Calculate IoU matrix
    iou_matrix = calculate_iou_matrix(pred_boxes, gt_boxes)

    # Apply label matching if required
    if label_match:
        for i, pred_box in enumerate(pred_boxes):
            pred_label = pred_box.get("label", "").lower()
            for j, gt_box in enumerate(gt_boxes):
                gt_label = gt_box.get("label", "").lower()
                if pred_label != gt_label:
                    iou_matrix[i, j] = 0.0

    # Greedy matching: sort by IoU and match highest first
    matched_pred_indices = []
    matched_gt_indices = []
    iou_scores = []
    used_gt_indices = set()

    # Create list of (iou, pred_idx, gt_idx) and sort by IoU descending
    matches = []
    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((iou_matrix[i, j], i, j))

    matches.sort(reverse=True, key=lambda x: x[0])

    # Greedy assignment
    for iou, pred_idx, gt_idx in matches:
        if pred_idx not in matched_pred_indices and gt_idx not in used_gt_indices:
            matched_pred_indices.append(pred_idx)
            matched_gt_indices.append(gt_idx)
            iou_scores.append(iou)
            used_gt_indices.add(gt_idx)

    return matched_pred_indices, matched_gt_indices, iou_scores


def compute_precision_recall(
    pred_boxes: List[Dict[str, Any]],
    gt_boxes: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    label_match: bool = True,
) -> Tuple[float, float, int, int, int]:
    """
    Compute precision and recall for a set of predictions.

    Args:
        pred_boxes: List of predicted bounding boxes
        gt_boxes: List of ground truth bounding boxes
        iou_threshold: Minimum IoU for a true positive (default: 0.5)
        label_match: If True, require label match for true positive (default: True)

    Returns:
        Tuple of (precision, recall, true_positives, false_positives, false_negatives)
    """
    matched_pred_indices, matched_gt_indices, _ = match_boxes(
        pred_boxes, gt_boxes, iou_threshold, label_match
    )

    true_positives = len(matched_pred_indices)
    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(gt_boxes) - len(matched_gt_indices)

    precision = true_positives / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
    recall = true_positives / len(gt_boxes) if len(gt_boxes) > 0 else 0.0

    return precision, recall, true_positives, false_positives, false_negatives


def compute_ap(
    pred_boxes: List[Dict[str, Any]],
    gt_boxes: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    label_match: bool = True,
) -> float:
    """
    Compute Average Precision (AP) for a single class using the 11-point interpolation method.

    This follows the Pascal VOC evaluation protocol.

    Args:
        pred_boxes: List of predicted bounding boxes (should be sorted by confidence if available)
        gt_boxes: List of ground truth bounding boxes
        iou_threshold: Minimum IoU for a true positive (default: 0.5)
        label_match: If True, require label match for true positive (default: True)

    Returns:
        Average Precision value between 0 and 1
    """
    if not pred_boxes:
        return 0.0 if gt_boxes else 1.0

    if not gt_boxes:
        return 0.0

    # Sort predictions by confidence if available, otherwise keep order
    if "confidence" in pred_boxes[0] or "score" in pred_boxes[0]:
        sort_key = lambda x: x.get("confidence", x.get("score", 0.0))
        pred_boxes = sorted(pred_boxes, key=sort_key, reverse=True)
    else:
        # If no confidence, assume all have equal confidence
        pred_boxes = pred_boxes.copy()

    # Track which ground truths have been matched
    gt_matched = [False] * len(gt_boxes)
    tp = []  # true positives
    fp = []  # false positives

    # Process each prediction in order of confidence
    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1

        # Find best matching ground truth
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[gt_idx]:
                continue

            # Check label match if required
            if label_match:
                pred_label = pred_box.get("label", "").lower()
                gt_label = gt_box.get("label", "").lower()
                if pred_label != gt_label:
                    continue

            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Determine if this is a true positive or false positive
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp.append(1)
            fp.append(0)
            gt_matched[best_gt_idx] = True
        else:
            tp.append(0)
            fp.append(1)

    # Compute cumulative TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Compute precision and recall at each threshold
    recalls = tp_cumsum / len(gt_boxes) if len(gt_boxes) > 0 else np.zeros_like(tp_cumsum)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum) if len(tp_cumsum + fp_cumsum) > 0 else np.zeros_like(tp_cumsum)

    # 11-point interpolation (Pascal VOC style)
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0

    return float(ap)


def compute_map(
    all_predictions: List[List[Dict[str, Any]]],
    all_ground_truths: List[List[Dict[str, Any]]],
    iou_threshold: float = 0.5,
    label_match: bool = True,
    per_class: bool = False,
) -> Dict[str, float]:
    """
    Compute mean Average Precision (mAP) across multiple images.

    Args:
        all_predictions: List of predictions per image, each is a list of bounding boxes
        all_ground_truths: List of ground truth per image, each is a list of bounding boxes
        iou_threshold: Minimum IoU for a true positive (default: 0.5)
        label_match: If True, require label match for true positive (default: True).
                     If False, boxes are matched regardless of label (box-only evaluation).
        per_class: If True, return mAP per class in addition to overall mAP (default: False).
                   Ignored if label_match=False.

    Returns:
        Dictionary with metrics:
        - 'map': Overall mean Average Precision
        - 'map_per_class': Dictionary of mAP per class (if per_class=True and label_match=True)
        - 'mean_iou': Mean IoU across all matched boxes
        - 'precision': Overall precision
        - 'recall': Overall recall
        - 'num_images': Number of images evaluated

    Note:
        For box-only evaluation (ignoring labels), use label_match=False or use
        compute_map_box_only() for a more convenient interface.
    """
    if len(all_predictions) != len(all_ground_truths):
        raise ValueError(
            f"Number of predictions ({len(all_predictions)}) must match "
            f"number of ground truth sets ({len(all_ground_truths)})"
        )

    # If label_match is False, treat all boxes as a single class (box-only evaluation)
    if not label_match:
        # Flatten all predictions and ground truths across images
        flat_preds = [box for boxes in all_predictions for box in boxes]
        flat_gts = [box for boxes in all_ground_truths for box in boxes]

        if not flat_gts:
            return {
                "map": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "mean_iou": 0.0,
                "num_images": len(all_predictions),
            }

        # Compute AP for all boxes together
        ap = compute_ap(flat_preds, flat_gts, iou_threshold, label_match=False)

        # Compute precision/recall
        precision, recall, tp, fp, fn = compute_precision_recall(
            flat_preds, flat_gts, iou_threshold, label_match=False
        )

        # Compute mean IoU across all matched boxes
        all_ious = []
        for pred_boxes, gt_boxes in zip(all_predictions, all_ground_truths):
            matched_pred_indices, matched_gt_indices, iou_scores = match_boxes(
                pred_boxes, gt_boxes, iou_threshold, label_match=False
            )
            all_ious.extend(iou_scores)

        mean_iou = np.mean(all_ious) if all_ious else 0.0

        return {
            "map": float(ap),
            "precision": float(precision),
            "recall": float(recall),
            "mean_iou": float(mean_iou),
            "num_images": len(all_predictions),
        }

    # Original per-class computation when label_match=True
    # Collect all unique classes
    all_classes = set()
    for gt_boxes in all_ground_truths:
        for box in gt_boxes:
            all_classes.add(box.get("label", "").lower())

    # Compute AP per class
    class_aps = {}
    class_tps = defaultdict(int)
    class_fps = defaultdict(int)
    class_fns = defaultdict(int)
    all_ious = []

    for class_name in all_classes:
        # Filter predictions and ground truth for this class
        class_preds = []
        class_gts = []

        for pred_boxes, gt_boxes in zip(all_predictions, all_ground_truths):
            class_pred = [b for b in pred_boxes if b.get("label", "").lower() == class_name]
            class_gt = [b for b in gt_boxes if b.get("label", "").lower() == class_name]

            if class_pred or class_gt:
                class_preds.append(class_pred)
                class_gts.append(class_gt)

        if class_gts:
            # Flatten for AP computation
            flat_preds = [box for boxes in class_preds for box in boxes]
            flat_gts = [box for boxes in class_gts for box in boxes]

            if flat_gts:
                ap = compute_ap(flat_preds, flat_gts, iou_threshold, label_match=True)
                class_aps[class_name] = ap

                # Compute precision/recall for this class
                _, _, tp, fp, fn = compute_precision_recall(
                    flat_preds, flat_gts, iou_threshold, label_match=True
                )
                class_tps[class_name] = tp
                class_fps[class_name] = fp
                class_fns[class_name] = fn

    # Compute overall metrics
    total_tp = sum(class_tps.values())
    total_fp = sum(class_fps.values())
    total_fn = sum(class_fns.values())

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    # Compute mean IoU across all matched boxes
    for pred_boxes, gt_boxes in zip(all_predictions, all_ground_truths):
        matched_pred_indices, matched_gt_indices, iou_scores = match_boxes(
            pred_boxes, gt_boxes, iou_threshold, label_match=True
        )
        all_ious.extend(iou_scores)

    mean_iou = np.mean(all_ious) if all_ious else 0.0

    # Compute overall mAP (mean of per-class APs)
    overall_map = np.mean(list(class_aps.values())) if class_aps else 0.0

    result = {
        "map": float(overall_map),
        "precision": float(overall_precision),
        "recall": float(overall_recall),
        "mean_iou": float(mean_iou),
        "num_images": len(all_predictions),
    }

    if per_class:
        result["map_per_class"] = {k: float(v) for k, v in class_aps.items()}
        result["precision_per_class"] = {
            k: float(class_tps[k] / (class_tps[k] + class_fps[k]))
            if (class_tps[k] + class_fps[k]) > 0
            else 0.0
            for k in class_aps.keys()
        }
        result["recall_per_class"] = {
            k: float(class_tps[k] / (class_tps[k] + class_fns[k]))
            if (class_tps[k] + class_fns[k]) > 0
            else 0.0
            for k in class_aps.keys()
        }

    return result


def compute_map_box_only(
    all_predictions: List[List[Dict[str, Any]]],
    all_ground_truths: List[List[Dict[str, Any]]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute mAP for box-only evaluation (ignoring labels/categories).

    This is useful for datasets like SKU110K where you only care about whether
    bounding boxes are correctly detected, regardless of category classification.

    Args:
        all_predictions: List of predictions per image, each is a list of bounding boxes.
                         Labels are ignored - only box coordinates matter.
        all_ground_truths: List of ground truth per image, each is a list of bounding boxes.
        iou_threshold: Minimum IoU for a true positive (default: 0.5)

    Returns:
        Dictionary with metrics:
        - 'map': Mean Average Precision (box-only)
        - 'precision': Overall precision
        - 'recall': Overall recall
        - 'mean_iou': Mean IoU across all matched boxes
        - 'num_images': Number of images evaluated

    Example:
        >>> # For SKU110K dataset - only care about box detection, not categories
        >>> predictions = [
        ...     [{"x1": 10, "y1": 10, "x2": 50, "y2": 50, "label": "product"}],
        ...     [{"x1": 20, "y1": 20, "x2": 60, "y2": 60, "label": "item"}],
        ... ]
        >>> ground_truths = [
        ...     [{"x1": 12, "y1": 12, "x2": 52, "y2": 52, "label": "sku"}],
        ...     [{"x1": 22, "y1": 22, "x2": 62, "y2": 62, "label": "product"}],
        ... ]
        >>> metrics = compute_map_box_only(predictions, ground_truths, iou_threshold=0.5)
        >>> print(f"mAP (box-only): {metrics['map']:.4f}")
    """
    return compute_map(
        all_predictions, all_ground_truths, iou_threshold=iou_threshold, label_match=False, per_class=False
    )


def compute_map_at_iou_thresholds(
    all_predictions: List[List[Dict[str, Any]]],
    all_ground_truths: List[List[Dict[str, Any]]],
    iou_thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    label_match: bool = True,
) -> Dict[str, Any]:
    """
    Compute mAP at multiple IoU thresholds (COCO-style evaluation).

    Args:
        all_predictions: List of predictions per image
        all_ground_truths: List of ground truth per image
        iou_thresholds: List of IoU thresholds to evaluate (default: 0.5 to 0.95 in 0.05 steps)
        label_match: If True, require label match for true positive (default: True).
                     If False, boxes are matched regardless of label (box-only evaluation).

    Returns:
        Dictionary with:
        - 'map_50': mAP at IoU=0.5
        - 'map_75': mAP at IoU=0.75
        - 'map': Average mAP across all thresholds (COCO-style)
        - 'map_per_threshold': Dictionary of mAP at each threshold
    """
    map_per_threshold = {}
    for threshold in iou_thresholds:
        metrics = compute_map(all_predictions, all_ground_truths, threshold, label_match, per_class=False)
        map_per_threshold[f"map_{int(threshold*100)}"] = metrics["map"]

    # COCO-style mAP: average across IoU thresholds from 0.5 to 0.95
    coco_map = np.mean(list(map_per_threshold.values()))

    result = {
        "map": float(coco_map),  # COCO-style mAP
        "map_50": map_per_threshold.get("map_50", 0.0),
        "map_75": map_per_threshold.get("map_75", 0.0),
        "map_per_threshold": map_per_threshold,
    }

    return result


def compute_map_box_only_at_iou_thresholds(
    all_predictions: List[List[Dict[str, Any]]],
    all_ground_truths: List[List[Dict[str, Any]]],
    iou_thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
) -> Dict[str, Any]:
    """
    Compute mAP at multiple IoU thresholds for box-only evaluation (ignoring labels).

    This is useful for datasets like SKU110K where you only care about box detection quality,
    regardless of category classification.

    Args:
        all_predictions: List of predictions per image (labels are ignored)
        all_ground_truths: List of ground truth per image
        iou_thresholds: List of IoU thresholds to evaluate (default: 0.5 to 0.95 in 0.05 steps)

    Returns:
        Dictionary with:
        - 'map': Average mAP across all thresholds (COCO-style, box-only)
        - 'map_50': mAP at IoU=0.5 (box-only)
        - 'map_75': mAP at IoU=0.75 (box-only)
        - 'map_per_threshold': Dictionary of mAP at each threshold (box-only)

    Example:
        >>> # For SKU110K - evaluate box detection quality across multiple IoU thresholds
        >>> metrics = compute_map_box_only_at_iou_thresholds(predictions, ground_truths)
        >>> print(f"mAP@[0.5:0.95]: {metrics['map']:.4f}")
        >>> print(f"mAP@0.5: {metrics['map_50']:.4f}")
    """
    return compute_map_at_iou_thresholds(
        all_predictions, all_ground_truths, iou_thresholds=iou_thresholds, label_match=False
    )

# **************************** Example Usage ****************************
if __name__ == "__main__":
    """
    Simple example demonstrating how to use the metrics functions.
    """
    print("=" * 70)
    print("2D Grounding Metrics - Usage Examples")
    print("=" * 70)

    # Example 1: Basic IoU calculation
    print("\n1. Basic IoU Calculation")
    print("-" * 70)
    box1 = {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "label": "object"}
    box2 = {"x1": 30, "y1": 30, "x2": 70, "y2": 70, "label": "object"}
    box3 = {"x1": 100, "y1": 100, "x2": 150, "y2": 150, "label": "object"}  # No overlap

    iou_12 = calculate_iou(box1, box2)
    iou_13 = calculate_iou(box1, box3)

    print(f"Box1: {box1}")
    print(f"Box2: {box2}")
    print(f"Box3: {box3}")
    print(f"\nIoU(box1, box2): {iou_12:.4f} (overlapping boxes)")
    print(f"IoU(box1, box3): {iou_13:.4f} (non-overlapping boxes)")

    # Example 2: Precision and Recall (single image)
    print("\n2. Precision and Recall (Single Image)")
    print("-" * 70)
    pred_boxes = [
        {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "label": "product"},
        {"x1": 60, "y1": 60, "x2": 100, "y2": 100, "label": "product"},
        {"x1": 200, "y1": 200, "x2": 250, "y2": 250, "label": "product"},  # False positive
    ]
    gt_boxes = [
        {"x1": 12, "y1": 12, "x2": 52, "y2": 52, "label": "product"},  # Matches pred[0]
        {"x1": 62, "y1": 62, "x2": 102, "y2": 102, "label": "product"},  # Matches pred[1]
        {"x1": 300, "y1": 300, "x2": 350, "y2": 350, "label": "product"},  # False negative
    ]

    precision, recall, tp, fp, fn = compute_precision_recall(
        pred_boxes, gt_boxes, iou_threshold=0.5, label_match=True
    )

    print(f"Predictions: {len(pred_boxes)} boxes")
    print(f"Ground Truth: {len(gt_boxes)} boxes")
    print(f"\nTrue Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Example 3: Box-only evaluation (ignoring labels) - SKU110K style
    print("\n3. Box-Only Evaluation (Ignoring Labels) - SKU110K Style")
    print("-" * 70)
    # Same boxes but with different labels - should still match based on IoU only
    pred_boxes_no_label = [
        {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "label": "product"},  # Will match gt[0]
        {"x1": 60, "y1": 60, "x2": 100, "y2": 100, "label": "item"},  # Will match gt[1] (different label!)
        {"x1": 200, "y1": 200, "x2": 250, "y2": 250, "label": "sku"},  # False positive
    ]
    gt_boxes_no_label = [
        {"x1": 12, "y1": 12, "x2": 52, "y2": 52, "label": "object"},  # Different label but matches pred[0]
        {"x1": 62, "y1": 62, "x2": 102, "y2": 102, "label": "product"},  # Different label but matches pred[1]
        {"x1": 300, "y1": 300, "x2": 350, "y2": 350, "label": "item"},  # False negative
    ]

    precision_box_only, recall_box_only, tp_box, fp_box, fn_box = compute_precision_recall(
        pred_boxes_no_label, gt_boxes_no_label, iou_threshold=0.5, label_match=False
    )

    print("Note: Labels are different but boxes match based on IoU only!")
    print(f"Predictions: {len(pred_boxes_no_label)} boxes (with various labels)")
    print(f"Ground Truth: {len(gt_boxes_no_label)} boxes (with different labels)")
    print(f"\nTrue Positives: {tp_box}")
    print(f"False Positives: {fp_box}")
    print(f"False Negatives: {fn_box}")
    print(f"Precision (box-only): {precision_box_only:.4f}")
    print(f"Recall (box-only): {recall_box_only:.4f}")

    # Example 4: mAP calculation across multiple images
    print("\n4. mAP Calculation (Multiple Images)")
    print("-" * 70)
    all_predictions = [
        [  # Image 1
            {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "label": "product"},
            {"x1": 60, "y1": 60, "x2": 100, "y2": 100, "label": "product"},
        ],
        [  # Image 2
            {"x1": 20, "y1": 20, "x2": 60, "y2": 60, "label": "product"},
            {"x1": 200, "y1": 200, "x2": 250, "y2": 250, "label": "product"},  # False positive
        ],
    ]
    all_ground_truths = [
        [  # Image 1
            {"x1": 12, "y1": 12, "x2": 52, "y2": 52, "label": "product"},
            {"x1": 62, "y1": 62, "x2": 102, "y2": 102, "label": "product"},
            {"x1": 300, "y1": 300, "x2": 350, "y2": 350, "label": "product"},  # False negative
        ],
        [  # Image 2
            {"x1": 22, "y1": 22, "x2": 62, "y2": 62, "label": "product"},
        ],
    ]

    # With label matching
    metrics_with_labels = compute_map(
        all_predictions, all_ground_truths, iou_threshold=0.5, label_match=True, per_class=False
    )
    print("With label matching:")
    print(f"  mAP: {metrics_with_labels['map']:.4f}")
    print(f"  Precision: {metrics_with_labels['precision']:.4f}")
    print(f"  Recall: {metrics_with_labels['recall']:.4f}")
    print(f"  Mean IoU: {metrics_with_labels['mean_iou']:.4f}")

    # Box-only (ignoring labels)
    metrics_box_only = compute_map_box_only(all_predictions, all_ground_truths, iou_threshold=0.5)
    print("\nBox-only (ignoring labels):")
    print(f"  mAP: {metrics_box_only['map']:.4f}")
    print(f"  Precision: {metrics_box_only['precision']:.4f}")
    print(f"  Recall: {metrics_box_only['recall']:.4f}")
    print(f"  Mean IoU: {metrics_box_only['mean_iou']:.4f}")

    # Example 5: COCO-style mAP at multiple IoU thresholds
    print("\n5. COCO-style mAP at Multiple IoU Thresholds (Box-Only)")
    print("-" * 70)
    coco_metrics = compute_map_box_only_at_iou_thresholds(
        all_predictions, all_ground_truths, iou_thresholds=[0.5, 0.75]
    )
    print("Box-only evaluation across IoU thresholds:")
    print(f"  mAP@[0.5:0.95]: {coco_metrics['map']:.4f}")
    print(f"  mAP@0.5: {coco_metrics['map_50']:.4f}")
    print(f"  mAP@0.75: {coco_metrics['map_75']:.4f}")
    print(f"  Per-threshold mAP: {coco_metrics['map_per_threshold']}")

    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)
