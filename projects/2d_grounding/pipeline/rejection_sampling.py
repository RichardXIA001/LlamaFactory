"""
Rejection Sampling Fine-tuning (RFT) for Recursive VLM Object Detection.

Key performance features:
  - **Batch VLM inference**: all crops at the same depth are batched into
    one ``inference_local_qwen3vl`` call instead of 1-by-1.
  - **Detection cache**: identical crop regions share a single VLM call.
    With greedy decoding (temperature=0), the full image and every quadrant
    is inferred exactly once regardless of how many trajectories sample it.
  - **Checkpoint / resume**: results are saved after each image.  On restart
    the pipeline skips already-completed images automatically.
  - **Multi-GPU**: images are distributed across GPUs via
    ``torch.multiprocessing``.  Each worker loads its own model copy on a
    dedicated GPU.

Usage:
    # Single-image test:
    python -m pipeline.rejection_sampling rft_test_single.yaml

    # Full pipeline (auto multi-GPU if num_gpus > 1):
    python -m pipeline.rejection_sampling rft_full.yaml
"""

import json
import os
import hashlib
import random
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from . import Detection
from .splitter import get_quadrant_boundaries, split_image_into_quadrants
from .mapper import (
    map_to_global,
    map_to_local,
    normalize_bbox_qwen,
    denormalize_bbox_qwen,
    clip_bbox,
)
from .merger import compute_iou, merge_detections


# ====================================================================
# Data Structures
# ====================================================================

@dataclass
class TrajectoryNode:
    depth: int
    quadrant_id: int
    quadrant_path: str
    region: Tuple[float, float, float, float]
    action: str = "detect"
    direct_detections: List[Detection] = field(default_factory=list)
    children: Dict[int, "TrajectoryNode"] = field(default_factory=dict)


@dataclass
class Trajectory:
    image_name: str
    root: Optional[TrajectoryNode] = None
    merged_detections: List[Detection] = field(default_factory=list)
    ap_score: float = 0.0
    recall: float = 0.0


@dataclass
class SFTExample:
    image_name: str
    crop_region: List[float]
    depth: int
    quadrant_path: str
    prompt: str
    target: str
    image_w: float = 0.0
    image_h: float = 0.0


# ====================================================================
# AP Evaluation
# ====================================================================

def compute_ap(
    predictions: List[Detection],
    ground_truths: List[Detection],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    if not ground_truths:
        return {"ap": 1.0 if not predictions else 0.0,
                "precision": 1.0, "recall": 1.0, "num_matched": 0}
    if not predictions:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0, "num_matched": 0}

    preds_sorted = sorted(predictions, key=lambda d: d.confidence, reverse=True)
    gt_matched = [False] * len(ground_truths)
    tp, fp = [], []

    for pred in preds_sorted:
        best_iou, best_idx = 0.0, -1
        for gi, gt in enumerate(ground_truths):
            if gt_matched[gi]:
                continue
            iou = compute_iou(pred.bbox, gt.bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = gi
        if best_iou >= iou_threshold and best_idx >= 0:
            tp.append(1); fp.append(0)
            gt_matched[best_idx] = True
        else:
            tp.append(0); fp.append(1)

    tp_cum = np.cumsum(tp).astype(float)
    fp_cum = np.cumsum(fp).astype(float)
    recalls = tp_cum / len(ground_truths)
    precisions = tp_cum / (tp_cum + fp_cum)

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    ap, prev_r = 0.0, 0.0
    for i in range(len(recalls)):
        ap += (recalls[i] - prev_r) * precisions[i]
        prev_r = recalls[i]

    return {"ap": float(ap), "precision": float(precisions[-1]),
            "recall": float(recalls[-1]), "num_matched": int(sum(gt_matched))}


# ====================================================================
# Ground-truth helpers
# ====================================================================

def gt_boxes_in_region(
    gt_list: List[Detection],
    region: Tuple[float, float, float, float],
) -> List[Detection]:
    rx1, ry1, rx2, ry2 = region
    result: List[Detection] = []
    for gt in gt_list:
        gx1, gy1, gx2, gy2 = gt.bbox
        cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
        if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
            continue
        cx1 = max(gx1, rx1); cy1 = max(gy1, ry1)
        cx2 = min(gx2, rx2); cy2 = min(gy2, ry2)
        if cx2 > cx1 and cy2 > cy1:
            result.append(Detection(bbox=(cx1, cy1, cx2, cy2),
                                    label=gt.label, confidence=gt.confidence))
    return result


# ====================================================================
# Batch VLM wrapper + detection cache
# ====================================================================

class BatchDetector:
    """Wraps VLM inference with batching and an on-disk detection cache.

    Instead of calling the VLM once per crop, this class:
      1. Collects all crop requests via ``request()``.
      2. Runs them all in one batched ``inference_local_qwen3vl`` call
         via ``flush()``.
      3. Caches results by crop region key so identical crops across
         trajectories are never re-inferred.
      4. Persists the cache to disk after each flush so we can resume.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: Any,
        prompt: str,
        parse_fn: Callable,
        max_new_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float = 1.0,
        batch_size: int = 8,
        cache_path: Optional[str] = None,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.prompt = prompt
        self.parse_fn = parse_fn
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size

        # In-memory cache: region_key -> List[Detection]
        self._cache: Dict[str, List[Detection]] = {}
        self._cache_path = cache_path

        # Pending requests: list of (region_key, crop_np, img_w, img_h)
        self._pending: List[Tuple[str, np.ndarray, int, int]] = []

        # Load existing cache from disk
        if cache_path and os.path.exists(cache_path):
            self._load_cache()

    # --- Cache key ---
    @staticmethod
    def region_key(region: Tuple[float, float, float, float],
                   traj_idx: int = 0,
                   use_sampling: bool = False) -> str:
        """Unique key for a crop region.

        With greedy decoding (use_sampling=False), the key is just the
        region coordinates — all trajectories share the same result.
        With sampling, each trajectory gets its own key.
        """
        r = tuple(round(v, 1) for v in region)
        if use_sampling:
            return f"{r}_{traj_idx}"
        return str(r)

    # --- Request / lookup ---
    def get_cached(self, key: str) -> Optional[List[Detection]]:
        return self._cache.get(key)

    def request(self, key: str, crop_np: np.ndarray, img_w: int, img_h: int) -> None:
        """Queue a crop for batch inference (if not already cached)."""
        if key not in self._cache:
            # Avoid duplicate pending requests
            if not any(p[0] == key for p in self._pending):
                self._pending.append((key, crop_np, img_w, img_h))

    def flush(self) -> None:
        """Run batched VLM inference on all pending requests."""
        if not self._pending:
            return

        from model.inference import inference_local_qwen3vl

        # Save crops to temp files
        tmp_paths = []
        keys = []
        widths = []
        heights = []

        for key, crop_np, w, h in self._pending:
            pil_img = Image.fromarray(crop_np)
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            pil_img.save(tmp, format="JPEG")
            tmp.close()
            tmp_paths.append(tmp.name)
            keys.append(key)
            widths.append(w)
            heights.append(h)

        try:
            do_sample = self.temperature > 0
            prompts = [self.prompt] * len(tmp_paths)

            responses = inference_local_qwen3vl(
                model=self.model,
                processor=self.processor,
                device=self.device,
                img_urls=tmp_paths,
                prompts=prompts,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else 1.0,
                top_p=self.top_p if do_sample else 1.0,
                repetition_penalty=1.1,
                batch_size=self.batch_size,
                verbose=False,
            )

            if isinstance(responses, str):
                responses = [responses]

            for key, resp, w, h in zip(keys, responses, widths, heights):
                boxes = self.parse_fn(resp, w, h)
                dets = [
                    Detection(
                        bbox=(b["x1"], b["y1"], b["x2"], b["y2"]),
                        label=b.get("label", "object"),
                        confidence=b.get("confidence", 1.0),
                    )
                    for b in boxes
                ]
                self._cache[key] = dets

        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

        self._pending.clear()
        self._save_cache()

    # --- Disk persistence ---
    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        data = {}
        for key, dets in self._cache.items():
            data[key] = [
                {"bbox": list(d.bbox), "label": d.label, "confidence": d.confidence}
                for d in dets
            ]
        with open(self._cache_path, "w") as f:
            json.dump(data, f)

    def _load_cache(self) -> None:
        if not self._cache_path or not os.path.exists(self._cache_path):
            return
        try:
            with open(self._cache_path) as f:
                data = json.load(f)
            for key, dets in data.items():
                self._cache[key] = [
                    Detection(
                        bbox=tuple(d["bbox"]), label=d["label"],
                        confidence=d["confidence"],
                    )
                    for d in dets
                ]
            print(f"  Loaded {len(self._cache)} cached detections from {self._cache_path}")
        except Exception as e:
            print(f"  Warning: failed to load cache: {e}")

    def clear_cache(self) -> None:
        self._cache.clear()
        self._pending.clear()


# ====================================================================
# Batched trajectory generation
# ====================================================================

def _collect_leaf_detections(node: TrajectoryNode) -> List[Detection]:
    if node.action == "detect" or not node.children:
        return list(node.direct_detections)
    leaves: List[Detection] = []
    for child in node.children.values():
        leaves.extend(_collect_leaf_detections(child))
    return leaves


def _plan_tree_splits(
    max_depth: int,
    n_trajectories: int,
    p_split_by_depth: Optional[Dict[int, float]] = None,
) -> List[List[Dict[int, bool]]]:
    """Pre-roll all random split decisions for N trajectories.

    Returns: list of N trajectory plans, each being a list of
    {quadrant_path_hash -> should_split} dicts per depth.
    This separates randomness from inference so we can batch.
    """
    if p_split_by_depth is None:
        p_split_by_depth = {0: 0.7, 1: 0.4}

    plans = []
    for _ in range(n_trajectories):
        plan = {}  # qpath -> should_split
        # Depth 0: always one node (the root)
        plan["0"] = random.random() < p_split_by_depth.get(0, 0.7)
        # Depth 1: 4 possible nodes (if root split)
        if plan["0"]:
            for qid in [1, 2, 3, 4]:
                qpath = f"0-{qid}"
                plan[qpath] = random.random() < p_split_by_depth.get(1, 0.4)
        # Depth 2+: 16 possible nodes (if depth-1 split), etc.
        for d in range(2, max_depth):
            parent_depth_paths = [k for k in plan if k.count("-") == d - 1 and plan[k]]
            for parent_path in parent_depth_paths:
                for qid in [1, 2, 3, 4]:
                    qpath = f"{parent_path}-{qid}"
                    plan[qpath] = random.random() < p_split_by_depth.get(d, 0.3)
        plans.append(plan)
    return plans


def generate_trajectories_batched(
    image_np: np.ndarray,
    gt_list: List[Detection],
    detector: BatchDetector,
    n_trajectories: int = 8,
    max_depth: int = 2,
    overlap_ratio: float = 0.1,
    iou_threshold: float = 0.5,
    image_name: str = "",
    p_split_by_depth: Optional[Dict[int, float]] = None,
) -> List[Trajectory]:
    """Generate N trajectories using batched VLM inference.

    Strategy:
      1. Pre-roll all split decisions for all trajectories.
      2. Walk the quadtree depth-by-depth.  At each depth, collect ALL
         unique crop regions needed across ALL trajectories.
      3. Batch-infer them in one VLM call.
      4. Assemble trajectory trees from cached results.

    This replaces the sequential per-node inference with:
      - Depth 0: 1 inference (the full image, shared by all trajectories)
      - Depth 1: up to 4 inferences (4 quadrants, batched)
      - Depth 2: up to 16 inferences (16 sub-quadrants, batched)
    Total: ~21 inferences max regardless of n_trajectories (with greedy
    decoding).  With sampling, up to 21 × n_trajectories but still batched.
    """
    if p_split_by_depth is None:
        p_split_by_depth = {0: 0.7, 1: 0.4}

    img_h, img_w = image_np.shape[:2]
    use_sampling = detector.temperature > 0

    # Pre-roll split decisions
    plans = _plan_tree_splits(max_depth, n_trajectories, p_split_by_depth)

    # Build region map: qpath -> (region, crop_np)
    # We compute this once and reuse across trajectories
    region_map: Dict[str, Tuple[Tuple[float, float, float, float], np.ndarray]] = {}
    full_region = (0.0, 0.0, float(img_w), float(img_h))
    region_map["0"] = (full_region, image_np)

    def _ensure_children(parent_path: str):
        """Compute and store quadrant regions/crops for a parent node."""
        if parent_path not in region_map:
            return
        parent_region, parent_crop = region_map[parent_path]
        quads = split_image_into_quadrants(parent_crop, overlap_ratio)
        prx1, pry1 = parent_region[0], parent_region[1]
        for qid, qinfo in quads.items():
            child_path = f"{parent_path}-{qid}"
            if child_path not in region_map:
                ox, oy = qinfo["offset"]
                cw, ch = qinfo["size"]
                child_region = (prx1 + ox, pry1 + oy, prx1 + ox + cw, pry1 + oy + ch)
                region_map[child_path] = (child_region, qinfo["image"])

    # --- Depth-by-depth batched inference ---
    for depth in range(max_depth + 1):
        # Collect all unique (qpath, traj_idx) pairs at this depth
        needed: Dict[str, List[int]] = defaultdict(list)  # cache_key -> [traj_indices]
        requests: Dict[str, Tuple[str, np.ndarray, int, int]] = {}

        for ti, plan in enumerate(plans):
            # Which qpaths does trajectory ti need at this depth?
            for qpath, should_split in plan.items():
                if qpath.count("-") != depth and not (depth == 0 and qpath == "0"):
                    continue
                if qpath.count("-") != depth:
                    continue

                # Check that parent was split (otherwise this node doesn't exist)
                if depth > 0:
                    parent_path = qpath.rsplit("-", 1)[0]
                    if not plan.get(parent_path, False):
                        continue

                # Ensure region exists
                if depth > 0:
                    _ensure_children(parent_path)

                if qpath not in region_map:
                    continue

                region, crop_np = region_map[qpath]
                cache_key = detector.region_key(region, ti, use_sampling)

                if detector.get_cached(cache_key) is None:
                    h_crop, w_crop = crop_np.shape[:2]
                    detector.request(cache_key, crop_np, w_crop, h_crop)

        # Flush all pending requests in one batch
        detector.flush()

    # --- Assemble trajectory trees from cache ---
    trajectories = []
    for ti, plan in enumerate(plans):

        def _build_node(qpath: str, depth: int) -> Optional[TrajectoryNode]:
            if qpath not in region_map:
                return None
            region, crop_np = region_map[qpath]
            qid = int(qpath.split("-")[-1]) if "-" in qpath else 0

            # Get cached detections
            cache_key = detector.region_key(region, ti, use_sampling)
            raw_dets = detector.get_cached(cache_key)
            if raw_dets is None:
                raw_dets = []

            # Map to global coords
            rx1, ry1, rx2, ry2 = region
            rw, rh = int(rx2 - rx1), int(ry2 - ry1)
            global_dets = map_to_global(raw_dets, offset=(int(rx1), int(ry1)),
                                        sub_size=(rw, rh))
            clipped = []
            for det in global_dets:
                cb = clip_bbox(det.bbox, img_w, img_h)
                if cb[2] > cb[0] and cb[3] > cb[1]:
                    clipped.append(Detection(bbox=cb, label=det.label,
                                             confidence=det.confidence,
                                             depth=depth, quadrant_id=qid))

            node = TrajectoryNode(
                depth=depth, quadrant_id=qid, quadrant_path=qpath,
                region=region, action="detect", direct_detections=clipped,
            )

            should_split = plan.get(qpath, False) and depth < max_depth
            if should_split:
                node.action = "split"
                for child_qid in [1, 2, 3, 4]:
                    child_path = f"{qpath}-{child_qid}"
                    child = _build_node(child_path, depth + 1)
                    if child:
                        node.children[child_qid] = child

            return node

        root = _build_node("0", 0)
        if root is None:
            continue

        all_leaves = _collect_leaf_detections(root)
        merged = merge_detections(all_leaves, iou_threshold=iou_threshold, prefer_deeper=True)
        ap_result = compute_ap(merged, gt_list, iou_threshold=iou_threshold)

        trajectories.append(Trajectory(
            image_name=image_name, root=root, merged_detections=merged,
            ap_score=ap_result["ap"], recall=ap_result["recall"],
        ))

    return trajectories


# ====================================================================
# Rejection Sampling
# ====================================================================

def reject_sample(
    trajectories: List[Trajectory],
    baseline_ap: float,
    top_k: int = 3,
    min_improvement: float = 0.05,
    min_absolute_ap: float = 0.3,
) -> List[Trajectory]:
    candidates = [
        t for t in trajectories
        if t.ap_score > baseline_ap + min_improvement
        and t.ap_score > min_absolute_ap
    ]
    if not candidates:
        above_floor = [t for t in trajectories if t.ap_score > min_absolute_ap]
        if above_floor:
            return [max(above_floor, key=lambda t: t.ap_score)]
        return []
    candidates.sort(key=lambda t: t.ap_score, reverse=True)
    return candidates[:top_k]


# ====================================================================
# Split-Decision Labelling
# ====================================================================

def label_split_decisions(
    node: TrajectoryNode,
    gt_list: List[Detection],
    iou_threshold: float = 0.5,
    min_ap_gain: float = 0.03,
) -> None:
    region_gt = gt_boxes_in_region(gt_list, node.region)
    if not node.children:
        node.action = "detect"
        return
    for child in node.children.values():
        label_split_decisions(child, gt_list, iou_threshold, min_ap_gain)
    ap_detect = compute_ap(node.direct_detections, region_gt, iou_threshold)["ap"]
    child_dets = []
    for child in node.children.values():
        child_dets.extend(_collect_leaf_detections(child))
    child_merged = merge_detections(child_dets, iou_threshold=iou_threshold, prefer_deeper=True)
    ap_split = compute_ap(child_merged, region_gt, iou_threshold)["ap"]
    if ap_split > ap_detect + min_ap_gain:
        node.action = "split"
    else:
        node.action = "detect"
        node.children = {}


# ====================================================================
# Convert to SFT Examples
# ====================================================================

def trajectory_to_sft(
    traj: Trajectory, img_w: float, img_h: float,
) -> List[SFTExample]:
    examples: List[SFTExample] = []

    def _walk(node: TrajectoryNode):
        rx1, ry1, rx2, ry2 = node.region
        crop_w, crop_h = rx2 - rx1, ry2 - ry1

        if node.depth == 0:
            prompt = (f"This is the full {int(img_w)}x{int(img_h)} image. "
                      f"Detect all objects in this image.")
        else:
            prompt = (f"This is a {int(crop_w)}x{int(crop_h)} region "
                      f"from a {int(img_w)}x{int(img_h)} image "
                      f"(depth {node.depth}, quadrant {node.quadrant_id}). "
                      f"Detect all objects in this image.")

        if node.action == "split":
            target = "<SPLIT>"
        else:
            local_dets = map_to_local(
                node.direct_detections,
                offset=(int(rx1), int(ry1)),
                sub_size=(int(crop_w), int(crop_h)),
                model_input_size=(1000, 1000),
            )
            target = json.dumps([{
                "label": d.label,
                "bbox": [max(0, min(1000, int(round(d.bbox[0])))),
                         max(0, min(1000, int(round(d.bbox[1])))),
                         max(0, min(1000, int(round(d.bbox[2])))),
                         max(0, min(1000, int(round(d.bbox[3]))))],
            } for d in local_dets])

        examples.append(SFTExample(
            image_name=traj.image_name, crop_region=[rx1, ry1, rx2, ry2],
            depth=node.depth, quadrant_path=node.quadrant_path,
            prompt=prompt, target=target, image_w=img_w, image_h=img_h,
        ))
        if node.action == "split":
            for child in node.children.values():
                _walk(child)

    _walk(traj.root)
    return examples


# ====================================================================
# Data Loading
# ====================================================================

def load_low_ap_data(json_path: str) -> List[Dict]:
    with open(json_path) as f:
        raw = json.load(f)
    result = []
    for entry in raw:
        gt_dets = [Detection(bbox=(b["x1"], b["y1"], b["x2"], b["y2"]),
                             label=b.get("label", "object"), confidence=1.0)
                   for b in entry["ground_truth"]]
        pred_dets = [Detection(bbox=(b["x1"], b["y1"], b["x2"], b["y2"]),
                               label=b.get("label", "object"),
                               confidence=b.get("confidence", 0.5))
                     for b in entry["predictions"]]
        all_x = [d.bbox[2] for d in gt_dets] + [d.bbox[2] for d in pred_dets]
        all_y = [d.bbox[3] for d in gt_dets] + [d.bbox[3] for d in pred_dets]
        baseline = compute_ap(pred_dets, gt_dets)
        result.append({
            "image_name": entry["image"],
            "gt_detections": gt_dets, "pred_detections": pred_dets,
            "baseline_ap": baseline["ap"], "baseline_recall": baseline["recall"],
            "image_w": max(all_x) if all_x else 1000.0,
            "image_h": max(all_y) if all_y else 1000.0,
        })
    return result


# ====================================================================
# Checkpoint helpers
# ====================================================================

def _load_checkpoint(output_path: str) -> Tuple[List[Dict], set]:
    """Load existing SFT examples and return (examples_list, done_image_names)."""
    if not os.path.exists(output_path):
        return [], set()
    try:
        with open(output_path) as f:
            existing = json.load(f)
        done = set(ex["image_name"] for ex in existing)
        print(f"  Resuming: {len(done)} images already done, "
              f"{len(existing)} existing examples")
        return existing, done
    except Exception as e:
        print(f"  Warning: failed to load checkpoint: {e}")
        return [], set()


def _save_checkpoint(output_path: str, all_output: List[Dict]) -> None:
    """Atomically save progress (write to temp then rename)."""
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(all_output, f, indent=2)
    os.replace(tmp_path, output_path)


# ====================================================================
# Full Pipeline  (single-GPU worker)
# ====================================================================

def run_rejection_sampling(
    low_ap_json: str,
    image_dir: str,
    detector: "BatchDetector",
    n_trajectories: int = 8,
    max_depth: int = 2,
    overlap_ratio: float = 0.1,
    iou_threshold: float = 0.5,
    top_k: int = 3,
    min_improvement: float = 0.05,
    min_absolute_ap: float = 0.3,
    min_ap_gain_for_split: float = 0.03,
    output_path: str = "sft_rft_training_data.json",
    image_subset: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[List[Dict], Dict]:
    images = load_low_ap_data(low_ap_json)
    if verbose:
        print(f"Loaded {len(images)} low-AP images from {low_ap_json}")

    # --- Resume from checkpoint ---
    all_output, done_images = _load_checkpoint(output_path)

    # --- Filter to subset if specified (multi-GPU partitioning) ---
    if image_subset is not None:
        subset_set = set(image_subset)
        images = [img for img in images if img["image_name"] in subset_set]
        if verbose:
            print(f"  Filtered to {len(images)} images for this worker")

    stats = {"total_images": len(images), "images_with_accepted": 0,
             "total_trajectories_generated": 0, "total_trajectories_accepted": 0,
             "total_sft_examples": len(all_output),
             "split_examples": 0, "detect_examples": 0}

    for idx, img_data in enumerate(images):
        name = img_data["image_name"]

        # Skip already completed
        if name in done_images:
            if verbose:
                print(f"  [{idx+1}/{len(images)}] {name}: already done, skipping")
            continue

        gt_dets = img_data["gt_detections"]
        baseline_ap = img_data["baseline_ap"]

        img_path = os.path.join(image_dir, name)
        if not os.path.exists(img_path):
            if verbose:
                print(f"  [{idx+1}/{len(images)}] {name}: IMAGE NOT FOUND, skipping")
            continue

        from utils.image_utils import load_image
        image_np = np.array(load_image(img_path, convert_mode="RGB"))
        real_h, real_w = image_np.shape[:2]

        if verbose:
            print(f"\n[{idx+1}/{len(images)}] {name}  "
                  f"({real_w}x{real_h}, {len(gt_dets)} GT, baseline AP={baseline_ap:.3f})")

        # --- Clear per-image cache (keep disk cache for resume) ---
        detector.clear_cache()
        detector._cache_path = output_path.replace(".json", f"_cache_{name}.json")
        detector._load_cache(  )

        # --- Generate trajectories (BATCHED) ---
        t0 = time.time()
        trajs = generate_trajectories_batched(
            image_np=image_np, gt_list=gt_dets, detector=detector,
            n_trajectories=n_trajectories, max_depth=max_depth,
            overlap_ratio=overlap_ratio, iou_threshold=iou_threshold,
            image_name=name,
        )
        stats["total_trajectories_generated"] += len(trajs)
        elapsed = time.time() - t0

        aps = [t.ap_score for t in trajs]
        if verbose:
            print(f"  Generated {len(trajs)} trajectories in {elapsed:.1f}s  "
                  f"AP range: [{min(aps):.3f}, {max(aps):.3f}]")

        # --- Reject sample ---
        accepted = reject_sample(trajs, baseline_ap=baseline_ap,
                                 top_k=top_k, min_improvement=min_improvement,
                                 min_absolute_ap=min_absolute_ap)
        stats["total_trajectories_accepted"] += len(accepted)

        if not accepted:
            if verbose:
                print(f"  No trajectories accepted (none beat baseline)")
            # Mark as done even if nothing accepted (so we don't retry)
            done_images.add(name)
            _save_checkpoint(output_path, all_output)
            continue

        stats["images_with_accepted"] += 1
        if verbose:
            print(f"  Accepted {len(accepted)} — "
                  f"AP: {[f'{t.ap_score:.3f}' for t in accepted]}")

        # --- Label + convert ---
        for traj in accepted:
            label_split_decisions(traj.root, gt_dets, iou_threshold, min_ap_gain_for_split)
            examples = trajectory_to_sft(traj, float(real_w), float(real_h))
            for ex in examples:
                out = {"image_name": ex.image_name, "crop_region": ex.crop_region,
                       "depth": ex.depth, "quadrant_path": ex.quadrant_path,
                       "prompt": ex.prompt, "target": ex.target,
                       "image_w": ex.image_w, "image_h": ex.image_h}
                all_output.append(out)
                if ex.target == "<SPLIT>":
                    stats["split_examples"] += 1
                else:
                    stats["detect_examples"] += 1

        # --- Save checkpoint after each image ---
        done_images.add(name)
        _save_checkpoint(output_path, all_output)
        if verbose:
            print(f"  Checkpoint saved ({len(all_output)} total examples)")

    stats["total_sft_examples"] = len(all_output)
    if verbose:
        print("\n" + "=" * 60)
        print("REJECTION SAMPLING SUMMARY")
        print("=" * 60)
        for k, v in stats.items():
            print(f"  {k}: {v}")
        if stats["total_sft_examples"] > 0:
            pct = stats["split_examples"] / stats["total_sft_examples"] * 100
            print(f"  split/detect ratio: {pct:.1f}% split, {100-pct:.1f}% detect")
        print(f"\nSaved {len(all_output)} examples -> {output_path}")

    return all_output, stats


# ====================================================================
# Multi-GPU worker
# ====================================================================

def _gpu_worker(
    gpu_id: int,
    image_names: List[str],
    cfg: Dict,
    output_path: str,
) -> None:
    """Worker function for one GPU. Loads its own model copy."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from model.loader import load_local_qwen3vl_model
    from utils.parse_utils import parse_bounding_boxes

    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading model on {device} for {len(image_names)} images...")

    lora_path = cfg.get("lora_path")
    is_lora = cfg.get("is_lora", False)
    if lora_path and str(lora_path).lower() in ("none", "null", ""):
        lora_path = None
        is_lora = False

    hf_token = cfg.get("hf_token")
    if isinstance(hf_token, str) and hf_token.lower() in ("none", "null", ""):
        hf_token = None

    model, processor, dev = load_local_qwen3vl_model(
        model_name=cfg["model_name"],
        lora_path=lora_path, is_lora=is_lora, device=device,
        model_scope=cfg.get("use_modelscope", False),
        enable_hf_mirror=cfg.get("enable_hf_mirror", False),
        hf_token=hf_token,
    )

    detector = BatchDetector(
        model=model, processor=processor, device=dev,
        prompt=cfg.get("prompt", "Detect all objects in this image."),
        parse_fn=parse_bounding_boxes,
        max_new_tokens=cfg.get("max_new_tokens", 8192),
        temperature=cfg.get("temperature", 0.7),
        top_p=cfg.get("top_p", 0.9),
        batch_size=cfg.get("batch_size", 4),
    )

    run_rejection_sampling(
        low_ap_json=cfg["low_ap_json"],
        image_dir=cfg["images_dir"],
        detector=detector,
        n_trajectories=cfg.get("n_trajectories", 8),
        max_depth=cfg.get("max_depth", 2),
        overlap_ratio=cfg.get("overlap_ratio", 0.1),
        iou_threshold=cfg.get("iou_threshold", 0.5),
        top_k=cfg.get("top_k", 3),
        min_improvement=cfg.get("min_improvement", 0.05),
        min_absolute_ap=cfg.get("min_absolute_ap", 0.3),
        min_ap_gain_for_split=cfg.get("min_ap_gain_for_split", 0.03),
        output_path=output_path,
        image_subset=image_names,
    )
    print(f"[GPU {gpu_id}] Done.")


# ====================================================================
# Config Loading + CLI
# ====================================================================

def load_config(config_path: str) -> Dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Rejection Sampling Fine-tuning for recursive VLM detection",
    )
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    output_dir = cfg.get("output_dir", "./rft_results")
    os.makedirs(output_dir, exist_ok=True)

    # --- Dispatch: single image test ---
    test_image = cfg.get("test_image")
    if test_image:
        from model.loader import load_local_qwen3vl_model
        from utils.parse_utils import parse_bounding_boxes

        lora_path = cfg.get("lora_path")
        is_lora = cfg.get("is_lora", False)
        if lora_path and str(lora_path).lower() in ("none", "null", ""):
            lora_path = None; is_lora = False
        hf_token = cfg.get("hf_token")
        if isinstance(hf_token, str) and hf_token.lower() in ("none", "null", ""):
            hf_token = None

        model, processor, device = load_local_qwen3vl_model(
            model_name=cfg["model_name"], lora_path=lora_path, is_lora=is_lora,
            device=cfg.get("device", "auto"),
            model_scope=cfg.get("use_modelscope", False),
            enable_hf_mirror=cfg.get("enable_hf_mirror", False),
            hf_token=hf_token,
        )

        output_path = os.path.join(output_dir, "test_single_image_sft.json")
        detector = BatchDetector(
            model=model, processor=processor, device=device,
            prompt=cfg.get("prompt", "Detect all objects in this image."),
            parse_fn=parse_bounding_boxes,
            max_new_tokens=cfg.get("max_new_tokens", 8192),
            temperature=cfg.get("temperature", 0.7),
            top_p=cfg.get("top_p", 0.9),
            batch_size=cfg.get("batch_size", 4),
            cache_path=os.path.join(output_dir, "detection_cache.json"),
        )

        test_single_image(
            image_path=test_image,
            low_ap_json=cfg["low_ap_json"],
            detector=detector,
            n_trajectories=cfg.get("n_trajectories", 8),
            max_depth=cfg.get("max_depth", 2),
            overlap_ratio=cfg.get("overlap_ratio", 0.1),
            iou_threshold=cfg.get("iou_threshold", 0.5),
            top_k=cfg.get("top_k", 3),
            min_improvement=cfg.get("min_improvement", 0.05),
            min_ap_gain_for_split=cfg.get("min_ap_gain_for_split", 0.03),
            output_path=output_path,
        )
        return

    # --- Full pipeline: decide single-GPU vs multi-GPU ---
    import torch
    num_gpus = cfg.get("num_gpus", 1)
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if num_gpus == -1:
        num_gpus = max(1, available_gpus)
    num_gpus = min(num_gpus, max(1, available_gpus))

    # Load image list for partitioning
    images = load_low_ap_data(cfg["low_ap_json"])
    all_image_names = [img["image_name"] for img in images]

    if num_gpus <= 1:
        # Single GPU path
        from model.loader import load_local_qwen3vl_model
        from utils.parse_utils import parse_bounding_boxes

        lora_path = cfg.get("lora_path")
        is_lora = cfg.get("is_lora", False)
        if lora_path and str(lora_path).lower() in ("none", "null", ""):
            lora_path = None; is_lora = False
        hf_token = cfg.get("hf_token")
        if isinstance(hf_token, str) and hf_token.lower() in ("none", "null", ""):
            hf_token = None

        model, processor, device = load_local_qwen3vl_model(
            model_name=cfg["model_name"], lora_path=lora_path, is_lora=is_lora,
            device=cfg.get("device", "auto"),
            model_scope=cfg.get("use_modelscope", False),
            enable_hf_mirror=cfg.get("enable_hf_mirror", False),
            hf_token=hf_token,
        )

        output_path = os.path.join(output_dir, "sft_rft_training_data.json")
        detector = BatchDetector(
            model=model, processor=processor, device=device,
            prompt=cfg.get("prompt", "Detect all objects in this image."),
            parse_fn=parse_bounding_boxes,
            max_new_tokens=cfg.get("max_new_tokens", 8192),
            temperature=cfg.get("temperature", 0.7),
            top_p=cfg.get("top_p", 0.9),
            batch_size=cfg.get("batch_size", 4),
        )

        run_rejection_sampling(
            low_ap_json=cfg["low_ap_json"],
            image_dir=cfg["images_dir"],
            detector=detector,
            n_trajectories=cfg.get("n_trajectories", 8),
            max_depth=cfg.get("max_depth", 2),
            overlap_ratio=cfg.get("overlap_ratio", 0.1),
            iou_threshold=cfg.get("iou_threshold", 0.5),
            top_k=cfg.get("top_k", 3),
            min_improvement=cfg.get("min_improvement", 0.05),
            min_absolute_ap=cfg.get("min_absolute_ap", 0.3),
            min_ap_gain_for_split=cfg.get("min_ap_gain_for_split", 0.03),
            output_path=output_path,
        )
    else:
        # Multi-GPU: partition images and spawn workers
        print(f"Launching {num_gpus} GPU workers for {len(all_image_names)} images")
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)

        # Round-robin partition
        partitions = [[] for _ in range(num_gpus)]
        for i, name in enumerate(all_image_names):
            partitions[i % num_gpus].append(name)

        processes = []
        for gpu_id in range(num_gpus):
            per_gpu_output = os.path.join(output_dir, f"sft_rft_gpu{gpu_id}.json")
            p = mp.Process(
                target=_gpu_worker,
                args=(gpu_id, partitions[gpu_id], cfg, per_gpu_output),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Merge per-GPU outputs
        merged_output = []
        for gpu_id in range(num_gpus):
            per_gpu_path = os.path.join(output_dir, f"sft_rft_gpu{gpu_id}.json")
            if os.path.exists(per_gpu_path):
                with open(per_gpu_path) as f:
                    merged_output.extend(json.load(f))

        final_path = os.path.join(output_dir, "sft_rft_training_data.json")
        with open(final_path, "w") as f:
            json.dump(merged_output, f, indent=2)
        print(f"\nMerged {len(merged_output)} examples from {num_gpus} GPUs -> {final_path}")


# ====================================================================
# Single-image test
# ====================================================================

def test_single_image(
    image_path: str,
    low_ap_json: str,
    detector: "BatchDetector",
    n_trajectories: int = 4,
    max_depth: int = 2,
    overlap_ratio: float = 0.1,
    iou_threshold: float = 0.5,
    top_k: int = 3,
    min_improvement: float = 0.05,
    min_ap_gain_for_split: float = 0.03,
    output_path: str = "test_single_image_sft.json",
):
    from utils.image_utils import load_image

    image_name = os.path.basename(image_path)
    image_np = np.array(load_image(image_path, convert_mode="RGB"))
    img_h, img_w = image_np.shape[:2]
    print(f"Image: {image_name}  ({img_w}x{img_h})")

    with open(low_ap_json) as f:
        all_entries = json.load(f)
    entry = next((e for e in all_entries if e["image"] == image_name), None)
    if entry is None:
        print(f"ERROR: '{image_name}' not found in {low_ap_json}")
        return

    gt_dets = [Detection(bbox=(b["x1"], b["y1"], b["x2"], b["y2"]),
                         label=b.get("label", "object"), confidence=1.0)
               for b in entry["ground_truth"]]
    pred_dets = [Detection(bbox=(b["x1"], b["y1"], b["x2"], b["y2"]),
                           label=b.get("label", "object"),
                           confidence=b.get("confidence", 0.5))
                 for b in entry["predictions"]]

    baseline = compute_ap(pred_dets, gt_dets, iou_threshold)
    print(f"Ground truth: {len(gt_dets)} boxes")
    print(f"Baseline: AP={baseline['ap']:.4f}  P={baseline['precision']:.4f}  "
          f"R={baseline['recall']:.4f}")

    print(f"\nGenerating {n_trajectories} trajectories (max_depth={max_depth}) ...")
    t0 = time.time()
    trajs = generate_trajectories_batched(
        image_np=image_np, gt_list=gt_dets, detector=detector,
        n_trajectories=n_trajectories, max_depth=max_depth,
        overlap_ratio=overlap_ratio, iou_threshold=iou_threshold,
        image_name=image_name,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s\n")

    print(f"{'Traj':>5}  {'AP':>7}  {'Recall':>7}  {'Merged':>7}  Root Action")
    print("-" * 50)
    for i, t in enumerate(trajs):
        print(f"  {i:3d}  {t.ap_score:7.4f}  {t.recall:7.4f}  "
              f"{len(t.merged_detections):7d}  {t.root.action}")

    accepted = reject_sample(trajs, baseline_ap=baseline["ap"],
                             top_k=top_k, min_improvement=min_improvement)
    print(f"\nAccepted {len(accepted)} / {len(trajs)} trajectories")
    if not accepted:
        print("No trajectories beat baseline.")
        return

    all_examples = []
    for traj in accepted:
        label_split_decisions(traj.root, gt_dets, iou_threshold, min_ap_gain_for_split)
        all_examples.extend(trajectory_to_sft(traj, float(img_w), float(img_h)))

    n_split = sum(1 for ex in all_examples if ex.target == "<SPLIT>")
    print(f"\n{len(all_examples)} SFT examples ({n_split} SPLIT, {len(all_examples)-n_split} detect)")
    for ex in all_examples:
        tgt = "<SPLIT>" if ex.target == "<SPLIT>" else f"[{len(json.loads(ex.target))} dets]"
        print(f"  d{ex.depth} {ex.quadrant_path:<10s} {[int(v) for v in ex.crop_region]} -> {tgt}")

    output = [{"image_name": ex.image_name, "crop_region": ex.crop_region,
               "depth": ex.depth, "quadrant_path": ex.quadrant_path,
               "prompt": ex.prompt, "target": ex.target,
               "image_w": ex.image_w, "image_h": ex.image_h}
              for ex in all_examples]
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()