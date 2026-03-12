"""
Evaluation script for 2D grounding with recursive split-detect-merge support.

Extends the original eval_2d_grounding_multigpu.py with:
  - ``--use_split``: enable model-driven recursive splitting via <SPLIT> token
  - Uses ``inference_with_split`` from model/inference.py
  - Reuses pipeline/splitter, pipeline/mapper, pipeline/merger for NMS

Usage:
    # Standard evaluation (no splitting):
    python eval_2d_grounding.py config.yaml

    # With recursive split-detect-merge:
    python eval_2d_grounding.py config.yaml --use_split

    # Multi-GPU:
    python eval_2d_grounding.py config.yaml --use_split --num_gpus 4
"""

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.inference import inference_local_qwen3vl, inference_with_split
from model.loader import load_local_qwen3vl_model
from utils.box_utils import validate_and_clip_boxes
from utils.image_utils import load_image
from utils.parse_utils import parse_bounding_boxes
from eval.metrics import compute_map_box_only, compute_coco_map_pycocotools


# ====================================================================
# Data loading helpers (unchanged)
# ====================================================================

def load_ground_truth_from_csv(csv_path: str) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Tuple[int, int]]]:
    ground_truth, image_sizes = {}, {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 7:
                continue
            image_name, x1, y1, x2, y2, label = row[0], *map(int, row[1:5]), row[5]
            img_width, img_height = (int(row[6]), int(row[7])) if len(row) > 7 else (None, None)
            if x2 <= x1 or y2 <= y1:
                continue
            ground_truth.setdefault(image_name, []).append(
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": label}
            )
            if img_width and img_height:
                image_sizes[image_name] = (img_width, img_height)
    return ground_truth, image_sizes


def get_image_size(image_path: str) -> Tuple[int, int]:
    try:
        return load_image(image_path).size
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        return (0, 0)


# ====================================================================
# Checkpoint helpers
# ====================================================================

def save_checkpoint(output_dir: Path, image_names, responses, all_predictions, all_ground_truths,
                    split_trees=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "responses.json", "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
    detailed = []
    for i, (name, pred, gt) in enumerate(zip(image_names, all_predictions, all_ground_truths)):
        entry = {
            "image": name, "num_predictions": len(pred), "num_ground_truth": len(gt),
            "predictions": pred, "ground_truth": gt,
        }
        if split_trees and name in split_trees:
            entry["split_tree"] = split_trees[name]
        detailed.append(entry)
    with open(output_dir / "detailed_results.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)
    print(f"  [Checkpoint] Saved to: {output_dir}")


# ====================================================================
# Standard evaluation (no split) — unchanged logic
# ====================================================================

def evaluate_standard(
    model, processor, device, image_paths, ground_truth,
    image_sizes=None, prompt=None, iou_threshold=0.5, box_only=True,
    batch_size=1, max_new_tokens=8192, do_sample=False, repetition_penalty=1.1,
    verbose=True, gpu_id=0, image_max_pixels=15873000, image_min_pixels=3136,
    output_dir=None,
) -> Dict[str, Any]:
    prompt = prompt or (
        "This is a retail shelf image. Detect and locate every individual product "
        "on the shelf, treating each separate item as a distinct object. "
        "Report bounding box coordinates for all items in JSON format."
    )
    image_names = [Path(p).name for p in image_paths]
    all_ground_truths = [ground_truth.get(name, []).copy() for name in image_names]

    if verbose:
        print(f"[GPU {gpu_id}] Running standard inference on {len(image_paths)} images...")
    t0 = time.time()
    responses = inference_local_qwen3vl(
        model=model, processor=processor, device=device,
        img_urls=image_paths, prompts=[prompt] * len(image_paths),
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        do_sample=do_sample, repetition_penalty=repetition_penalty,
        verbose=verbose, max_pixels=image_max_pixels, min_pixels=image_min_pixels,
    )
    t_inference = time.time() - t0
    if verbose:
        print(f"[GPU {gpu_id}] Inference done in {t_inference:.1f}s "
              f"({t_inference/len(image_paths):.2f}s/img)")

    # Parse boxes
    all_predictions = []
    for img_path, response, img_name in zip(image_paths, responses, image_names):
        w, h = image_sizes[img_name] if (image_sizes and img_name in image_sizes) else get_image_size(img_path)
        if w == 0 or h == 0:
            all_predictions.append([])
            continue
        boxes = validate_and_clip_boxes(parse_bounding_boxes(response, w, h, iou_threshold=0.95), w, h)
        all_predictions.append(boxes)

    responses_dict = dict(zip(image_names, responses))

    if output_dir:
        ckpt_dir = Path(output_dir) / f"checkpoint_gpu{gpu_id}" if gpu_id > 0 else Path(output_dir)
        save_checkpoint(ckpt_dir, image_names, responses_dict, all_predictions, all_ground_truths)

    metrics = compute_map_box_only(all_predictions, all_ground_truths, iou_threshold) if box_only else \
        __import__("eval.metrics", fromlist=["compute_map"]).compute_map(all_predictions, all_ground_truths, iou_threshold, True, False)
    metrics.update({"predictions": all_predictions, "ground_truths": all_ground_truths,
                    "image_names": image_names, "responses": responses_dict})
    return metrics


# ====================================================================
# Split-detect-merge evaluation
# ====================================================================

def evaluate_with_split(
    model, processor, device, image_paths, ground_truth,
    image_sizes=None, prompt=None, iou_threshold=0.5, box_only=True,
    max_new_tokens=8192, do_sample=False, repetition_penalty=1.1,
    split_token="<SPLIT>", max_depth=2, overlap_ratio=0.1,
    batch_size=4, prefer_deeper=True,
    verbose=True, gpu_id=0, image_max_pixels=15873000, image_min_pixels=3136,
    output_dir=None,
) -> Dict[str, Any]:
    """Run recursive split-detect-merge evaluation.

    For each image, calls ``inference_with_split`` which:
      1. Runs the VLM on the full image
      2. If the model outputs <SPLIT>, splits into 4 quadrants and recurses
      3. Maps all leaf detections to global coords
      4. Merges with depth-aware NMS
    """
    prompt = prompt or (
        "This is a retail shelf image. Detect and locate every individual product "
        "on the shelf, treating each separate item as a distinct object. "
        "Report bounding box coordinates for all items in JSON format."
    )
    image_names = [Path(p).name for p in image_paths]
    all_ground_truths = [ground_truth.get(name, []).copy() for name in image_names]

    if verbose:
        print(f"[GPU {gpu_id}] Running split-detect-merge inference on {len(image_paths)} images...")
        print(f"  split_token={split_token}, max_depth={max_depth}, overlap={overlap_ratio}")

    all_predictions = []
    all_split_trees = {}
    responses_dict = {}
    total_vlm_calls = 0
    t0 = time.time()

    for i, (img_path, img_name) in enumerate(zip(image_paths, image_names)):
        t_img = time.time()

        result = inference_with_split(
            model=model, processor=processor, device=device,
            img_url=img_path, prompt=prompt, parse_fn=parse_bounding_boxes,
            split_token=split_token, max_depth=max_depth,
            overlap_ratio=overlap_ratio, iou_threshold=iou_threshold,
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=1.0, top_p=1.0, repetition_penalty=repetition_penalty,
            batch_size=batch_size, prefer_deeper=prefer_deeper,
            verbose=False,
            min_pixels=image_min_pixels, max_pixels=image_max_pixels,
        )

        # Validate and clip the merged detections
        w, h = image_sizes[img_name] if (image_sizes and img_name in image_sizes) else get_image_size(img_path)
        boxes = validate_and_clip_boxes(result["detections"], w, h) if w > 0 and h > 0 else []

        all_predictions.append(boxes)
        all_split_trees[img_name] = result["split_tree"]
        responses_dict[img_name] = result["raw_response"]
        total_vlm_calls += result["num_vlm_calls"]

        elapsed = time.time() - t_img
        if verbose and (i + 1) % 5 == 0 or i == 0:
            avg = (time.time() - t0) / (i + 1)
            eta = avg * (len(image_paths) - i - 1)
            action = result["split_tree"].get("action", "?")
            print(f"  [{i+1}/{len(image_paths)}] {img_name}: {len(boxes)} dets, "
                  f"{result['num_vlm_calls']} calls, root={action}, "
                  f"{elapsed:.1f}s (ETA {eta/60:.1f}min)")

    t_total = time.time() - t0
    if verbose:
        print(f"[GPU {gpu_id}] Split inference done in {t_total:.1f}s "
              f"({t_total/len(image_paths):.1f}s/img, {total_vlm_calls} total VLM calls)")

    # Checkpoint
    if output_dir:
        ckpt_dir = Path(output_dir) / f"checkpoint_gpu{gpu_id}" if gpu_id > 0 else Path(output_dir)
        save_checkpoint(ckpt_dir, image_names, responses_dict,
                        all_predictions, all_ground_truths, all_split_trees)

    # Compute metrics
    metrics = compute_map_box_only(all_predictions, all_ground_truths, iou_threshold) if box_only else \
        __import__("eval.metrics", fromlist=["compute_map"]).compute_map(all_predictions, all_ground_truths, iou_threshold, True, False)
    metrics.update({
        "predictions": all_predictions, "ground_truths": all_ground_truths,
        "image_names": image_names, "responses": responses_dict,
        "split_trees": all_split_trees, "total_vlm_calls": total_vlm_calls,
    })
    return metrics


# ====================================================================
# Multi-GPU worker
# ====================================================================

def evaluate_on_single_gpu(gpu_id, image_paths, ground_truth, image_sizes, args_dict, return_dict):
    try:
        device = f"cuda:{gpu_id}"
        print(f"[GPU {gpu_id}] Starting on {len(image_paths)} images...")
        t_load = time.time()
        model, processor, _ = load_local_qwen3vl_model(
            model_name=args_dict["model_name"], lora_path=args_dict.get("lora_path"),
            is_lora=args_dict.get("is_lora", False), device=device,
            model_scope=args_dict.get("use_modelscope", False),
            enable_hf_mirror=args_dict.get("enable_hf_mirror", False),
            hf_token=args_dict.get("hf_token"),
        )
        print(f"[GPU {gpu_id}] Model loaded in {time.time()-t_load:.1f}s")

        use_split = args_dict.get("use_split", False)

        if use_split:
            metrics = evaluate_with_split(
                model=model, processor=processor, device=device,
                image_paths=image_paths, ground_truth=ground_truth,
                image_sizes=image_sizes,
                prompt=args_dict.get("prompt"),
                iou_threshold=args_dict.get("iou_threshold", 0.5),
                box_only=args_dict.get("box_only", True),
                max_new_tokens=args_dict.get("max_new_tokens", 8192),
                do_sample=args_dict.get("do_sample", False),
                repetition_penalty=args_dict.get("repetition_penalty", 1.1),
                split_token=args_dict.get("split_token", "<SPLIT>"),
                max_depth=args_dict.get("max_depth", 2),
                overlap_ratio=args_dict.get("overlap_ratio", 0.1),
                batch_size=args_dict.get("batch_size", 4),
                prefer_deeper=args_dict.get("prefer_deeper", True),
                verbose=(gpu_id == 0), gpu_id=gpu_id,
                image_max_pixels=args_dict.get("image_max_pixels", 15873000),
                image_min_pixels=args_dict.get("image_min_pixels", 3136),
                output_dir=args_dict.get("output_dir"),
            )
        else:
            metrics = evaluate_standard(
                model=model, processor=processor, device=device,
                image_paths=image_paths, ground_truth=ground_truth,
                image_sizes=image_sizes,
                prompt=args_dict.get("prompt"),
                iou_threshold=args_dict.get("iou_threshold", 0.5),
                box_only=args_dict.get("box_only", True),
                batch_size=args_dict.get("batch_size", 1),
                max_new_tokens=args_dict.get("max_new_tokens", 8192),
                do_sample=args_dict.get("do_sample", False),
                repetition_penalty=args_dict.get("repetition_penalty", 1.1),
                verbose=(gpu_id == 0), gpu_id=gpu_id,
                image_max_pixels=args_dict.get("image_max_pixels", 15873000),
                image_min_pixels=args_dict.get("image_min_pixels", 3136),
                output_dir=args_dict.get("output_dir"),
            )

        del model
        torch.cuda.empty_cache()
        return_dict[gpu_id] = metrics
        print(f"[GPU {gpu_id}] Done.")
    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        import traceback; traceback.print_exc()
        return_dict[gpu_id] = {"error": str(e)}


def merge_metrics(gpu_metrics, iou_threshold=0.5, box_only=True):
    all_preds, all_gts, all_names, all_resp = [], [], [], {}
    all_trees = {}
    total_vlm = 0
    for gpu_id in sorted(gpu_metrics.keys()):
        m = gpu_metrics[gpu_id]
        if "error" in m:
            print(f"Warning: GPU {gpu_id} had an error, skipping")
            continue
        all_preds.extend(m["predictions"]); all_gts.extend(m["ground_truths"])
        all_names.extend(m["image_names"]); all_resp.update(m["responses"])
        if "split_trees" in m:
            all_trees.update(m["split_trees"])
        total_vlm += m.get("total_vlm_calls", 0)

    metrics = compute_map_box_only(all_preds, all_gts, iou_threshold) if box_only else \
        __import__("eval.metrics", fromlist=["compute_map"]).compute_map(all_preds, all_gts, iou_threshold, True, False)
    metrics.update({"predictions": all_preds, "ground_truths": all_gts,
                    "image_names": all_names, "responses": all_resp,
                    "split_trees": all_trees, "total_vlm_calls": total_vlm})
    return metrics


def evaluate_multi_gpu(args, test_images, ground_truth, image_sizes, num_gpus=None):
    available = torch.cuda.device_count()
    num_gpus = min(num_gpus or available, available)
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")

    mode = "split-detect-merge" if args.use_split else "standard"
    print(f"\nMulti-GPU: {num_gpus} GPUs, {len(test_images)} images, mode={mode}")
    chunks = [[] for _ in range(num_gpus)]
    for i, p in enumerate(test_images):
        chunks[i % num_gpus].append(p)

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    for gpu_id in range(num_gpus):
        if not chunks[gpu_id]:
            continue
        p = mp.Process(target=evaluate_on_single_gpu,
                       args=(gpu_id, chunks[gpu_id], ground_truth, image_sizes, vars(args), return_dict))
        p.start(); processes.append(p)
    for p in processes:
        p.join()

    print("\nMerging results...")
    return merge_metrics(dict(return_dict), args.iou_threshold, args.box_only)


# ====================================================================
# Results saving
# ====================================================================

def save_results(output_dir: Path, metrics, args, coco_metrics=None, timing=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    if coco_metrics:
        m = {"num_images": metrics["num_images"],
             "map": coco_metrics["map"], "map_50": coco_metrics["map_50"], "map_75": coco_metrics["map_75"],
             "map_small": coco_metrics.get("map_small"), "map_medium": coco_metrics.get("map_medium"),
             "map_large": coco_metrics.get("map_large"),
             "precision": metrics["precision"], "recall": metrics["recall"], "mean_iou": metrics["mean_iou"]}
    else:
        m = {k: metrics[k] for k in ["num_images", "map", "precision", "recall", "mean_iou"]}
    m.update({
        "iou_threshold": args.iou_threshold, "box_only": args.box_only,
        "num_gpus": args.num_gpus, "use_split": args.use_split,
    })
    if args.use_split:
        m["split_config"] = {
            "max_depth": args.max_depth, "overlap_ratio": args.overlap_ratio,
            "split_token": args.split_token, "prefer_deeper": args.prefer_deeper,
            "total_vlm_calls": metrics.get("total_vlm_calls", 0),
        }
    if timing:
        m["timing"] = timing
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    print(f"Final results saved to: {output_dir / 'metrics.json'}")


# ====================================================================
# Config + CLI
# ====================================================================

def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(OmegaConf.load(Path(config_path).absolute()), resolve=True)
    except ImportError:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def parse_args(config: Dict) -> Any:
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate 2D grounding with optional split-detect-merge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    bool_type = lambda x: str(x).lower() in ["true", "1", "yes"]

    # Model
    parser.add_argument("--model_name", default=config.get("model_name", "Qwen/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--lora_path", default=config.get("lora_path"))
    parser.add_argument("--is_lora", type=bool_type, default=config.get("is_lora", False))

    # Data
    parser.add_argument("--images_dir", default=config.get("images_dir", "/root/Codes/data/SKU110K_fixed/images"))
    parser.add_argument("--annotations_csv", default=config.get("annotations_csv",
                        "/root/Codes/data/SKU110K_fixed/annotations/annotations_test.csv"))
    parser.add_argument("--image_prefix", default=config.get("image_prefix", "test_"))
    parser.add_argument("--max_images", type=int, default=config.get("max_images", 10))

    # Eval
    parser.add_argument("--iou_threshold", type=float, default=config.get("iou_threshold", 0.5))
    parser.add_argument("--box_only", type=bool_type, default=config.get("box_only", True))
    parser.add_argument("--prompt", default=config.get("prompt"))

    # Inference
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size", 1))
    parser.add_argument("--max_new_tokens", type=int, default=config.get("max_new_tokens", 8192))
    parser.add_argument("--do_sample", type=bool_type, default=config.get("do_sample", False))
    parser.add_argument("--repetition_penalty", type=float, default=config.get("repetition_penalty", 1.1))

    # Split-detect-merge
    parser.add_argument("--use_split", type=bool_type, default=config.get("use_split", False),
                        help="Enable recursive split-detect-merge via <SPLIT> token")
    parser.add_argument("--split_token", default=config.get("split_token", "<SPLIT>"))
    parser.add_argument("--max_depth", type=int, default=config.get("max_depth", 2))
    parser.add_argument("--overlap_ratio", type=float, default=config.get("overlap_ratio", 0.1))
    parser.add_argument("--prefer_deeper", type=bool_type, default=config.get("prefer_deeper", True))

    # Hardware
    parser.add_argument("--device", default=config.get("device", "auto"))
    parser.add_argument("--num_gpus", type=int, default=config.get("num_gpus", 1))

    # Output
    parser.add_argument("--output_dir", default=config.get("output_dir"))

    # Misc
    parser.add_argument("--use_modelscope", action="store_true", default=config.get("use_modelscope", False))
    parser.add_argument("--enable_hf_mirror", type=bool_type, default=config.get("enable_hf_mirror", False))
    parser.add_argument("--hf_token", default=config.get("hf_token"))
    parser.add_argument("--image_max_pixels", type=int, default=config.get("image_max_pixels", 15873000))
    parser.add_argument("--image_min_pixels", type=int, default=config.get("image_min_pixels", 3136))

    return parser.parse_args()


# ====================================================================
# Main
# ====================================================================

def main():
    config = {}
    if len(sys.argv) > 1 and sys.argv[1].endswith((".yaml", ".yml")):
        config = load_config_from_yaml(sys.argv[1])
        sys.argv = [sys.argv[0]] + sys.argv[2:]
    args = parse_args(config)

    # GPU setup
    num_available = torch.cuda.device_count()
    if num_available == 0:
        args.num_gpus, args.device = 0, "cpu"
    if args.num_gpus == -1:
        args.num_gpus = num_available
    args.num_gpus = min(args.num_gpus, num_available)

    # Load data
    ground_truth, image_sizes = load_ground_truth_from_csv(args.annotations_csv)
    images_dir = Path(args.images_dir)
    test_images = [str(p) for p in sorted(images_dir.glob(f"{args.image_prefix}*.jpg"))[:args.max_images]]
    if not test_images:
        print(f"Error: No test images found in {images_dir}"); exit(1)

    mode = "split-detect-merge" if args.use_split else "standard"
    print(f"Found {len(test_images)} test images, GT for {len(ground_truth)} images")
    print(f"Mode: {mode}")
    if args.use_split:
        print(f"  split_token={args.split_token}, max_depth={args.max_depth}, "
              f"overlap={args.overlap_ratio}, prefer_deeper={args.prefer_deeper}")

    t_total_start = time.time()

    # Run evaluation
    if args.num_gpus > 1:
        metrics = evaluate_multi_gpu(args, test_images, ground_truth, image_sizes, args.num_gpus)
    else:
        print(f"\nLoading model: {args.model_name}")
        t_load = time.time()
        model, processor, device = load_local_qwen3vl_model(
            model_name=args.model_name, lora_path=args.lora_path, is_lora=args.is_lora,
            device=args.device, model_scope=args.use_modelscope,
            enable_hf_mirror=args.enable_hf_mirror, hf_token=args.hf_token)
        print(f"Model loaded on {device} in {time.time()-t_load:.1f}s")

        if args.use_split:
            metrics = evaluate_with_split(
                model=model, processor=processor, device=device,
                image_paths=test_images, ground_truth=ground_truth,
                image_sizes=image_sizes, prompt=args.prompt,
                iou_threshold=args.iou_threshold, box_only=args.box_only,
                max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
                repetition_penalty=args.repetition_penalty,
                split_token=args.split_token, max_depth=args.max_depth,
                overlap_ratio=args.overlap_ratio, batch_size=args.batch_size,
                prefer_deeper=args.prefer_deeper,
                image_max_pixels=args.image_max_pixels,
                image_min_pixels=args.image_min_pixels,
                output_dir=args.output_dir,
            )
        else:
            metrics = evaluate_standard(
                model=model, processor=processor, device=device,
                image_paths=test_images, ground_truth=ground_truth,
                image_sizes=image_sizes, prompt=args.prompt,
                iou_threshold=args.iou_threshold, box_only=args.box_only,
                batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample, repetition_penalty=args.repetition_penalty,
                image_max_pixels=args.image_max_pixels,
                image_min_pixels=args.image_min_pixels,
                output_dir=args.output_dir,
            )

    # Checkpoint merged results
    if args.output_dir:
        save_checkpoint(
            Path(args.output_dir), metrics["image_names"], metrics["responses"],
            metrics["predictions"], metrics["ground_truths"],
            metrics.get("split_trees"),
        )

    # Basic results
    print(f"\n{'='*70}")
    print(f"Results ({mode}): {metrics['num_images']} images")
    print(f"mAP@{args.iou_threshold}: {metrics['map']:.4f} | P: {metrics['precision']:.4f} | "
          f"R: {metrics['recall']:.4f} | IoU: {metrics['mean_iou']:.4f}")
    if args.use_split:
        print(f"Total VLM calls: {metrics.get('total_vlm_calls', '?')}")

    # COCO mAP
    t_coco = time.time()
    coco = compute_coco_map_pycocotools(
        metrics["predictions"], metrics["ground_truths"],
        image_sizes=image_sizes, image_names=metrics["image_names"])
    print(f"\nCOCO mAP@[.5:.95]: {coco['map']:.4f} | @.5: {coco['map_50']:.4f} | @.75: {coco['map_75']:.4f}")
    t_coco_elapsed = time.time() - t_coco

    t_total = time.time() - t_total_start

    if args.output_dir:
        timing = {
            "total_seconds": round(t_total, 1),
            "total_minutes": round(t_total / 60, 1),
            "seconds_per_image": round(t_total / metrics['num_images'], 2),
            "coco_eval_seconds": round(t_coco_elapsed, 1),
        }
        save_results(Path(args.output_dir), metrics, args, coco, timing)

    print(f"\n{'='*70}")
    print(f"Total time: {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"Throughput: {metrics['num_images']/t_total:.2f} img/s "
          f"({t_total/metrics['num_images']:.2f} s/img)")
    print(f"{'='*70}")
    print("Done!")


if __name__ == "__main__":
    main()