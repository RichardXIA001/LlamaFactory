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
Evaluation script for 2D grounding tasks with Multi-GPU support.

Usage:
    python eval_2d_grounding_multigpu.py config.yaml
    python eval_2d_grounding_multigpu.py config.yaml --num_gpus 4
    python eval_2d_grounding_multigpu.py config.yaml --num_gpus -1
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

from model.inference import inference_local_qwen3vl
from model.loader import load_local_qwen3vl_model
from utils.box_utils import validate_and_clip_boxes
from utils.image_utils import load_image
from utils.parse_utils import parse_bounding_boxes
from eval.metrics import compute_map_box_only, compute_coco_map_pycocotools


def load_ground_truth_from_csv(csv_path: str) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Tuple[int, int]]]:
    """Load ground truth bounding boxes from CSV file."""
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


def save_checkpoint(output_dir: Path, image_names, responses, all_predictions, all_ground_truths):
    """Save raw responses and parsed results BEFORE metric computation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "responses.json", "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
    detailed = [
        {"image": name, "num_predictions": len(pred), "num_ground_truth": len(gt),
         "predictions": pred, "ground_truth": gt}
        for name, pred, gt in zip(image_names, all_predictions, all_ground_truths)
    ]
    with open(output_dir / "detailed_results.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)
    print(f"  [Checkpoint] Saved responses + detections to: {output_dir}")


def evaluate_detection_results(
    model, processor, device, image_paths, ground_truth,
    image_sizes=None, prompt=None, iou_threshold=0.5, box_only=True,
    batch_size=1, max_new_tokens=512, do_sample=False, repetition_penalty=1.0,
    verbose=True, gpu_id=0, image_max_pixels=15873000, image_min_pixels=3136,
    output_dir=None,
) -> Dict[str, Any]:
    """Run inference, parse boxes, checkpoint, then compute metrics."""
    prompt = prompt or (
        "Locate every instance that belongs to the following categories: 'objects'. "
        "Report bbox coordinates in JSON format."
    )
    image_names = [Path(p).name for p in image_paths]
    all_ground_truths = [ground_truth.get(name, []).copy() for name in image_names]

    if verbose:
        print(f"[GPU {gpu_id}] Running inference on {len(image_paths)} images...")
    t_inference_start = time.time()
    responses = inference_local_qwen3vl(
        model=model, processor=processor, device=device,
        img_urls=image_paths, prompts=[prompt] * len(image_paths),
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        do_sample=do_sample, repetition_penalty=repetition_penalty,
        verbose=verbose, max_pixels=image_max_pixels, min_pixels=image_min_pixels,
    )
    t_inference = time.time() - t_inference_start
    avg_per_img = t_inference / len(image_paths) if image_paths else 0
    if verbose:
        print(f"[GPU {gpu_id}] Inference done in {t_inference:.1f}s "
              f"({avg_per_img:.2f}s/img, ~{avg_per_img*len(image_paths)/60:.1f}min total)")

    if verbose:
        print(f"[GPU {gpu_id}] Parsing bounding boxes...")
    t_parse_start = time.time()
    all_predictions = []
    for i, (img_path, response, img_name) in enumerate(zip(image_paths, responses, image_names)):
        w, h = image_sizes[img_name] if (image_sizes and img_name in image_sizes) else get_image_size(img_path)
        if w == 0 or h == 0:
            all_predictions.append([])
            continue
        boxes = validate_and_clip_boxes(parse_bounding_boxes(response, w, h, iou_threshold=0.95), w, h)
        all_predictions.append(boxes)
        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t_parse_start
            eta = elapsed / (i + 1) * (len(image_paths) - i - 1)
            print(f"[GPU {gpu_id}] Parsed {i+1}/{len(image_paths)} images "
                  f"({elapsed:.1f}s elapsed, ETA {eta:.1f}s)")
    t_parse = time.time() - t_parse_start
    if verbose:
        print(f"[GPU {gpu_id}] Parsing done in {t_parse:.1f}s")

    responses_dict = dict(zip(image_names, responses))

    # Checkpoint before metrics
    if output_dir:
        ckpt_dir = Path(output_dir) / f"checkpoint_gpu{gpu_id}" if gpu_id > 0 else Path(output_dir)
        save_checkpoint(ckpt_dir, image_names, responses_dict, all_predictions, all_ground_truths)

    if verbose:
        print(f"[GPU {gpu_id}] Computing metrics...")
    if box_only:
        metrics = compute_map_box_only(all_predictions, all_ground_truths, iou_threshold)
    else:
        from eval.metrics import compute_map
        metrics = compute_map(all_predictions, all_ground_truths, iou_threshold, True, False)

    metrics.update({"predictions": all_predictions, "ground_truths": all_ground_truths,
                    "image_names": image_names, "responses": responses_dict})
    return metrics


def evaluate_on_single_gpu(gpu_id, image_paths, ground_truth, image_sizes, args_dict, return_dict):
    """Worker function for single GPU evaluation."""
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
        metrics = evaluate_detection_results(
            model=model, processor=processor, device=device,
            image_paths=image_paths, ground_truth=ground_truth, image_sizes=image_sizes,
            prompt=args_dict.get("prompt"), iou_threshold=args_dict.get("iou_threshold", 0.5),
            box_only=args_dict.get("box_only", True), batch_size=args_dict.get("batch_size", 1),
            max_new_tokens=args_dict.get("max_new_tokens", 8192),
            do_sample=args_dict.get("do_sample", False),
            repetition_penalty=args_dict.get("repetition_penalty", 1.1),
            verbose=(gpu_id == 0), gpu_id=gpu_id, output_dir=args_dict.get("output_dir"),
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
    """Merge metrics from multiple GPUs."""
    all_preds, all_gts, all_names, all_resp = [], [], [], {}
    for gpu_id in sorted(gpu_metrics.keys()):
        m = gpu_metrics[gpu_id]
        if "error" in m:
            print(f"Warning: GPU {gpu_id} had an error, skipping")
            continue
        all_preds.extend(m["predictions"]); all_gts.extend(m["ground_truths"])
        all_names.extend(m["image_names"]); all_resp.update(m["responses"])

    metrics = compute_map_box_only(all_preds, all_gts, iou_threshold) if box_only else \
        __import__("eval.metrics", fromlist=["compute_map"]).compute_map(all_preds, all_gts, iou_threshold, True, False)
    metrics.update({"predictions": all_preds, "ground_truths": all_gts,
                    "image_names": all_names, "responses": all_resp})
    return metrics


def evaluate_multi_gpu(args, test_images, ground_truth, image_sizes, num_gpus=None):
    """Distribute evaluation across multiple GPUs."""
    available = torch.cuda.device_count()
    num_gpus = min(num_gpus or available, available)
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")

    print(f"\nMulti-GPU: {num_gpus} GPUs, {len(test_images)} images")
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


def save_results(output_dir: Path, metrics, args, coco_metrics=None, timing=None):
    """Save final evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if coco_metrics:
        m = {"num_images": metrics["num_images"],
             "map": coco_metrics["map"], "map_50": coco_metrics["map_50"], "map_75": coco_metrics["map_75"],
             "map_small": coco_metrics.get("map_small"), "map_medium": coco_metrics.get("map_medium"),
             "map_large": coco_metrics.get("map_large"),
             "precision": metrics["precision"], "recall": metrics["recall"], "mean_iou": metrics["mean_iou"]}
    else:
        m = {k: metrics[k] for k in ["num_images", "map", "precision", "recall", "mean_iou"]}
    m.update({"iou_threshold": args.iou_threshold, "box_only": args.box_only, "num_gpus": args.num_gpus})
    if timing:
        m["timing"] = timing
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    print(f"Final results saved to: {output_dir / 'metrics.json'}")


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
    parser = argparse.ArgumentParser(description="Evaluate 2D grounding (Multi-GPU)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    bool_type = lambda x: str(x).lower() in ["true", "1", "yes"]
    parser.add_argument("--model_name", default=config.get("model_name", "Qwen/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--lora_path", default=config.get("lora_path"))
    parser.add_argument("--is_lora", type=bool_type, default=config.get("is_lora", False))
    parser.add_argument("--images_dir", default=config.get("images_dir", "/root/Codes/data/SKU110K_fixed/images"))
    parser.add_argument("--annotations_csv", default=config.get("annotations_csv",
                        "/root/Codes/data/SKU110K_fixed/annotations/annotations_test.csv"))
    parser.add_argument("--image_prefix", default=config.get("image_prefix", "test_"))
    parser.add_argument("--max_images", type=int, default=config.get("max_images", 10))
    parser.add_argument("--iou_threshold", type=float, default=config.get("iou_threshold", 0.5))
    parser.add_argument("--box_only", type=bool_type, default=config.get("box_only", True))
    parser.add_argument("--prompt", default=config.get("prompt"))
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size", 1))
    parser.add_argument("--max_new_tokens", type=int, default=config.get("max_new_tokens", 8192))
    parser.add_argument("--do_sample", type=bool_type, default=config.get("do_sample", False))
    parser.add_argument("--repetition_penalty", type=float, default=config.get("repetition_penalty", 1.1))
    parser.add_argument("--device", default=config.get("device", "auto"))
    parser.add_argument("--num_gpus", type=int, default=config.get("num_gpus", 1))
    parser.add_argument("--output_dir", default=config.get("output_dir"))
    parser.add_argument("--use_modelscope", action="store_true", default=config.get("use_modelscope", False))
    parser.add_argument("--enable_hf_mirror", type=bool_type, default=config.get("enable_hf_mirror", False))
    parser.add_argument("--hf_token", default=config.get("hf_token"))
    parser.add_argument("--image_max_pixels", type=int, default=config.get("image_max_pixels", 15873000))
    parser.add_argument("--image_min_pixels", type=int, default=config.get("image_min_pixels", 3136))
    return parser.parse_args()


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
    print(f"Found {len(test_images)} test images, GT for {len(ground_truth)} images")

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
        metrics = evaluate_detection_results(
            model, processor, device, test_images, ground_truth, image_sizes,
            args.prompt, args.iou_threshold, args.box_only, args.batch_size,
            args.max_new_tokens, args.do_sample, args.repetition_penalty, True,
            image_max_pixels=args.image_max_pixels, image_min_pixels=args.image_min_pixels,
            output_dir=args.output_dir)

    # Checkpoint merged results before COCO mAP
    if args.output_dir:
        save_checkpoint(Path(args.output_dir), metrics["image_names"],
                        metrics["responses"], metrics["predictions"], metrics["ground_truths"])

    # Basic results
    print(f"\n{'='*70}\nResults: {metrics['num_images']} images")
    print(f"mAP@{args.iou_threshold}: {metrics['map']:.4f} | P: {metrics['precision']:.4f} | "
          f"R: {metrics['recall']:.4f} | IoU: {metrics['mean_iou']:.4f}")

    # COCO mAP via pycocotools
    t_coco = time.time()
    coco = compute_coco_map_pycocotools(
        metrics["predictions"], metrics["ground_truths"],
        image_sizes=image_sizes, image_names=metrics["image_names"])
    print(f"\nCOCO mAP@[.5:.95]: {coco['map']:.4f} | @.5: {coco['map_50']:.4f} | @.75: {coco['map_75']:.4f}")
    t_coco_elapsed = time.time() - t_coco
    print(f"  (COCO eval took {t_coco_elapsed:.1f}s)")

    t_total = time.time() - t_total_start

    if args.output_dir:
        timing = {
            "total_seconds": round(t_total, 1),
            "total_minutes": round(t_total / 60, 1),
            "seconds_per_image": round(t_total / metrics['num_images'], 2),
            "images_per_second": round(metrics['num_images'] / t_total, 2),
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