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

Evaluates VLM detection by running inference, parsing bounding boxes,
loading ground truth, and computing metrics (mAP, IoU, precision, recall).

Usage:
    # Single GPU (default)
    python eval_2d_grounding_multigpu.py config.yaml
    
    # Multi-GPU (e.g., 4 GPUs)
    python eval_2d_grounding_multigpu.py config.yaml --num_gpus 4
    
    # Use all available GPUs
    python eval_2d_grounding_multigpu.py config.yaml --num_gpus -1
"""

import csv
import json
import os
import sys
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
from eval.metrics import compute_map_box_only, compute_map_box_only_at_iou_thresholds


def load_ground_truth_from_csv(
    csv_path: str, image_dir: Optional[str] = None
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Tuple[int, int]]]:
    """
    Load ground truth bounding boxes from CSV file.
    
    CSV format: image_name,x1,y1,x2,y2,label,image_width,image_height
    
    Returns:
        (ground_truth dict, image_sizes dict)
    """
    ground_truth = {}
    image_sizes = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 7:
                continue

            image_name, x1, y1, x2, y2, label = row[0], *map(int, row[1:5]), row[5]
            img_width, img_height = (int(row[6]), int(row[7])) if len(row) > 7 else (None, None)

            # Validate box
            if x2 <= x1 or y2 <= y1:
                continue

            if image_name not in ground_truth:
                ground_truth[image_name] = []

            ground_truth[image_name].append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": label
            })

            if img_width and img_height:
                image_sizes[image_name] = (img_width, img_height)

    return ground_truth, image_sizes


def get_image_size(image_path: str) -> Tuple[int, int]:
    """Get image dimensions (width, height)."""
    try:
        return load_image(image_path).size
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        return (0, 0)


def evaluate_detection_results(
    model,
    processor,
    device: str,
    image_paths: List[str],
    ground_truth: Dict[str, List[Dict[str, Any]]],
    image_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
    prompt: Optional[str] = None,
    iou_threshold: float = 0.5,
    box_only: bool = True,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    repetition_penalty: float = 1.0,
    verbose: bool = True,
    gpu_id: int = 0,
    image_max_pixels: int = 15873000,
    image_min_pixels: int = 3136,
) -> Dict[str, Any]:
    """
    Evaluate detection results from VLM model.
    
    Returns:
        Dict with keys: map, precision, recall, mean_iou, num_images,
        predictions, ground_truths, image_names, responses
    """
    prompt = prompt or (
        "Locate every instance that belongs to the following categories: 'objects'. "
        "Report bbox coordinates in JSON format."
    )

    # Prepare data
    image_names = [Path(p).name for p in image_paths]
    all_ground_truths = [
        ground_truth.get(name, []).copy() for name in image_names
    ]

    # Run inference
    if verbose:
        print(f"[GPU {gpu_id}] Running inference on {len(image_paths)} images...")
    
    responses = inference_local_qwen3vl(
        model=model,
        processor=processor,
        device=device,
        img_urls=image_paths,
        prompts=[prompt] * len(image_paths),
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        verbose=verbose,
        max_pixels=image_max_pixels,
        min_pixels=image_min_pixels,
    )

    # Parse predictions
    if verbose:
        print(f"[GPU {gpu_id}] Parsing bounding boxes from responses...")
    
    all_predictions = []
    for i, (img_path, response, img_name) in enumerate(zip(image_paths, responses, image_names)):
        # Get image size
        if image_sizes and img_name in image_sizes:
            w, h = image_sizes[img_name]
        else:
            w, h = get_image_size(img_path)

        if w == 0 or h == 0:
            if verbose:
                print(f"[GPU {gpu_id}] Warning: Could not get size for {img_name}, skipping...")
            all_predictions.append([])
            continue

        boxes = parse_bounding_boxes(response, w, h, iou_threshold=0.95)
        boxes = validate_and_clip_boxes(boxes, w, h)
        all_predictions.append(boxes)

        if verbose and i % 10 == 0:
            print(f"[GPU {gpu_id}] Processed {i+1}/{len(image_paths)} images...")

    # Compute metrics
    if verbose:
        print(f"[GPU {gpu_id}] Computing evaluation metrics...")

    if box_only:
        metrics = compute_map_box_only(all_predictions, all_ground_truths, iou_threshold)
    else:
        from eval.metrics import compute_map
        metrics = compute_map(all_predictions, all_ground_truths, iou_threshold, True, False)

    # Build responses dict: image_name -> response
    responses_dict = {name: resp for name, resp in zip(image_names, responses)}

    metrics.update({
        "predictions": all_predictions,
        "ground_truths": all_ground_truths,
        "image_names": image_names,
        "responses": responses_dict
    })

    return metrics


def evaluate_on_single_gpu(
    gpu_id: int,
    image_paths: List[str],
    ground_truth: Dict[str, List[Dict[str, Any]]],
    image_sizes: Dict[str, Tuple[int, int]],
    args_dict: Dict[str, Any],
    return_dict: dict,
):
    """
    Worker function for single GPU evaluation.
    
    Args:
        gpu_id: GPU device index
        image_paths: List of image paths to process on this GPU
        ground_truth: Ground truth annotations
        image_sizes: Image dimensions dict
        args_dict: Arguments as dictionary (for multiprocessing compatibility)
        return_dict: Shared dict to store results
    """
    try:
        device = f"cuda:{gpu_id}"
        print(f"[GPU {gpu_id}] Starting evaluation on {len(image_paths)} images...")
        
        # Load model on this specific GPU
        model, processor, _ = load_local_qwen3vl_model(
            model_name=args_dict["model_name"],
            lora_path=args_dict.get("lora_path"),
            is_lora=args_dict.get("is_lora", False),
            device=device,
            model_scope=args_dict.get("use_modelscope", False),
            enable_hf_mirror=args_dict.get("enable_hf_mirror", False),
            hf_token=args_dict.get("hf_token"),
        )
        print(f"[GPU {gpu_id}] Model loaded successfully")
        
        # Run evaluation on this GPU's subset
        metrics = evaluate_detection_results(
            model=model,
            processor=processor,
            device=device,
            image_paths=image_paths,
            ground_truth=ground_truth,
            image_sizes=image_sizes,
            prompt=args_dict.get("prompt"),
            iou_threshold=args_dict.get("iou_threshold", 0.5),
            box_only=args_dict.get("box_only", True),
            batch_size=args_dict.get("batch_size", 1),
            max_new_tokens=args_dict.get("max_new_tokens", 8192),
            do_sample=args_dict.get("do_sample", False),
            repetition_penalty=args_dict.get("repetition_penalty", 1.1),
            verbose=(gpu_id == 0),  # Only first GPU prints verbose output
            gpu_id=gpu_id,
        )
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()
        
        return_dict[gpu_id] = metrics
        print(f"[GPU {gpu_id}] Evaluation completed. Processed {len(image_paths)} images.")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return_dict[gpu_id] = {"error": str(e)}


def merge_metrics(
    gpu_metrics: Dict[int, Dict[str, Any]],
    iou_threshold: float = 0.5,
    box_only: bool = True,
) -> Dict[str, Any]:
    """
    Merge metrics from multiple GPUs and recompute overall metrics.
    
    Args:
        gpu_metrics: Dict mapping GPU ID to metrics dict
        iou_threshold: IoU threshold for mAP computation
        box_only: Whether to use box-only evaluation
        
    Returns:
        Merged metrics dict
    """
    all_predictions = []
    all_ground_truths = []
    all_image_names = []
    all_responses = {}
    
    # Merge results in GPU order
    for gpu_id in sorted(gpu_metrics.keys()):
        m = gpu_metrics[gpu_id]
        
        # Skip GPUs that had errors
        if "error" in m:
            print(f"Warning: GPU {gpu_id} had an error, skipping its results")
            continue
            
        all_predictions.extend(m["predictions"])
        all_ground_truths.extend(m["ground_truths"])
        all_image_names.extend(m["image_names"])
        all_responses.update(m["responses"])
    
    # Recompute metrics on merged results
    if box_only:
        metrics = compute_map_box_only(all_predictions, all_ground_truths, iou_threshold)
    else:
        from eval.metrics import compute_map
        metrics = compute_map(all_predictions, all_ground_truths, iou_threshold, True, False)
    
    metrics.update({
        "predictions": all_predictions,
        "ground_truths": all_ground_truths,
        "image_names": all_image_names,
        "responses": all_responses,
    })
    
    return metrics


def evaluate_multi_gpu(
    args,
    test_images: List[str],
    ground_truth: Dict[str, List[Dict[str, Any]]],
    image_sizes: Dict[str, Tuple[int, int]],
    num_gpus: int = None,
) -> Dict[str, Any]:
    """
    Distribute evaluation across multiple GPUs.
    
    Args:
        args: Parsed arguments
        test_images: List of all test image paths
        ground_truth: Ground truth annotations
        image_sizes: Image dimensions dict
        num_gpus: Number of GPUs to use (None = all available)
        
    Returns:
        Merged metrics from all GPUs
    """
    available_gpus = torch.cuda.device_count()
    
    if num_gpus is None or num_gpus <= 0:
        num_gpus = available_gpus
    else:
        num_gpus = min(num_gpus, available_gpus)
    
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")
    
    print(f"\n{'='*70}")
    print(f"Multi-GPU Evaluation")
    print(f"{'='*70}")
    print(f"Available GPUs: {available_gpus}")
    print(f"Using GPUs: {num_gpus}")
    print(f"Total images: {len(test_images)}")
    print(f"Images per GPU: ~{len(test_images) // num_gpus}")
    print(f"{'='*70}\n")
    
    # Split images across GPUs (round-robin for better balance)
    chunks = [[] for _ in range(num_gpus)]
    for i, img_path in enumerate(test_images):
        chunks[i % num_gpus].append(img_path)
    
    for gpu_id, chunk in enumerate(chunks):
        print(f"GPU {gpu_id}: {len(chunk)} images")
    
    # Convert args to dict for multiprocessing
    args_dict = vars(args)
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Use multiprocessing manager for shared results
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Start processes
    processes = []
    for gpu_id in range(num_gpus):
        if not chunks[gpu_id]:
            continue
            
        p = mp.Process(
            target=evaluate_on_single_gpu,
            args=(
                gpu_id,
                chunks[gpu_id],
                ground_truth,
                image_sizes,
                args_dict,
                return_dict,
            )
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Check for errors
    for gpu_id, metrics in return_dict.items():
        if "error" in metrics:
            print(f"Error on GPU {gpu_id}: {metrics['error']}")
    
    # Merge results from all GPUs
    print("\nMerging results from all GPUs...")
    merged_metrics = merge_metrics(
        dict(return_dict),
        iou_threshold=args.iou_threshold,
        box_only=args.box_only,
    )
    
    return merged_metrics


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        from omegaconf import OmegaConf
        config = OmegaConf.load(Path(config_path).absolute())
        return OmegaConf.to_container(config, resolve=True)
    except ImportError:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def save_results(output_dir: Path, metrics: Dict, args, coco_metrics: Dict = None) -> None:
    """Save evaluation results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary metrics
    if coco_metrics:
        metrics_to_save = {
            "num_images": metrics["num_images"],
            "map": coco_metrics["map"],
            "map_50": coco_metrics["map_50"],
            "map_75": coco_metrics["map_75"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "mean_iou": metrics["mean_iou"]
        }
    else:
        metrics_to_save = {k: metrics[k] for k in ["num_images", "map", "precision", "recall", "mean_iou"]}
    
    metrics_to_save.update({
        "iou_threshold": args.iou_threshold,
        "box_only": args.box_only,
        "num_gpus": args.num_gpus,
    })
    if "map_per_class" in metrics:
        metrics_to_save["map_per_class"] = metrics["map_per_class"]
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    
    # Save detailed predictions
    detailed = [
        {
            "image": name,
            "num_predictions": len(pred),
            "num_ground_truth": len(gt),
            "predictions": pred,
            "ground_truth": gt
        }
        for name, pred, gt in zip(metrics["image_names"], metrics["predictions"], metrics["ground_truths"])
    ]
    
    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(detailed, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - metrics.json")
    print(f"  - detailed_results.json")
    print(f"  - responses.json")


def parse_args(config: Dict) -> Any:
    """Parse command-line arguments with config file defaults."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate 2D grounding detection results (Multi-GPU support)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument("--model_name", default=config.get("model_name", "Qwen/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--lora_path", default=config.get("lora_path", None))
    parser.add_argument("--is_lora", type=lambda x: str(x).lower() in ["true", "1", "yes"], 
                        default=config.get("is_lora", False))
    
    # Data arguments
    parser.add_argument("--images_dir", default=config.get("images_dir", "/root/Codes/data/SKU110K_fixed/images"))
    parser.add_argument("--annotations_csv", default=config.get("annotations_csv", 
                        "/root/Codes/data/SKU110K_fixed/annotations/annotations_test.csv"))
    parser.add_argument("--image_prefix", default=config.get("image_prefix", "test_"))
    parser.add_argument("--max_images", type=int, default=config.get("max_images", 10))
    
    # Evaluation arguments
    parser.add_argument("--iou_threshold", type=float, default=config.get("iou_threshold", 0.5))
    parser.add_argument("--box_only", type=lambda x: str(x).lower() in ["true", "1", "yes"], 
                        default=config.get("box_only", True))
    parser.add_argument("--prompt", default=config.get("prompt", None))
    
    # Inference arguments
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size", 1))
    parser.add_argument("--max_new_tokens", type=int, default=config.get("max_new_tokens", 8192))
    parser.add_argument("--do_sample", type=lambda x: str(x).lower() in ["true", "1", "yes"], 
                        default=config.get("do_sample", False))
    parser.add_argument("--repetition_penalty", type=float, default=config.get("repetition_penalty", 1.1))
    
    # GPU arguments
    parser.add_argument("--device", default=config.get("device", "auto"))
    parser.add_argument("--num_gpus", type=int, default=config.get("num_gpus", 1),
                        help="Number of GPUs to use. -1 for all available GPUs, 1 for single GPU")
    
    # Output arguments
    parser.add_argument("--output_dir", default=config.get("output_dir", None))
    
    # Model loading arguments
    parser.add_argument("--use_modelscope", action="store_true", default=config.get("use_modelscope", False))
    parser.add_argument("--enable_hf_mirror", type=lambda x: str(x).lower() in ["true", "1", "yes"], 
                        default=config.get("enable_hf_mirror", False))
    parser.add_argument("--hf_token", default=config.get("hf_token", None))
    parser.add_argument("--image_max_pixels", type=int, default=config.get("image_max_pixels", 15873000))
    parser.add_argument("--image_min_pixels", type=int, default=config.get("image_min_pixels", 3136))
    return parser.parse_args()


def main():
    """Main entry point."""
    # Load config if YAML file provided
    config = {}
    if len(sys.argv) > 1 and sys.argv[1].endswith((".yaml", ".yml")):
        config_path = sys.argv[1]
        print(f"Loading configuration from: {config_path}")
        config = load_config_from_yaml(config_path)
        sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove config from argv
    
    args = parse_args(config)
    
    print("=" * 70)
    print("2D Grounding Evaluation (Multi-GPU Support)")
    print("=" * 70)
    
    # Check GPU availability
    num_available_gpus = torch.cuda.device_count()
    print(f"\nAvailable GPUs: {num_available_gpus}")
    
    if num_available_gpus == 0:
        print("Warning: No CUDA GPUs available. Falling back to CPU.")
        args.num_gpus = 0
        args.device = "cpu"
    
    # Determine number of GPUs to use
    if args.num_gpus == -1:
        args.num_gpus = num_available_gpus
    elif args.num_gpus > num_available_gpus:
        print(f"Warning: Requested {args.num_gpus} GPUs but only {num_available_gpus} available.")
        args.num_gpus = num_available_gpus
    
    use_multi_gpu = args.num_gpus > 1

    # Load ground truth
    print(f"\nLoading ground truth from: {args.annotations_csv}")
    ground_truth, image_sizes = load_ground_truth_from_csv(args.annotations_csv)
    print(f"Loaded ground truth for {len(ground_truth)} images")

    # Get test images
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        exit(1)

    test_images = [
        str(p) for p in sorted(images_dir.glob(f"{args.image_prefix}*.jpg"))[:args.max_images]
    ]

    if not test_images:
        print(f"Error: No test images found in {images_dir} with prefix '{args.image_prefix}'")
        exit(1)

    print(f"\nFound {len(test_images)} test images")
    print(f"First 5: {', '.join([Path(p).name for p in test_images[:5]])}...")

    # Run evaluation
    print(f"\nMax new tokens: {args.max_new_tokens}")
    
    if use_multi_gpu:
        # Multi-GPU evaluation
        metrics = evaluate_multi_gpu(
            args=args,
            test_images=test_images,
            ground_truth=ground_truth,
            image_sizes=image_sizes,
            num_gpus=args.num_gpus,
        )
    else:
        # Single GPU evaluation (original behavior)
        print(f"\nLoading model: {args.model_name}")
        model, processor, device = load_local_qwen3vl_model(
            model_name=args.model_name,
            lora_path=args.lora_path,
            is_lora=args.is_lora,
            device=args.device,
            model_scope=args.use_modelscope,
            enable_hf_mirror=args.enable_hf_mirror,
            hf_token=args.hf_token
        )
        print(f"Model loaded on {device}")
        
        metrics = evaluate_detection_results(
            model, processor, device, test_images, ground_truth, image_sizes,
            args.prompt, args.iou_threshold, args.box_only, args.batch_size,
            args.max_new_tokens, args.do_sample, args.repetition_penalty, True,
            image_max_pixels=args.image_max_pixels,
            image_min_pixels=args.image_min_pixels,
        )

    # Print results
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Number of images: {metrics['num_images']}")
    print(f"mAP@{args.iou_threshold} (box-only): {metrics['map']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")

    if args.output_dir:
        # Save original responses
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        with open(Path(args.output_dir) / "responses.json", "w", encoding="utf-8") as f:
            json.dump(metrics["responses"], f, indent=2, ensure_ascii=False)

        print(f"Responses saved to: {Path(args.output_dir) / 'responses.json'}")

    # COCO-style evaluation
    print("\n" + "-" * 70)
    print("COCO-style mAP at multiple IoU thresholds:")
    coco_metrics = compute_map_box_only_at_iou_thresholds(
        metrics["predictions"], metrics["ground_truths"],
        [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )
    print(f"mAP@[0.5:0.95]: {coco_metrics['map']:.4f}")
    print(f"mAP@0.5: {coco_metrics['map_50']:.4f}")
    print(f"mAP@0.75: {coco_metrics['map_75']:.4f}")

    # Save results
    if args.output_dir:
        save_results(Path(args.output_dir), metrics, args, coco_metrics)

    print("\n" + "=" * 70)
    print("Evaluation completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()