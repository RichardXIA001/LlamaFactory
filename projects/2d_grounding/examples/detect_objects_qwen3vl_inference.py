#!/usr/bin/env python3
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
Object detection script using Qwen3VL-4B model.
Detects objects in an image and outputs bounding boxes with labels.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from utils.draw_utils import draw_labeled_boxes
from utils.image_utils import load_image
from utils.parse_utils import parse_bounding_boxes

from qwen_vl_utils import process_vision_info  # 需要 pip install qwen-vl-utils

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def get_image_paths(images_dir: Path, split: str) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    candidates = [p for p in images_dir.iterdir() if is_image_file(p)]
    if split in ("train", "test"):
        prefix = f"{split}_"
        candidates = [p for p in candidates if p.name.startswith(prefix)]

    return sorted(candidates)


def build_output_paths(
    image_path: Path,
    output_dir: Optional[Path] = None,
    output_path_override: Optional[Path] = None,
) -> Tuple[Path, Path]:
    if output_path_override is not None:
        output_img_path = output_path_override
        output_pred_path = output_path_override.with_suffix(".json")
        return output_img_path, output_pred_path

    if output_dir is None:
        raise ValueError("output_dir is required when output_path_override is not provided.")

    output_img_path = output_dir / "img" / image_path.name
    output_pred_path = output_dir / "predictions" / f"{image_path.stem}.json"
    return output_img_path, output_pred_path


def build_batch_inputs(processor, images: List[Any], prompt: str, device: str):
    messages_batch = []
    for image in images:
        messages_batch.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        )

    texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]

    image_inputs_batch = []
    video_inputs_batch = []
    for messages in messages_batch:
        image_inputs, video_inputs = process_vision_info(messages)
        image_inputs_batch.append(image_inputs)
        video_inputs_batch.append(video_inputs)

    inputs = processor(
        text=texts,
        images=image_inputs_batch,
        videos=video_inputs_batch,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(device)


def decode_batch_outputs(processor, generated_ids, inputs) -> List[str]:
    input_length = inputs["input_ids"].shape[1]
    generated_ids_trimmed = generated_ids[:, input_length:]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def run_batch_inference(
    image_paths: List[Path],
    output_dir: Optional[Path],
    output_path_override: Optional[Path],
    model,
    processor,
    device: str,
    prompt: str,
    batch_size: int,
) -> None:
    total = len(image_paths)
    if total == 0:
        print("[WARNING] No images found to process.")
        return

    if output_dir is not None:
        (output_dir / "img").mkdir(parents=True, exist_ok=True)
        (output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    for start in range(0, total, batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = []
        for image_path in batch_paths:
            try:
                image = load_image(str(image_path), convert_mode="RGB")
                images.append(image)
            except Exception as e:
                print(f"[ERROR] Failed to load image {image_path}: {e}")
                images.append(None)

        valid_items = [(p, img) for p, img in zip(batch_paths, images) if img is not None]
        if not valid_items:
            continue

        batch_paths = [item[0] for item in valid_items]
        images = [item[1] for item in valid_items]

        inputs = build_batch_inputs(processor, images, prompt, device)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                repetition_penalty=1.2,
            )

        responses = decode_batch_outputs(processor, generated_ids, inputs)

        for image_path, image, response_text in zip(batch_paths, images, responses):
            print("-" * 70)
            print(f"[LOG] Response for {image_path.name} (length: {len(response_text)} chars)")
            print("-" * 70)

            img_w, img_h = image.size
            boxes = parse_bounding_boxes(response_text, img_w, img_h)
            print(f"[LOG] Parsed {len(boxes)} bounding box(es) for {image_path.name}")

            output_img_path, output_pred_path = build_output_paths(
                image_path,
                output_dir=output_dir,
                output_path_override=output_path_override if len(image_paths) == 1 else None,
            )

            draw_labeled_boxes(image, boxes, str(output_img_path))

            with open(output_pred_path, "w") as f:
                json.dump(
                    {
                        "image_path": str(image_path),
                        "image_size": image.size,
                        "detections": boxes,
                    },
                    f,
                    indent=2,
                )
            print(f"[LOG] Saved predictions to: {output_pred_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Detect objects in an image using Qwen3VL-4B model"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to the input image (optional for batch mode)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the output image with bounding boxes (single image mode only)"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/root/Codes/data/SKU110K_fixed/images",
        help="Directory containing input images for batch inference",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "all"],
        default="test",
        help="Dataset split to process (train/test/all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for batch inference (will create ./img and ./predictions)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Model name or path (default: Qwen/Qwen3-VL-4B-Instruct)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for object detection (default: uses a standard prompt)"
    )
    parser.add_argument(
        "--use_modelscope",
        action="store_true",
        help="Use ModelScope (Chinese mirror) instead of Hugging Face"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token for authentication (or set HF_TOKEN env var)"
    )
    
    args = parser.parse_args()
    
    # Validate input mode
    image_path = Path(args.image_path) if args.image_path else None
    if image_path is None and args.output_dir is None:
        print("Error: --output_dir is required for batch mode.")
        sys.exit(1)
    
    # Determine device
    print("[LOG] Determining device...")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[LOG] Using device: {device}")
    
    # Handle ModelScope or Hugging Face
    use_modelscope = args.use_modelscope or os.getenv("USE_MODELSCOPE_HUB", "").lower() == "true"
    model_path = None  # Will store ModelScope path if used
    if use_modelscope:
        print("[LOG] Using ModelScope (Chinese mirror)...")
        try:
            from modelscope import snapshot_download  # type: ignore
            from modelscope.hub.api import HubApi  # type: ignore
        except ImportError:
            print("[ERROR] ModelScope not installed. Install with: pip install modelscope")
            sys.exit(1)
    
    # Handle Hugging Face token
    hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token and not use_modelscope:
        print("[LOG] Using Hugging Face token for authentication...")
        os.environ["HF_TOKEN"] = hf_token
        from huggingface_hub import login
        try:
            login(token=hf_token)
            print("[LOG] Hugging Face login successful!")
        except Exception as e:
            print(f"[WARNING] Failed to login to Hugging Face: {e}")
    
    # Initialize the model and processor
    print("[LOG] Loading Qwen3VL processor...")
    try:
        if use_modelscope:
            # Download from ModelScope first
            print(f"[LOG] Downloading model from ModelScope: {args.model_name}")
            model_path = snapshot_download(args.model_name, cache_dir=None)
            print(f"[LOG] Model downloaded to: {model_path}")
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
        else:
            processor = AutoProcessor.from_pretrained(
                args.model_name,
                trust_remote_code=True,
                token=hf_token
            )
        print("[LOG] Processor loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load processor: {e}")
        print("\n[TIP] If you're in China, try:")
        print("  1. Use ModelScope: --use_modelscope")
        print("  2. Or login to Hugging Face: huggingface-cli login")
        print("  3. Or set token: --hf_token YOUR_TOKEN")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("[LOG] Loading Qwen3VL model configuration...")
    try:
        if use_modelscope:
            # Use the already downloaded model path
            if model_path is None:
                model_path = snapshot_download(args.model_name, cache_dir=None)
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            load_path = model_path
        else:
            config = AutoConfig.from_pretrained(
                args.model_name,
                trust_remote_code=True,
                token=hf_token
            )
            load_path = args.model_name
        
        print(f"[LOG] Config loaded. Model type: {config.model_type}")
        
        # Determine the correct model class based on config
        print("[LOG] Determining correct model class...")
        if type(config) in AutoModelForImageTextToText._model_mapping.keys():
            load_class = AutoModelForImageTextToText
            print("[LOG] Using AutoModelForImageTextToText")
        elif type(config) in AutoModelForVision2Seq._model_mapping.keys():
            load_class = AutoModelForVision2Seq
            print("[LOG] Using AutoModelForVision2Seq")
        else:
            load_class = AutoModelForCausalLM
            print("[LOG] Using AutoModelForCausalLM (fallback)")
        
        print("[LOG] Loading model weights...")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        print(f"[LOG] Using dtype: {dtype}")
        
        model = load_class.from_pretrained(
            load_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            token=hf_token if not use_modelscope else None
        )
        model.eval()
        print(f"[LOG] Model loaded successfully on {device}!")
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Failed to load model: {error_msg}")
        
        # Provide helpful suggestions
        if "401" in error_msg or "Unauthorized" in error_msg:
            print("\n[SOLUTION] Authentication error detected!")
            print("Options:")
            print("  1. Use ModelScope (recommended for China):")
            print("     python /root/Codes/LlamaFactory/2d_grounding/test/detect_objects_qwen3vl.py --use_modelscope ...")
            print("  2. Login to Hugging Face:")
            print("     huggingface-cli login")
            print("  3. Use token:")
            print("     python /root/Codes/LlamaFactory/2d_grounding/test/detect_objects_qwen3vl.py --hf_token YOUR_TOKEN ...")
            print("  4. Set environment variable:")
            print("     export HF_TOKEN=your_token_here")
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            print("\n[SOLUTION] Network error detected!")
            print("Try using ModelScope (Chinese mirror):")
            print("  python /root/Codes/LlamaFactory/2d_grounding/test/detect_objects_qwen3vl.py --use_modelscope ...")
        
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Prepare the prompt
    print("[LOG] Preparing prompt...")
    if args.prompt is None:
        # prompt = (
        #     f"Please detect all objects in this image (size: {img_width}x{img_height} pixels). "
        #     f"For each object you detect, provide the bounding box coordinates.\n\n"
        #     "Requirements:\n"
        #     "1. Identify each object with a clear label/name\n"
        #     "2. Provide bounding box coordinates in the format [x1, y1, x2, y2] where:\n"
        #     f"   - x1, y1 are the top-left corner coordinates (in pixels)\n"
        #     f"   - x2, y2 are the bottom-right corner coordinates (in pixels)\n"
        #     f"   - x coordinates range from 0 to {img_width}\n"
        #     f"   - y coordinates range from 0 to {img_height}\n\n"
        #     "Please format your response as JSON array:\n"
        #     '[{"label": "object_name", "bbox": [x1, y1, x2, y2]}, ...]\n\n'
        #     "Or use this format (one per line):\n"
        #     "object_name: [x1, y1, x2, y2]"
        # )
        # prompt = (
        # "Detect every single item in the image, and there should be a lot of items in the image."
        # "Output the bounding box coordinates for each detected object.\n"
        # "Format your output as a list of JSON objects: "
        # '[{"label": "object_name", "bbox": [ymin, xmin, ymax, xmax]}, ...]\n'
        # "Note: The bounding box coordinates should be normalized to [0, 1000]." 
        # )
        prompt = (
            'Locate every instance that belongs to the following categories: "objects". '
            "Report bbox coordinates in JSON format."
        )
    else:
        prompt = args.prompt
    print(f"[LOG] Prompt prepared (length: {len(prompt)} chars)")

    print("[LOG] Running object detection...")
    try:
        if image_path is not None:
            if not image_path.exists():
                print(f"Error: Image file not found: {image_path}")
                sys.exit(1)

            if args.output_path is None:
                output_path = image_path.parent / f"{image_path.stem}_detected{image_path.suffix}"
            else:
                output_path = Path(args.output_path)

            run_batch_inference(
                [image_path],
                output_dir=None,
                output_path_override=output_path,
                model=model,
                processor=processor,
                device=device,
                prompt=prompt,
                batch_size=max(1, args.batch_size),
            )
        else:
            images_dir = Path(args.images_dir)
            output_dir = Path(args.output_dir) if args.output_dir else None
            image_paths = get_image_paths(images_dir, args.split)
            print(f"[LOG] Found {len(image_paths)} images in split '{args.split}'.")

            run_batch_inference(
                image_paths,
                output_dir=output_dir,
                output_path_override=None,
                model=model,
                processor=processor,
                device=device,
                prompt=prompt,
                batch_size=max(1, args.batch_size),
            )

        print("[LOG] All done!")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
