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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

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


def main():
    parser = argparse.ArgumentParser(
        description="Detect objects in an image using Qwen3VL-4B model"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the output image with bounding boxes (default: input_path with '_detected' suffix)"
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
    
    # Validate input image
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Set output path
    if args.output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_detected{image_path.suffix}"
    else:
        output_path = Path(args.output_path)
    
    # Load image
    try:
        image = load_image(str(image_path), convert_mode="RGB")
        print(f"Loaded image: {image_path} (size: {image.size})")
    except Exception as e:
        print(f"Error loading image: {e}")
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
            print("     python scripts/detect_objects_qwen3vl.py --use_modelscope ...")
            print("  2. Login to Hugging Face:")
            print("     huggingface-cli login")
            print("  3. Use token:")
            print("     python scripts/detect_objects_qwen3vl.py --hf_token YOUR_TOKEN ...")
            print("  4. Set environment variable:")
            print("     export HF_TOKEN=your_token_here")
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            print("\n[SOLUTION] Network error detected!")
            print("Try using ModelScope (Chinese mirror):")
            print("  python scripts/detect_objects_qwen3vl.py --use_modelscope ...")
        
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Prepare the prompt
    print("[LOG] Preparing prompt...")
    if args.prompt is None:
        img_width, img_height = image.size
        prompt = (
         'Find all individual products and items in this shelf image. Detect each separate item independently, even if they are similar or touching. Return bounding box coordinates for every single item in JSON format.'
        )
    else:
        prompt = args.prompt
    print(f"[LOG] Prompt prepared (length: {len(prompt)} chars)")
    
    # Prepare messages in Qwen3VL format
    print("[LOG] Preparing messages...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    print("[LOG] Messages prepared")
    
    # Prepare inputs using the processor
    print("[LOG] Running object detection...")
    try:
        # Qwen3VL processor can handle messages directly with images
        print("[LOG] Processing messages with processor...")
        print("[LOG] Processor type:", type(processor).__name__)
        
        # Try different approaches based on processor API
        print("[LOG] Attempting to process messages...")
        # --- 修改前的代码 (你的代码) ---
        # inputs = processor(messages, padding=True, return_tensors="pt") 
        # ... (以及后面的 except 块)

        # --- 修改后的代码 (推荐) ---
        print("[LOG] Processing inputs...")
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 关键：使用 process_vision_info 处理图片分辨率和网格
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],              # 注意这里必须是列表
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        
        print("[LOG] Moving inputs to device...")
        inputs = inputs.to(device)
        print(f"[LOG] Inputs prepared. Keys: {list(inputs.keys())}")
        if 'input_ids' in inputs:
            print(f"[LOG] Input IDs shape: {inputs['input_ids'].shape}")
        
        print("[LOG] Starting model generation...")
        with torch.no_grad():
            # --- 修改后的代码 ---
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=2048,      # 检测任务通常不需要特别长
                do_sample=False,         # ❌ 关闭采样，使用贪婪搜索，结果更稳定
                repetition_penalty=1.1,  # ✅ 增加重复惩罚，防止 "coffee, shelf, coffee..."
                # temperature=0.1,       # do_sample=False 时 temperature 无效，可注释掉
            )
        print(f"[LOG] Generation complete. Output shape: {generated_ids.shape}")
        
        # Decode the response (remove input tokens)
        print("[LOG] Decoding response...")
        input_length = inputs['input_ids'].shape[1]
        generated_ids_trimmed = generated_ids[:, input_length:]
        print(f"[LOG] Trimmed generated IDs shape: {generated_ids_trimmed.shape}")
        
        response_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        print(f"[LOG] Response decoded (length: {len(response_text)} chars)")
        print("\n[LOG] Model response received:")
        print("-" * 70)
        print(response_text)
        print("-" * 70)
        
        # Parse bounding boxes
        print("[LOG] Parsing bounding boxes from response...")
        img_w, img_h = image.size
        boxes = parse_bounding_boxes(response_text, img_w, img_h)        
        print(f"[LOG] Parsed {len(boxes)} bounding box(es)")
        if not boxes:
            print("\nWarning: Could not parse bounding boxes from the response.")
            print("The model response may not contain structured bounding box information.")
            print("\nThis could happen if:")
            print("1. The model doesn't support bounding box output in this format")
            print("2. The response format is different than expected")
            print("3. The image doesn't contain detectable objects")
            print("\nYou can:")
            print("- Try a different/custom prompt using --prompt")
            print("- Manually check the model response above")
            print("- The full response has been saved in the JSON output file")
            
            # Still save the response even if no boxes were parsed
            json_output_path = output_path.with_suffix('.json')
            with open(json_output_path, 'w') as f:
                json.dump({
                    'image_path': str(image_path),
                    'image_size': image.size,
                    'detections': [],
                    'model_response': response_text,
                    'note': 'No bounding boxes could be parsed from the response'
                }, f, indent=2)
            print(f"\nResponse saved to: {json_output_path}")
            sys.exit(1)
        
        print(f"\n[LOG] Detected {len(boxes)} object(s):")
        for i, box in enumerate(boxes, 1):
            print(f"  {i}. {box['label']}: [{box['x1']:.1f}, {box['y1']:.1f}, {box['x2']:.1f}, {box['y2']:.1f}]")
        
        # Draw bounding boxes on the image
        print("[LOG] Drawing bounding boxes on image...")
        draw_labeled_boxes(image, boxes, str(output_path))
        print("[LOG] Bounding boxes drawn")
        
        # Also save the detection results as JSON
        print("[LOG] Saving detection results to JSON...")
        json_output_path = output_path.with_suffix('.json')
        with open(json_output_path, 'w') as f:
            json.dump({
                'image_path': str(image_path),
                'image_size': image.size,
                'detections': boxes,
                'model_response': response_text
            }, f, indent=2)
        print(f"[LOG] Detection results saved to: {json_output_path}")
        print("[LOG] All done!")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
