#!/usr/bin/env python3

"""
Example command:

HF_ENDPOINT=https://hf-mirror.com python /root/Codes/LlamaFactory/2d_grounding/test/detect_objects_qwen3vl_official.py \
  --mode local \
  --model_path Qwen/Qwen3-VL-4B-Instruct \
  --image /root/Codes/data/SKU110K_fixed/images/test_0.jpg \
  --output /root/Codes/data/SKU110K_fixed/detections/test_0_detected.jpg

"""
import argparse
import ast
import base64
import os
import sys
from PIL import ImageColor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.box_utils import convert_bbox_2d_to_abs
from utils.draw_utils import draw_labeled_boxes
from utils.image_utils import load_image
from utils.parse_utils import parse_json_markdown

# =============================================================================
# Part 1: Official Utility Functions (Reused from 2d_grounding.ipynb)
# =============================================================================

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def plot_bounding_boxes(im, bounding_boxes, output_path=None):
    """
    Plots bounding boxes on an image.
    (Adapted from Official Qwen3-VL Notebook to support saving)
    """
    # Load the image
    img = im.copy()
    width, height = img.size
    print(f"[LOG] Image size: {img.size}")
    
    # Define a list of colors
    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 
        'gray', 'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 
        'maroon', 'teal', 'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes_text = parse_json_markdown(bounding_boxes)

    try:
        # Use ast.literal_eval for safe evaluation of the string representation
        json_output = ast.literal_eval(bounding_boxes_text)
    except Exception as e:
        print(f"[WARNING] JSON parsing failed: {e}")
        # Simple fallback recovery
        try:
            end_idx = bounding_boxes_text.rfind('" புள்ளies') # Adjust based on errors if needed
            if end_idx == -1: end_idx = bounding_boxes_text.rfind('"}') + len('"}')
            truncated_text = bounding_boxes_text[:end_idx] + "]"
            json_output = ast.literal_eval(truncated_text)
        except:
            print("[ERROR] Could not parse model output. Raw output:")
            print(bounding_boxes)
            return

    if not isinstance(json_output, list):
        json_output = [json_output]

    print(f"[LOG] Detected {len(json_output)} objects.")

    boxes = []
    for bounding_box in json_output:
        if "bbox_2d" not in bounding_box:
            continue

        bbox = bounding_box.get("bbox_2d")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue

        abs_bbox = convert_bbox_2d_to_abs(bbox[:4], width, height)
        label = str(bounding_box.get("label", ""))
        boxes.append({"label": label, **abs_bbox})

    draw_labeled_boxes(
        img,
        boxes,
        output_path=output_path,
        colors=colors,
        save_message=f"[SUCCESS] Result saved to: {output_path}" if output_path else None,
    )

# =============================================================================
# Part 2: API Inference (Reused from 2d_grounding.ipynb)
# =============================================================================

def inference_with_openai_api(img_path, prompt, api_key, base_url, model_name):
    """
    Official API implementation using OpenAI client.
    """
    from openai import OpenAI
    
    if not api_key:
        raise ValueError("API Key is required for API mode.")

    # Encode image
    if os.path.exists(img_path):
        with open(img_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            img_url = f"data:image/jpeg;base64,{base64_image}"
    elif img_path.startswith("http"):
        img_url = img_path
    else:
        raise ValueError("Invalid image path or URL")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": img_url},
                    # "min_pixels": ... # Optional: control pixels
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    print(f"[LOG] Sending request to API (Model: {model_name})...")
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.001, # Keep low for deterministic coordinates
    )
    return completion.choices[0].message.content

# =============================================================================
# Part 3: Local Inference (Custom implementation matching Qwen3-VL specs)
# =============================================================================

def inference_local(img_path, prompt, model_path, device):
    """
    Local inference using transformers and qwen_vl_utils.
    Supports local model weights and Qwen3-VL dynamic resolution.
    """
    import torch
    from transformers import AutoModel, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print(f"[LOG] Loading model from: {model_path} on {device}")
    
    # Load model
    # Note: Qwen3-VL typically uses the Qwen2VL architecture class in transformers
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("[LOG] Model loaded successfully.")

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": img_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preprocessing
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process vision info (Dynamic resolution handling)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Generation
    print("[LOG] Generating...")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,       # Greedy search for coordinates
        repetition_penalty=1.1 # Prevent loops
    )
    
    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Object Detection (Official Implementation)")
    
    # Input/Output
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Path to save output image (default: input_detected.jpg)")
    
    # Mode Selection
    parser.add_argument("--mode", type=str, choices=["local", "api"], default="local", help="Inference mode")
    
    # Local Settings
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", 
                        help="Local model path or HF Hub ID (For Qwen3, point to your Qwen3 weights)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for local inference")
    
    # API Settings
    parser.add_argument("--api_key", type=str, default=os.environ.get("DASHSCOPE_API_KEY"), help="API Key")
    # parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API Base URL")
    parser.add_argument("--base_url", type=str, default="https://realmrouter.cn/v1", help="API Base URL")

    parser.add_argument("--api_model", type=str, default="qwen3-vl-235b-a22b-instruct", help="API Model Name")
    
    args = parser.parse_args()

    # Determine Output Path
    if args.output is None:
        base, ext = os.path.splitext(args.image)
        args.output = f"{base}_detected{ext}"

    # Default Prompt for Detection (as seen in official examples)
    prompt = "Detect all objects in this image. output the objects in json format with bbox_2d and label."

    print(f"[LOG] Processing image: {args.image}")

    try:
        if args.mode == "local":
            response = inference_local(args.image, prompt, args.model_path, args.device)
        else:
            response = inference_with_openai_api(
                args.image, prompt, args.api_key, args.base_url, args.api_model
            )
            
        print("-" * 40)
        print(f"[LOG] Raw Model Output:\n{response}")
        print("-" * 40)

        # Visualization
        # Need to open original image for drawing
        img = load_image(args.image, convert_mode=None)
            
        plot_bounding_boxes(img, response, output_path=args.output)
        
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()