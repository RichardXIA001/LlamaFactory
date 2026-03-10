"""
Convert RFT rejection sampling output into LlamaFactory SFT training format.

This script:
  1. Reads the RFT output JSON (from rejection_sampling.py)
  2. For each example:
     - If depth==0: uses the original full image
     - If depth>0: crops the sub-region from the original image and saves it
  3. Converts the target format:
     - DETECT nodes: bbox -> bbox_2d, wrapped in markdown code block
     - SPLIT nodes: "<SPLIT>" as the assistant response
  4. Writes a LlamaFactory-compatible JSON dataset

Output structure:
    {output_images_dir}/
        train_155_d1_q0-1.jpg       # cropped sub-region
        train_155_d1_q0-2.jpg
        ...
    {output_json}                    # LlamaFactory dataset JSON

Usage:
    python -m utils.create_rft_dataset configs/rft_test_single.yaml

    # Or with explicit paths:
    python utils/create_rft_dataset.py \
        --rft_json rft_results/sft_rft_training_data.json \
        --images_dir /root/Codes/data/SKU110K_fixed/images \
        --output_images_dir /root/Codes/data/SKU110K_fixed/SKU110K_rft_260309 \
        --output_json /root/Codes/LlamaFactory/data/sku110k_rft_260309.json
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# Handle imports for both module and standalone script usage
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.image_utils import load_image


# ====================================================================
# Format conversion
# ====================================================================

def format_detect_target(target_json: str) -> str:
    """Convert RFT detect target to LlamaFactory assistant format.

    Input:  '[{"label":"object","bbox":[100,200,300,400]}, ...]'
    Output: '```json\n[\n\t{"bbox_2d": [100,200,300,400], "label": "object"}, ...\n]\n```'
    """
    dets = json.loads(target_json)

    # Convert bbox -> bbox_2d and reorder keys to match existing SFT format
    formatted_lines = []
    for det in dets:
        bbox = det.get("bbox", det.get("bbox_2d", []))
        label = det.get("label", "object")
        formatted_lines.append(
            f'\t{{"bbox_2d": {json.dumps(bbox)}, "label": "{label}"}}'
        )

    inner = ",\n".join(formatted_lines)
    return f"```json\n[\n{inner}\n]\n```"


def format_split_target() -> str:
    """Format the SPLIT target for LlamaFactory."""
    return "<SPLIT>"


def build_sft_example(
    image_path: str,
    prompt: str,
    target: str,
    is_split: bool,
    system_message: str = "You are a helpful assistant for retail product detection.",
) -> Dict[str, Any]:
    """Build one LlamaFactory SFT training example."""
    if is_split:
        assistant_content = format_split_target()
    else:
        assistant_content = format_detect_target(target)

    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"<image>{prompt}"},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [image_path],
    }


# ====================================================================
# Image cropping
# ====================================================================

def get_crop_image_path(
    image_name: str,
    depth: int,
    quadrant_path: str,
    output_images_dir: str,
) -> str:
    """Generate a unique filename for a cropped sub-region."""
    stem = Path(image_name).stem
    ext = Path(image_name).suffix or ".jpg"
    if depth == 0:
        # Full image — no cropping needed, but copy/symlink to output dir
        return os.path.join(output_images_dir, f"{stem}{ext}")
    else:
        # Sub-region: encode depth and quadrant path
        qpath_safe = quadrant_path.replace("-", "_")
        return os.path.join(output_images_dir, f"{stem}_d{depth}_q{qpath_safe}{ext}")


def crop_and_save(
    source_image_path: str,
    crop_region: List[float],
    output_path: str,
) -> bool:
    """Crop a region from the source image and save it.

    Returns True if successful, False otherwise.
    """
    if os.path.exists(output_path):
        return True  # Already cropped (resume support)

    try:
        img = load_image(source_image_path, convert_mode="RGB")
        x1, y1, x2, y2 = [int(v) for v in crop_region]
        crop = img.crop((x1, y1, x2, y2))
        crop.save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"  WARNING: Failed to crop {output_path}: {e}")
        return False


# ====================================================================
# Main conversion
# ====================================================================

def convert_rft_to_sft(
    rft_json_path: str,
    images_dir: str,
    output_images_dir: str,
    output_json_path: str,
    system_message: str = "You are a helpful assistant for retail product detection.",
    verbose: bool = True,
    default_prompt: str = None
) -> List[Dict]:
    """Convert RFT output to LlamaFactory SFT format.

    Args:
        rft_json_path:      Path to the RFT output JSON.
        images_dir:         Directory with original source images.
        output_images_dir:  Where to save cropped sub-region images.
        output_json_path:   Where to write the LlamaFactory dataset JSON.
        system_message:     System prompt for the SFT examples.
        verbose:            Print progress.

    Returns:
        List of SFT examples (also written to output_json_path).
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)

    # Load RFT data
    with open(rft_json_path) as f:
        rft_data = json.load(f)

    if verbose:
        print(f"Loaded {len(rft_data)} RFT examples from {rft_json_path}")
        n_split = sum(1 for ex in rft_data if ex["target"] == "<SPLIT>")
        print(f"  SPLIT: {n_split}, DETECT: {len(rft_data) - n_split}")

    sft_examples = []
    skipped = 0

    for i, rft_ex in enumerate(rft_data):
        image_name = rft_ex["image_name"]
        depth = rft_ex["depth"]
        quadrant_path = rft_ex["quadrant_path"]
        crop_region = rft_ex["crop_region"]
        if default_prompt:
            prompt = default_prompt
        else:
            prompt = rft_ex["prompt"]
        target = rft_ex["target"]
        is_split = target == "<SPLIT>"

        source_path = os.path.join(images_dir, image_name)
        if not os.path.exists(source_path):
            if verbose:
                print(f"  WARNING: source image not found: {source_path}")
            skipped += 1
            continue

        # Determine output image path
        crop_path = get_crop_image_path(
            image_name, depth, quadrant_path, output_images_dir,
        )

        # Crop (or copy full image for depth 0)
        if depth == 0:
            # For depth 0, either symlink or just point to original
            # We copy to keep the dataset self-contained
            if not os.path.exists(crop_path):
                img = load_image(source_path, convert_mode="RGB")
                img.save(crop_path, quality=95)
        else:
            success = crop_and_save(source_path, crop_region, crop_path)
            if not success:
                skipped += 1
                continue

        # Build SFT example
        sft_ex = build_sft_example(
            image_path=crop_path,
            prompt=prompt,
            target=target,
            is_split=is_split,
            system_message=system_message,
        )
        sft_examples.append(sft_ex)

    # Save
    with open(output_json_path, "w") as f:
        json.dump(sft_examples, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\nConversion complete:")
        print(f"  Total SFT examples: {len(sft_examples)}")
        print(f"  Skipped: {skipped}")
        n_split_out = sum(
            1 for ex in sft_examples
            if ex["messages"][2]["content"] == "<SPLIT>"
        )
        print(f"  SPLIT examples: {n_split_out}")
        print(f"  DETECT examples: {len(sft_examples) - n_split_out}")
        print(f"  Cropped images saved to: {output_images_dir}")
        print(f"  SFT dataset saved to: {output_json_path}")

    return sft_examples


# ====================================================================
# CLI
# ====================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert RFT rejection sampling output to LlamaFactory SFT format",
    )
    # Support both config-based and explicit-path modes
    parser.add_argument("config_or_rft_json",
                        help="Path to YAML config (same as rejection_sampling) "
                             "or directly to the RFT output JSON")
    parser.add_argument("--images_dir", default=None,
                        help="Directory with original images (overrides config)")
    parser.add_argument("--output_images_dir", default=None,
                        help="Where to save cropped images (default: SKU110K_rft "
                             "next to original images)")
    parser.add_argument("--output_json", default=None,
                        help="Output LlamaFactory dataset JSON path")
    parser.add_argument("--system_message", default=None,
                        help="System prompt for SFT examples")
    parser.add_argument("--prompt", default=None,
                        help="Prompt for SFT examples")
    args = parser.parse_args()

    prompt = args.prompt or "This is a retail shelf image. Detect and locate every individual product on the shelf, treating each separate item as a distinct object. Report bounding box coordinates for all items in JSON format."

    input_path = args.config_or_rft_json

    # --- Detect if input is a YAML config or directly a JSON ---
    if input_path.endswith((".yaml", ".yml")):
        import yaml
        with open(input_path) as f:
            cfg = yaml.safe_load(f)

        output_dir = cfg.get("output_dir", "./rft_results")
        test_image = cfg.get("test_image")

        # Find the RFT output JSON
        if test_image:
            rft_json = os.path.join(output_dir, "test_single_image_sft.json")
        else:
            rft_json = os.path.join(output_dir, "sft_rft_training_data.json")

        images_dir = args.images_dir or cfg.get("images_dir")
        # For single-image mode, derive images_dir from test_image
        if not images_dir and test_image:
            images_dir = str(Path(test_image).parent)

        system_message = args.system_message or cfg.get("system_message",
            "You are a helpful assistant for retail product detection.")

    else:
        # Direct JSON path
        rft_json = input_path
        images_dir = args.images_dir
        system_message = args.system_message or \
            "You are a helpful assistant for retail product detection."

    if not images_dir:
        print("ERROR: --images_dir is required when not using a YAML config")
        return

    if not os.path.exists(rft_json):
        print(f"ERROR: RFT output not found: {rft_json}")
        return

    # --- Resolve output paths ---
    output_images_dir = args.output_images_dir or \
        os.path.join(os.path.dirname(images_dir), "SKU110K_rft")

    output_json = args.output_json or \
        os.path.join(os.path.dirname(os.path.abspath(rft_json)), "sku110k_rft_sft.json")

    print(f"RFT input:       {rft_json}")
    print(f"Source images:   {images_dir}")
    print(f"Output images:   {output_images_dir}")
    print(f"Output JSON:     {output_json}")
    print()

    convert_rft_to_sft(
        rft_json_path=rft_json,
        images_dir=images_dir,
        output_images_dir=output_images_dir,
        output_json_path=output_json,
        system_message=system_message,
        default_prompt=prompt,
    )


if __name__ == "__main__":
    main()