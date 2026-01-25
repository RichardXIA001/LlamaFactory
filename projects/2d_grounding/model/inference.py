import os
import io
import requests
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info


def inference_local_qwen3vl(
    model,
    processor,
    device,
    img_urls,
    prompts,
    min_pixels=64 * 32 * 32,
    max_pixels=9800 * 32 * 32,
    max_new_tokens=8192,
    do_sample=False,
    repetition_penalty=1.1,
    batch_size=8,
    verbose=True,
):
    """
    Run safe batch inference with automatic chunking.
    
    Args:
        model: Loaded Qwen-VL model
        processor: Loaded processor
        device: torch.device
        img_urls: Single image path/URL or list of paths/URLs
        prompts: Single prompt (used for all images) or list of prompts (one per image)
        min_pixels: Minimum image pixels (will upscale if smaller)
        max_pixels: Maximum image pixels (will downscale if larger)
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to use sampling
        repetition_penalty: Repetition penalty
        batch_size: Number of images to process at once (adjust based on GPU memory)
        verbose: Print progress
    
    Returns:
        Single string (if single image input) or list of strings (if multiple images)
    """
    
    # ========== Sub-function 1: Load and resize image ==========
    def _load_and_resize_image(img_url):
        """Load image from path/URL and resize to enforce pixel bounds."""
        # Load image
        if os.path.exists(img_url):
            image = Image.open(img_url).convert("RGB")
        elif isinstance(img_url, str) and (img_url.startswith("http://") or img_url.startswith("https://")):
            resp = requests.get(img_url, timeout=60)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        else:
            raise ValueError(f"Invalid img_url: {img_url}")
        
        # Resize to enforce pixel bounds
        w, h = image.size
        pixels = w * h
        
        def _resize_keep_aspect(img, target_pixels):
            w0, h0 = img.size
            scale = (target_pixels / float(w0 * h0)) ** 0.5
            new_w, new_h = max(1, int(w0 * scale)), max(1, int(h0 * scale))
            return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        if pixels > max_pixels:
            image = _resize_keep_aspect(image, max_pixels)
        elif pixels < min_pixels:
            image = _resize_keep_aspect(image, min_pixels)
        
        return image
    
    # ========== Sub-function 2: Process a single batch ==========
    def _process_batch(batch_urls, batch_prompts):
        """Process a single batch of images."""
        # Load all images in this batch
        images = [_load_and_resize_image(url) for url in batch_urls]
        
        # Build messages for batch
        all_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for img, prompt in zip(images, batch_prompts)
        ]
        
        # Prepare inputs (chat template -> vision info -> processor)
        texts = [
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in all_messages
        ]
        
        # Process vision info for all messages
        all_image_inputs = []
        all_video_inputs = []
        for messages in all_messages:
            image_inputs, video_inputs = process_vision_info(messages)
            all_image_inputs.extend(image_inputs if image_inputs else [])
            all_video_inputs.extend(video_inputs if video_inputs else [])
        
        # Process all inputs together
        inputs = processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
            )
        
        # Trim prompt tokens and decode
        input_len = inputs["input_ids"].shape[1]
        generated_ids_trimmed = generated_ids[:, input_len:]
        
        outputs = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        return outputs
    
    # ========== Main function logic ==========
    
    # Handle single image input (convert to list)
    single_input = False
    if isinstance(img_urls, str):
        img_urls = [img_urls]
        single_input = True
    
    # Handle single prompt for all images
    if isinstance(prompts, str):
        prompts = [prompts] * len(img_urls)
    
    # Validate inputs
    if len(img_urls) != len(prompts):
        raise ValueError(
            f"Number of images ({len(img_urls)}) must match number of prompts ({len(prompts)})"
        )
    
    # Process in batches
    all_results = []
    total_images = len(img_urls)
    total_batches = (total_images + batch_size - 1) // batch_size
    import time
    start_time = time.time()
    for batch_idx in range(0, total_images, batch_size):

        batch_urls = img_urls[batch_idx:batch_idx + batch_size]
        batch_prompts = prompts[batch_idx:batch_idx + batch_size]
        
        current_batch = batch_idx // batch_size + 1
        if verbose:
            print(f"Processing batch {current_batch}/{total_batches} "
                  f"({len(batch_urls)} images)...")
        
        # Process this batch
        batch_results = _process_batch(batch_urls, batch_prompts)
        all_results.extend(batch_results)
        print(f"Time taken to process batch {current_batch}/{total_batches} "
                  f"({len(batch_urls)} images): {time.time() - start_time} seconds")
        # Clear GPU cache between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Return single string if single input, otherwise return list
    if single_input:
        return all_results[0]
    else:
        return all_results

if __name__ == "__main__":
    from loader import load_local_qwen3vl_model
    model, processor, device = load_local_qwen3vl_model(
    model_name="Qwen/Qwen3-VL-4B-Instruct",
    device="auto",
    dtype="auto",
    model_scope=True
)
    img_urls = ["/root/Codes/data/SKU110K_fixed/images/test_8.jpg"]
    prompts = ["This is a retail shelf image. Detect and locate every individual product on the shelf, treating each separate item as a distinct object. Report bounding box coordinates for all items in JSON format."]
    result = inference_local_qwen3vl(model, processor, device, img_urls, prompts)
    print(result)