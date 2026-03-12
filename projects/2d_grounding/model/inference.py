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
    temperature=1.0,
    top_p=1.0,
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
        temperature: Sampling temperature (only used when do_sample=True).
            Higher values produce more diverse outputs.
        top_p: Nucleus sampling cutoff (only used when do_sample=True).
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
            generate_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
            )
            if do_sample:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = top_p
            generated_ids = model.generate(**generate_kwargs)
        
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

def inference_with_split(
    model,
    processor,
    device,
    img_url,
    prompt,
    parse_fn,
    split_token="<SPLIT>",
    max_depth=2,
    overlap_ratio=0.1,
    iou_threshold=0.5,
    max_new_tokens=8192,
    do_sample=False,
    temperature=1.0,
    top_p=1.0,
    repetition_penalty=1.1,
    batch_size=4,
    prefer_deeper=True,
    remove_artifacts=True,
    artifact_margin=0.02,
    verbose=False,
    min_pixels=64 * 32 * 32,
    max_pixels=9800 * 32 * 32,
):
    """
    Recursive split-detect-merge inference driven by the model's <SPLIT> token.

    The model decides whether to split: if its output contains ``split_token``,
    the image is split into 4 overlapping quadrants and each is processed
    recursively.  Otherwise the output is parsed as bounding boxes.

    All detections from leaf nodes are mapped back to global coordinates and
    merged with depth-aware NMS.

    Uses existing pipeline modules:
      - pipeline.splitter.split_image_into_quadrants
      - pipeline.mapper.map_to_global, clip_bbox
      - pipeline.merger.merge_detections, remove_boundary_artifacts

    Args:
        model / processor / device: loaded Qwen3-VL components.
        img_url: path to the input image.
        prompt: detection prompt string.
        parse_fn: ``(response_text, img_w, img_h) -> List[dict]`` — your
            existing ``parse_bounding_boxes`` from utils.parse_utils.
        split_token: the token that signals "split this image".
        max_depth: maximum recursion depth (2 → up to 21 VLM calls).
        overlap_ratio: quadrant overlap fraction (matches splitter.py).
        iou_threshold: NMS suppression threshold.
        max_new_tokens: generation budget per VLM call.
        do_sample / temperature / top_p / repetition_penalty: generation params.
        batch_size: how many quadrant crops to batch per VLM call.
        prefer_deeper: if True, NMS prefers detections from deeper levels.
        remove_artifacts: if True, remove thin boundary slivers after merge.
        artifact_margin: margin ratio for artifact removal.
        verbose: print progress per depth level.
        min_pixels / max_pixels: image resize bounds for VLM.

    Returns:
        dict with keys:
            "detections": List[dict] — final merged boxes in global pixel coords,
                each with keys x1, y1, x2, y2, label.
            "split_tree": dict — recursion tree showing SPLIT/DETECT at each node.
            "num_vlm_calls": int — total VLM inference calls made.
            "raw_response": str — the depth-0 raw VLM response.
    """
    import numpy as np
    from PIL import Image as PILImage

    # Lazy imports from pipeline modules
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline import Detection
    from pipeline.splitter import split_image_into_quadrants
    from pipeline.mapper import map_to_global, clip_bbox
    from pipeline.merger import merge_detections, remove_boundary_artifacts as _remove_artifacts

    # Load the original image
    if os.path.exists(img_url):
        pil_img = PILImage.open(img_url).convert("RGB")
    else:
        raise ValueError(f"Image not found: {img_url}")

    image_np = np.array(pil_img)
    img_h, img_w = image_np.shape[:2]
    num_vlm_calls = 0

    def _run_vlm_batch(crop_list):
        """Run VLM on a list of (crop_np,) and return list of response strings."""
        nonlocal num_vlm_calls
        import tempfile
        tmp_paths = []
        try:
            for crop_np in crop_list:
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                PILImage.fromarray(crop_np).save(tmp, format="JPEG")
                tmp.close()
                tmp_paths.append(tmp.name)

            responses = inference_local_qwen3vl(
                model=model, processor=processor, device=device,
                img_urls=tmp_paths,
                prompts=[prompt] * len(tmp_paths),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample, temperature=temperature, top_p=top_p,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size, verbose=False,
                min_pixels=min_pixels, max_pixels=max_pixels,
            )
            num_vlm_calls += len(tmp_paths)
            if isinstance(responses, str):
                responses = [responses]
            return responses
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def _recurse(crop_np, region, depth, quad_id, qpath):
        """Recursively detect on a crop, splitting if the model says <SPLIT>."""
        rx1, ry1, rx2, ry2 = region
        rw, rh = int(rx2 - rx1), int(ry2 - ry1)
        h_crop, w_crop = crop_np.shape[:2]

        # Run VLM on this crop
        responses = _run_vlm_batch([crop_np])
        response = responses[0]

        tree_node = {
            "depth": depth, "quadrant_id": quad_id, "quadrant_path": qpath,
            "region": [rx1, ry1, rx2, ry2],
            "action": "detect", "raw_response": response[:200],
        }

        # Check for SPLIT token
        if split_token in response and depth < max_depth:
            tree_node["action"] = "split"
            tree_node["children"] = {}

            # Split the crop into quadrants
            quadrants = split_image_into_quadrants(crop_np, overlap_ratio)

            # Batch all 4 quadrant VLM calls
            child_crops = []
            child_infos = []
            for child_qid, qinfo in quadrants.items():
                child_crop = qinfo["image"]
                ox, oy = qinfo["offset"]
                cw, ch = qinfo["size"]
                child_region = (rx1 + ox, ry1 + oy, rx1 + ox + cw, ry1 + oy + ch)
                child_path = f"{qpath}-{child_qid}"
                child_crops.append(child_crop)
                child_infos.append((child_qid, child_region, child_path, cw, ch))

            # Batch inference on all 4 quadrants
            child_responses = _run_vlm_batch(child_crops)

            all_child_dets = []
            for (child_qid, child_region, child_path, cw, ch), child_resp, child_crop_np in \
                    zip(child_infos, child_responses, child_crops):

                # Check if child also wants to split
                if split_token in child_resp and depth + 1 < max_depth:
                    # Recurse deeper
                    child_dets, child_tree = _recurse(
                        child_crop_np, child_region, depth + 1, child_qid, child_path,
                    )
                    tree_node["children"][child_qid] = child_tree
                    all_child_dets.extend(child_dets)
                else:
                    # Parse detections
                    crx1, cry1, crx2, cry2 = child_region
                    crw, crh = int(crx2 - crx1), int(cry2 - cry1)
                    ch_crop, cw_crop = child_crop_np.shape[:2]

                    boxes = parse_fn(child_resp, cw_crop, ch_crop)
                    child_dets_raw = [
                        Detection(
                            bbox=(b["x1"], b["y1"], b["x2"], b["y2"]),
                            label=b.get("label", "object"),
                            confidence=b.get("confidence", 1.0),
                            depth=depth + 1, quadrant_id=child_qid,
                        )
                        for b in boxes
                    ]
                    # Map to global
                    global_dets = map_to_global(
                        child_dets_raw,
                        offset=(int(crx1), int(cry1)),
                        sub_size=(crw, crh),
                    )
                    # Clip
                    for det in global_dets:
                        cb = clip_bbox(det.bbox, img_w, img_h)
                        if cb[2] > cb[0] and cb[3] > cb[1]:
                            all_child_dets.append(Detection(
                                bbox=cb, label=det.label, confidence=det.confidence,
                                depth=det.depth, quadrant_id=det.quadrant_id,
                            ))

                    tree_node["children"][child_qid] = {
                        "depth": depth + 1, "quadrant_id": child_qid,
                        "quadrant_path": child_path,
                        "region": list(child_region),
                        "action": "detect",
                        "num_detections": len(child_dets_raw),
                    }

            return all_child_dets, tree_node

        else:
            # Parse as detections at this level
            boxes = parse_fn(response, w_crop, h_crop)
            dets_raw = [
                Detection(
                    bbox=(b["x1"], b["y1"], b["x2"], b["y2"]),
                    label=b.get("label", "object"),
                    confidence=b.get("confidence", 1.0),
                    depth=depth, quadrant_id=quad_id,
                )
                for b in boxes
            ]
            global_dets = map_to_global(
                dets_raw, offset=(int(rx1), int(ry1)), sub_size=(rw, rh),
            )
            clipped = []
            for det in global_dets:
                cb = clip_bbox(det.bbox, img_w, img_h)
                if cb[2] > cb[0] and cb[3] > cb[1]:
                    clipped.append(Detection(
                        bbox=cb, label=det.label, confidence=det.confidence,
                        depth=det.depth, quadrant_id=det.quadrant_id,
                    ))
            tree_node["num_detections"] = len(clipped)
            return clipped, tree_node

    # --- Run recursive inference ---
    full_region = (0.0, 0.0, float(img_w), float(img_h))
    all_dets, split_tree = _recurse(image_np, full_region, depth=0, quad_id=0, qpath="0")

    if verbose:
        print(f"  Recursive inference: {num_vlm_calls} VLM calls, "
              f"{len(all_dets)} raw detections")

    # --- Merge with NMS ---
    merged = merge_detections(all_dets, iou_threshold=iou_threshold, prefer_deeper=prefer_deeper)

    if remove_artifacts:
        merged = _remove_artifacts(merged, (img_w, img_h), artifact_margin)

    if verbose:
        print(f"  After NMS + artifact removal: {len(merged)} detections")

    # Convert Detection objects to dicts
    result_boxes = [
        {"x1": d.bbox[0], "y1": d.bbox[1], "x2": d.bbox[2], "y2": d.bbox[3],
         "label": d.label}
        for d in merged
    ]

    return {
        "detections": result_boxes,
        "split_tree": split_tree,
        "num_vlm_calls": num_vlm_calls,
        "raw_response": split_tree.get("raw_response", ""),
    }


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