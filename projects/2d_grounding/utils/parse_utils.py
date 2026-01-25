import json
import re
from typing import Any, Dict, List


def parse_json_markdown(json_output: str) -> str:
    """
    Parse out markdown fencing from the model response.
    """
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i + 1 :])
            json_output = json_output.split("```")[0]
            break

    if "```" not in json_output and "[" not in json_output[:10]:
        start = json_output.find("[")
        end = json_output.rfind("]")
        if start != -1 and end != -1:
            json_output = json_output[start : end + 1]
    return json_output


def parse_bounding_boxes(
    response_text: str,
    image_width: int,
    image_height: int,
    iou_threshold: float = 0.95,
) -> List[Dict[str, Any]]:
    """
    Parse bounding boxes and de-duplicate by label+IoU.
    Also robust to "broken JSON array" by salvaging per-object JSON blocks.
    """
    boxes: List[Dict[str, Any]] = []
    cleaned_text = parse_json_markdown(response_text)

    def get_bbox_field(data: Dict[str, Any]) -> Any:
        if "bbox_2d" in data:
            return data["bbox_2d"]
        if "bbox" in data:
            return data["bbox"]
        return None

    def convert_bbox(norm_box):
        try:
            n_x1, n_y1, n_x2, n_y2 = map(float, norm_box)
        except Exception:
            return None
        return {
            "x1": (n_x1 / 1000.0) * image_width,
            "y1": (n_y1 / 1000.0) * image_height,
            "x2": (n_x2 / 1000.0) * image_width,
            "y2": (n_y2 / 1000.0) * image_height,
        }

    def calculate_iou(box1, box2):
        x_left = max(box1["x1"], box2["x1"])
        y_top = max(box1["y1"], box2["y1"])
        x_right = min(box1["x2"], box2["x2"])
        y_bottom = min(box1["y2"], box2["y2"])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
        box2_area = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
        union_area = box1_area + box2_area - intersection_area
        return 0.0 if union_area == 0 else (intersection_area / union_area)

    # 1) Best case: parse whole JSON
    try:
        direct_json = json.loads(cleaned_text)
        if isinstance(direct_json, dict):
            direct_json = [direct_json]
        if isinstance(direct_json, list):
            for data in direct_json:
                if isinstance(data, dict) and "label" in data:
                    bbox = get_bbox_field(data)
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        scaled_bbox = convert_bbox(bbox[:4])
                        if scaled_bbox:
                            boxes.append({"label": str(data["label"]), **scaled_bbox})
    except Exception:
        pass

    # 2) Robust salvage: extract each { ... } object that has BOTH label and bbox/bbox_2d
    if not boxes:
        # order-agnostic lookaheads: must contain "label" and "bbox"/"bbox_2d"
        obj_pattern = r"\{(?=[^{}]*\"label\")(?=[^{}]*\"bbox(?:_2d)?\")[^{}]*\}"
        for m in re.findall(obj_pattern, cleaned_text, flags=re.IGNORECASE | re.DOTALL):
            try:
                data = json.loads(m)
                if isinstance(data, dict) and "label" in data:
                    bbox = get_bbox_field(data)
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        scaled_bbox = convert_bbox(bbox[:4])
                        if scaled_bbox:
                            boxes.append({"label": str(data["label"]), **scaled_bbox})
            except Exception:
                continue

    # 3) De-duplicate
    unique_boxes: List[Dict[str, Any]] = []
    for box in boxes:
        is_duplicate = False
        for existing_box in unique_boxes:
            if box["label"].lower() != existing_box["label"].lower():
                continue
            if calculate_iou(box, existing_box) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_boxes.append(box)

    return unique_boxes

if __name__ == "__main__":
    # This is the "broken JSON array" case you described:
    # - many valid objects
    # - one truncated / malformed object at the end
    test_response = """```json
    [
        {"bbox_2d": [0, 108, 47, 210], "label": "individual products and items"},
        {"bbox_2d": [45, 113, 86, 209], "label": "individual products and items"},
        {"bbox_2d": [75, 69, 218, 208], "label": "individual products and items"},
        {"bbox_2d": [215, 94, 250, 208], "label": "individual products and items"},
    ]        
    """

    # Choose any image size you want for scaling
    W, H = 1920, 1080

    results = parse_bounding_boxes(test_response, image_width=W, image_height=H)
    print(results)
    print(f"Parsed boxes: {len(results)}")
    for i, b in enumerate(results[:10]):  # show first 10
        print(
            f"{i:02d} label={b['label']!r} "
            f"x1={b['x1']:.1f} y1={b['y1']:.1f} x2={b['x2']:.1f} y2={b['y2']:.1f}"
        )

    # Basic sanity checks
    assert len(results) >= 4, "Should recover at least the valid objects before the broken one."
    assert all("label" in b for b in results)
    assert all(k in b for b in results for k in ("x1", "y1", "x2", "y2"))
    print("OK: salvaged valid boxes and ignored the broken trailing object.")