from typing import Any, Dict, List, Sequence


def convert_bbox_2d_to_abs(bbox_2d: Sequence[float], width: int, height: int) -> Dict[str, int]:
    """
    Convert Qwen-style normalized bbox_2d ([0, 1000] range) to absolute pixels.
    """
    abs_x1 = int(bbox_2d[0] / 1000 * width)
    abs_y1 = int(bbox_2d[1] / 1000 * height)
    abs_x2 = int(bbox_2d[2] / 1000 * width)
    abs_y2 = int(bbox_2d[3] / 1000 * height)

    if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1
    if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

    abs_x1 = max(0, min(abs_x1, width))
    abs_y1 = max(0, min(abs_y1, height))
    abs_x2 = max(0, min(abs_x2, width))
    abs_y2 = max(0, min(abs_y2, height))

    return {"x1": abs_x1, "y1": abs_y1, "x2": abs_x2, "y2": abs_y2}


def validate_and_clip_boxes(
    boxes: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
) -> List[Dict[str, Any]]:
    """
    Validate and clip bounding boxes to image boundaries.
    """
    validated_boxes = []
    for box in boxes:
        x1 = max(0, min(box["x1"], image_width - 1))
        y1 = max(0, min(box["y1"], image_height - 1))
        x2 = max(x1 + 1, min(box["x2"], image_width))
        y2 = max(y1 + 1, min(box["y2"], image_height))

        if x2 > x1 and y2 > y1:
            validated_boxes.append(
                {
                    "label": box.get("label", ""),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
        else:
            print(
                "Warning: Skipping invalid box for "
                f"'{box.get('label', '')}': "
                f"[{box['x1']}, {box['y1']}, {box['x2']}, {box['y2']}]"
            )

    return validated_boxes
