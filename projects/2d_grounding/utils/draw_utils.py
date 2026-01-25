from typing import Any, Dict, Iterable, List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

from .box_utils import validate_and_clip_boxes


def load_font(size: int = 16) -> ImageFont.ImageFont:
    """
    Try to load a TTF font; fall back to default.
    """
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()


def draw_labeled_boxes(
    image: Image.Image,
    boxes: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    colors: Optional[Sequence[Any]] = None,
    line_width: int = 3,
    save_message: Optional[str] = None,
    show_if_no_output: bool = True,
) -> None:
    """
    Draw bounding boxes and labels on a PIL image.
    """
    image_width, image_height = image.size
    boxes = validate_and_clip_boxes(boxes, image_width, image_height)

    if not boxes:
        print("Warning: No valid bounding boxes to draw.")
        if output_path:
            image.save(output_path)
            print(save_message or f"Output image saved to: {output_path}")
        elif show_if_no_output:
            image.show()
        return

    draw = ImageDraw.Draw(image)
    font = load_font(16)

    if colors is None:
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (128, 0, 255),
        ]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        label = str(box.get("label", ""))
        color = colors[i % len(colors)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        if label:
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            label_y = max(0, y1 - text_height - 4)
            draw.rectangle(
                [x1, label_y, x1 + text_width + 4, y1],
                fill=color,
                outline=color,
            )
            draw.text((x1 + 2, label_y), label, fill=(255, 255, 255), font=font)

    if output_path:
        image.save(output_path)
        print(save_message or f"Output image saved to: {output_path}")
    elif show_if_no_output:
        image.show()
