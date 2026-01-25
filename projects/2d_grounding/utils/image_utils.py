from typing import Optional

import requests
from PIL import Image


def load_image(image_path: str, convert_mode: Optional[str] = "RGB") -> Image.Image:
    """
    Load an image from a local path or URL.
    """
    if image_path.startswith(("http://", "https://")):
        response = requests.get(image_path, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)
    else:
        image = Image.open(image_path)

    if convert_mode:
        image = image.convert(convert_mode)
    return image
