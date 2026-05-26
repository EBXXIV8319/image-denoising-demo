from __future__ import annotations

import numpy as np
from PIL import Image
from skimage import util


def pil_to_float_rgb(image: Image.Image, max_side: int = 900) -> np.ndarray:
    image = image.convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(background, image).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def float_to_uint8(image: np.ndarray) -> np.ndarray:
    return util.img_as_ubyte(np.clip(image, 0.0, 1.0))
