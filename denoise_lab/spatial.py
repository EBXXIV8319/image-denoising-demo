from __future__ import annotations

import numpy as np
from scipy import ndimage


def mean_filter(image: np.ndarray, size: int) -> np.ndarray:
    size = max(1, int(size))
    if image.ndim == 3:
        result = np.stack([ndimage.uniform_filter(image[..., channel], size=size) for channel in range(image.shape[-1])], axis=-1)
    else:
        result = ndimage.uniform_filter(image, size=size)
    return np.clip(result, 0.0, 1.0)


def median_filter(image: np.ndarray, size: int) -> np.ndarray:
    size = max(1, int(size))
    if image.ndim == 3:
        result = np.stack([ndimage.median_filter(image[..., channel], size=size) for channel in range(image.shape[-1])], axis=-1)
    else:
        result = ndimage.median_filter(image, size=size)
    return np.clip(result, 0.0, 1.0)
