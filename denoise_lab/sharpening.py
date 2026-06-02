from __future__ import annotations

import numpy as np
from scipy import ndimage


def unsharp_mask(image: np.ndarray, radius: float, amount: float, threshold: float = 0.0) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    sigma = max(float(radius), 0.1)
    strength = max(float(amount), 0.0)
    threshold = max(float(threshold), 0.0)

    if image.ndim == 3:
        blurred = np.stack([ndimage.gaussian_filter(image[..., channel], sigma=sigma) for channel in range(image.shape[-1])], axis=-1)
    else:
        blurred = ndimage.gaussian_filter(image, sigma=sigma)

    detail = image - blurred
    if threshold > 0:
        detail = np.where(np.abs(detail) >= threshold, detail, 0.0)

    return np.clip(image + strength * detail, 0.0, 1.0)
