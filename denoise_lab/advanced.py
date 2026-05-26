from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import restoration

from .analysis import to_gray


def bilateral_filter(image: np.ndarray, sigma_color: float, sigma_spatial: float) -> np.ndarray:
    result = restoration.denoise_bilateral(
        image,
        sigma_color=float(sigma_color),
        sigma_spatial=float(sigma_spatial),
        channel_axis=-1 if image.ndim == 3 else None,
    )
    return np.clip(result, 0.0, 1.0)


def nlm_filter(image: np.ndarray, h: float, patch_size: int, patch_distance: int) -> np.ndarray:
    try:
        sigma = np.mean(restoration.estimate_sigma(image, channel_axis=-1 if image.ndim == 3 else None))
    except ImportError:
        gray = to_gray(image)
        high_pass = gray - ndimage.median_filter(gray, size=3)
        sigma = 1.4826 * np.median(np.abs(high_pass - np.median(high_pass)))

    result = restoration.denoise_nl_means(
        image,
        h=max(float(h) * sigma, 0.01),
        patch_size=int(patch_size),
        patch_distance=int(patch_distance),
        fast_mode=True,
        channel_axis=-1 if image.ndim == 3 else None,
    )
    return np.clip(result, 0.0, 1.0)
