from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage, stats

from .analysis import to_gray


@dataclass(frozen=True)
class NoReferenceResult:
    niqe_like: float
    sharpness: float
    contrast: float
    naturalness: float

    def to_dict(self) -> dict:
        return asdict(self)


def niqe_like_quality(image: np.ndarray) -> NoReferenceResult:
    gray = to_gray(image)
    gray = np.clip(gray, 0.0, 1.0).astype(np.float64)

    mu = ndimage.gaussian_filter(gray, sigma=1.2)
    sigma = np.sqrt(np.maximum(ndimage.gaussian_filter(gray * gray, sigma=1.2) - mu * mu, 0.0))
    mscn = (gray - mu) / (sigma + 1.0 / 255.0)
    values = mscn.ravel()

    skew = float(abs(stats.skew(values, bias=False, nan_policy="omit")))
    kurtosis = float(abs(stats.kurtosis(values, fisher=False, bias=False, nan_policy="omit") - 3.0))
    naturalness = float(np.nan_to_num(skew + kurtosis, nan=10.0, posinf=10.0, neginf=10.0))

    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    gradient = np.sqrt(gx * gx + gy * gy)
    sharpness = float(np.mean(gradient))
    contrast = float(np.std(gray))

    blur_penalty = 1.0 / (sharpness + 1e-3)
    contrast_penalty = 1.0 / (contrast + 1e-3)
    niqe_like = float(naturalness + 0.08 * blur_penalty + 0.02 * contrast_penalty)

    return NoReferenceResult(
        niqe_like=niqe_like,
        sharpness=sharpness,
        contrast=contrast,
        naturalness=naturalness,
    )
