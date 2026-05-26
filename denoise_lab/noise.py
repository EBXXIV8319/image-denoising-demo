from __future__ import annotations

import numpy as np
from skimage import util


NOISE_TYPES = ["高斯噪声", "椒盐噪声", "混合噪声", "周期噪声", "无：上传图像已含噪"]


def add_noise(
    image: np.ndarray,
    noise_type: str,
    gaussian_sigma: float,
    sp_amount: float,
    periodic_strength: float,
    periodic_frequency: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.clip(image, 0.0, 1.0)

    if noise_type in {"无：上传图像已含噪", "None"}:
        return result

    if noise_type in {"高斯噪声", "混合噪声"}:
        result = util.random_noise(result, mode="gaussian", mean=0.0, var=float(gaussian_sigma) ** 2, rng=rng)

    if noise_type in {"椒盐噪声", "混合噪声"}:
        result = util.random_noise(result, mode="s&p", amount=float(sp_amount), rng=rng)

    if noise_type == "周期噪声":
        h = result.shape[0]
        yy = np.arange(h)[:, None]
        wave = np.sin(2 * np.pi * int(periodic_frequency) * yy / max(h, 1))
        result = result + float(periodic_strength) * wave[..., None]

    return np.clip(result, 0.0, 1.0)
