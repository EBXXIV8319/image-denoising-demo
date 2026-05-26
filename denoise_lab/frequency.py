from __future__ import annotations

import numpy as np


FREQUENCY_FILTERS = ["Gaussian Low-Pass", "Butterworth Low-Pass", "Band-Stop"]


def radial_frequency_grid(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    y = np.arange(h) - h / 2
    x = np.arange(w) - w / 2
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx * xx + yy * yy)
    max_radius = np.sqrt((h / 2) ** 2 + (w / 2) ** 2)
    return radius / max(max_radius, 1.0)


def frequency_response(
    shape: tuple[int, int],
    filter_type: str,
    cutoff_percent: float = 24.0,
    order: int = 3,
    band_center_percent: float = 22.0,
    band_width_percent: float = 4.0,
    band_depth: float = 0.95,
) -> np.ndarray:
    radius = radial_frequency_grid(shape)
    cutoff = float(np.clip(cutoff_percent / 100.0, 0.02, 0.98))

    if filter_type == "Gaussian Low-Pass":
        response = np.exp(-(radius * radius) / (2 * cutoff * cutoff))
    elif filter_type == "Butterworth Low-Pass":
        order = max(1, int(order))
        response = 1.0 / (1.0 + (radius / max(cutoff, 1e-6)) ** (2 * order))
    elif filter_type == "Band-Stop":
        center = float(np.clip(band_center_percent / 100.0, 0.02, 0.98))
        width = float(np.clip(band_width_percent / 100.0, 0.005, 0.40))
        depth = float(np.clip(band_depth, 0.0, 1.0))
        response = 1.0 - depth * np.exp(-((radius - center) ** 2) / (2 * width * width))
    else:
        raise ValueError(f"Unsupported frequency filter: {filter_type}")

    return np.clip(response, 0.0, 1.0).astype(np.float32)


def frequency_filter(image: np.ndarray, response: np.ndarray) -> np.ndarray:
    channels = [image] if image.ndim == 2 else [image[..., channel] for channel in range(image.shape[-1])]
    filtered = []
    for channel in channels:
        spectrum = np.fft.fftshift(np.fft.fft2(channel))
        restored = np.fft.ifft2(np.fft.ifftshift(spectrum * response))
        filtered.append(np.real(restored))

    if image.ndim == 2:
        return np.clip(filtered[0], 0.0, 1.0)
    return np.clip(np.stack(filtered, axis=-1), 0.0, 1.0)
