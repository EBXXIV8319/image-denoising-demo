from __future__ import annotations

import numpy as np


FREQUENCY_FILTERS = ["Gaussian Low-Pass", "Butterworth Low-Pass", "Butterworth Notch Reject"]


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
    notch_u: int = 0,
    notch_v: int = 81,
    notch_radius: float = 8.0,
) -> np.ndarray:
    if filter_type == "Gaussian Low-Pass":
        radius = radial_frequency_grid(shape)
        cutoff = float(np.clip(cutoff_percent / 100.0, 0.02, 0.98))
        response = np.exp(-(radius * radius) / (2 * cutoff * cutoff))
    elif filter_type == "Butterworth Low-Pass":
        radius = radial_frequency_grid(shape)
        cutoff = float(np.clip(cutoff_percent / 100.0, 0.02, 0.98))
        order = max(1, int(order))
        response = 1.0 / (1.0 + (radius / max(cutoff, 1e-6)) ** (2 * order))
    elif filter_type == "Butterworth Notch Reject":
        response = butterworth_notch_reject(shape, notch_u, notch_v, notch_radius, order, band_depth)
    else:
        raise ValueError(f"Unsupported frequency filter: {filter_type}")

    return np.clip(response, 0.0, 1.0).astype(np.float32)


def butterworth_notch_reject(
    shape: tuple[int, int],
    notch_u: int,
    notch_v: int,
    radius: float,
    order: int,
    depth: float,
) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    center_u = w / 2
    center_v = h / 2
    du = float(notch_u)
    dv = float(notch_v)
    radius = max(float(radius), 1e-6)
    order = max(1, int(order))
    depth = float(np.clip(depth, 0.0, 1.0))

    d_positive = np.sqrt((xx - center_u - du) ** 2 + (yy - center_v - dv) ** 2)
    d_negative = np.sqrt((xx - center_u + du) ** 2 + (yy - center_v + dv) ** 2)
    response = 1.0 / (1.0 + (radius * radius / np.maximum(d_positive * d_negative, 1e-12)) ** order)
    return 1.0 - depth * (1.0 - response)


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
