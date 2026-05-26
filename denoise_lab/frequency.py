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
    notches: list[dict[str, float]] | None = None,
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
        if notches is None:
            notches = [{"u": notch_u, "v": notch_v, "radius": notch_radius, "order": order, "depth": band_depth}]
        response = butterworth_notch_reject(shape, notches)
    else:
        raise ValueError(f"Unsupported frequency filter: {filter_type}")

    return np.clip(response, 0.0, 1.0).astype(np.float32)


def butterworth_notch_reject(
    shape: tuple[int, int],
    notches: list[dict[str, float]],
) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    center_u = w / 2
    center_v = h / 2
    response = np.ones(shape, dtype=np.float64)

    for notch in notches:
        du = float(notch.get("u", 0))
        dv = float(notch.get("v", 0))
        radius = max(float(notch.get("radius", 8.0)), 1e-6)
        order = max(1, int(notch.get("order", 2)))
        depth = float(np.clip(notch.get("depth", 0.95), 0.0, 1.0))

        d_positive = np.sqrt((xx - center_u - du) ** 2 + (yy - center_v - dv) ** 2)
        d_negative = np.sqrt((xx - center_u + du) ** 2 + (yy - center_v + dv) ** 2)
        pair_response = 1.0 / (1.0 + (radius * radius / np.maximum(d_positive * d_negative, 1e-12)) ** order)
        response *= 1.0 - depth * (1.0 - pair_response)

    return np.clip(response, 0.0, 1.0)


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
