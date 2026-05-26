from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import color, exposure, feature, filters, metrics


@dataclass(frozen=True)
class MetricResult:
    mse: float
    psnr: float
    ssim: float


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.clip(image, 0.0, 1.0)
    return color.rgb2gray(np.clip(image, 0.0, 1.0))


def magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(spectrum))
    return exposure.rescale_intensity(magnitude, out_range=(0.0, 1.0))


def edge_map(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    return feature.canny(filters.gaussian(gray, sigma=0.6), sigma=1.0).astype(float)


def histogram(image: np.ndarray, bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    gray = to_gray(image)
    values, edges = np.histogram(gray.ravel(), bins=bins, range=(0.0, 1.0), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, values


def axis_spectra(image: np.ndarray) -> dict[str, np.ndarray]:
    gray = to_gray(image)
    h, w = gray.shape
    spectrum_x = np.fft.fftshift(np.fft.fft(gray, axis=1), axes=1)
    spectrum_y = np.fft.fftshift(np.fft.fft(gray, axis=0), axes=0)
    profile_x = np.log1p(np.abs(spectrum_x[h // 2, :]))
    profile_y = np.log1p(np.abs(spectrum_y[:, w // 2]))
    return {
        "x_frequency": np.linspace(-0.5, 0.5, w) * w,
        "x_profile": profile_x,
        "y_frequency": np.linspace(-0.5, 0.5, h) * h,
        "y_profile": profile_y,
    }


def reference_metrics(reference: np.ndarray, image: np.ndarray) -> MetricResult:
    reference = np.clip(reference, 0.0, 1.0)
    image = np.clip(image, 0.0, 1.0)
    min_side = min(reference.shape[:2])
    if min_side < 3:
        ssim = float("nan")
    else:
        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        ssim = float(
            metrics.structural_similarity(
                reference,
                image,
                data_range=1.0,
                channel_axis=-1 if reference.ndim == 3 else None,
                win_size=win_size,
            )
        )
    return MetricResult(
        mse=float(metrics.mean_squared_error(reference, image)),
        psnr=float(metrics.peak_signal_noise_ratio(reference, image, data_range=1.0)),
        ssim=ssim,
    )
