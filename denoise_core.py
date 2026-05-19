from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image
from scipy import ndimage, signal
from skimage import color, exposure, feature, filters, metrics, restoration, util


FilterName = Literal[
    "mean",
    "median",
    "frequency",
    "bilateral",
    "nlm",
]


@dataclass(frozen=True)
class MetricResult:
    mse: float
    psnr: float
    ssim: float


def pil_to_float_rgb(image: Image.Image, max_side: int = 900) -> np.ndarray:
    """Convert an uploaded image to an RGB float array in [0, 1]."""
    image = image.convert("RGBA")
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(bg, image).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def float_to_uint8(image: np.ndarray) -> np.ndarray:
    return util.img_as_ubyte(np.clip(image, 0.0, 1.0))


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return color.rgb2gray(np.clip(image, 0.0, 1.0))


def add_noise(
    image: np.ndarray,
    noise_type: str,
    gaussian_sigma: float,
    sp_amount: float,
    speckle_var: float,
    periodic_strength: float,
    periodic_frequency: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.clip(image, 0.0, 1.0)

    if noise_type in {"无：上传图像已含噪", "None"}:
        return result

    if noise_type in {"高斯噪声", "混合噪声"}:
        result = util.random_noise(
            result,
            mode="gaussian",
            mean=0.0,
            var=float(gaussian_sigma) ** 2,
            rng=rng,
        )

    if noise_type in {"椒盐噪声", "混合噪声"}:
        result = util.random_noise(
            result,
            mode="s&p",
            amount=float(sp_amount),
            rng=rng,
        )

    if noise_type == "斑点噪声":
        result = util.random_noise(
            result,
            mode="speckle",
            var=float(speckle_var),
            rng=rng,
        )

    if noise_type == "周期噪声":
        h, w = result.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        wave = np.sin(2 * np.pi * periodic_frequency * xx / max(w, 1))
        wave += np.cos(2 * np.pi * periodic_frequency * yy / max(h, 1))
        wave = periodic_strength * wave[..., None] / 2.0
        result = result + wave

    return np.clip(result, 0.0, 1.0)


def mean_filter(image: np.ndarray, size: int) -> np.ndarray:
    size = max(1, int(size))
    if image.ndim == 3:
        return np.stack(
            [ndimage.uniform_filter(image[..., c], size=size) for c in range(image.shape[-1])],
            axis=-1,
        )
    return ndimage.uniform_filter(image, size=size)


def median_filter(image: np.ndarray, size: int) -> np.ndarray:
    size = max(1, int(size))
    if image.ndim == 3:
        return np.stack(
            [ndimage.median_filter(image[..., c], size=size) for c in range(image.shape[-1])],
            axis=-1,
        )
    return ndimage.median_filter(image, size=size)


def bilateral_filter(image: np.ndarray, sigma_color: float, sigma_spatial: float) -> np.ndarray:
    return restoration.denoise_bilateral(
        image,
        sigma_color=float(sigma_color),
        sigma_spatial=float(sigma_spatial),
        channel_axis=-1 if image.ndim == 3 else None,
    )


def nlm_filter(image: np.ndarray, h: float, patch_size: int, patch_distance: int) -> np.ndarray:
    try:
        sigma = np.mean(
            restoration.estimate_sigma(
                image,
                channel_axis=-1 if image.ndim == 3 else None,
            )
        )
    except ImportError:
        gray = to_gray(image)
        high_pass = gray - ndimage.median_filter(gray, size=3)
        sigma = 1.4826 * np.median(np.abs(high_pass - np.median(high_pass)))
    strength = max(float(h), 0.01)
    return restoration.denoise_nl_means(
        image,
        h=max(strength * sigma, 0.01),
        patch_size=int(patch_size),
        patch_distance=int(patch_distance),
        fast_mode=True,
        channel_axis=-1 if image.ndim == 3 else None,
    )


def radial_frequency_grid(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    y = np.arange(h) - h / 2
    x = np.arange(w) - w / 2
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx * xx + yy * yy)
    max_radius = np.sqrt((h / 2) ** 2 + (w / 2) ** 2)
    return radius / max(max_radius, 1.0)


def _iir_radial_response(
    radius: np.ndarray,
    family: str,
    cutoff: float,
    order: int,
    ripple: float,
    attenuation: float,
) -> np.ndarray:
    cutoff = float(np.clip(cutoff, 0.02, 0.98))
    order = int(np.clip(order, 1, 8))
    ripple = float(np.clip(ripple, 0.01, 6.0))
    attenuation = float(np.clip(attenuation, 10.0, 100.0))

    if family == "Chebyshev I":
        sos = signal.cheby1(order, ripple, cutoff, btype="lowpass", output="sos")
    elif family == "Chebyshev II":
        sos = signal.cheby2(order, attenuation, cutoff, btype="lowpass", output="sos")
    elif family == "Elliptic":
        sos = signal.ellip(order, ripple, attenuation, cutoff, btype="lowpass", output="sos")
    else:
        raise ValueError(f"Unsupported IIR family: {family}")

    w, h = signal.sosfreqz(sos, worN=4096)
    normalized_w = w / np.pi
    response = np.abs(h)
    response = response / max(np.max(response), 1e-12)
    return np.interp(radius, normalized_w, response)


def frequency_response(
    shape: tuple[int, int],
    filter_type: str,
    cutoff_percent: float,
    order: int,
    ripple: float,
    attenuation: float,
    band_center_percent: float,
    band_width_percent: float,
    band_depth: float,
) -> np.ndarray:
    radius = radial_frequency_grid(shape)
    cutoff = float(np.clip(cutoff_percent / 100.0, 0.02, 0.98))

    if filter_type == "Ideal Low-Pass":
        response = (radius <= cutoff).astype(np.float32)
    elif filter_type == "Gaussian Low-Pass":
        response = np.exp(-(radius * radius) / (2 * cutoff * cutoff))
    elif filter_type == "Butterworth Low-Pass":
        order = max(1, int(order))
        response = 1.0 / (1.0 + (radius / max(cutoff, 1e-6)) ** (2 * order))
    elif filter_type in {"Chebyshev I", "Chebyshev II", "Elliptic"}:
        response = _iir_radial_response(radius, filter_type, cutoff, order, ripple, attenuation)
    elif filter_type == "Band-Stop":
        center = float(np.clip(band_center_percent / 100.0, 0.02, 0.98))
        width = float(np.clip(band_width_percent / 100.0, 0.005, 0.40))
        depth = float(np.clip(band_depth, 0.0, 1.0))
        response = 1.0 - depth * np.exp(-((radius - center) ** 2) / (2 * width * width))
    else:
        raise ValueError(f"Unsupported frequency filter: {filter_type}")

    return np.clip(response, 0.0, 1.0).astype(np.float32)


def frequency_filter(image: np.ndarray, response: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        channels = [image]
    else:
        channels = [image[..., c] for c in range(image.shape[-1])]

    filtered = []
    for channel in channels:
        spectrum = np.fft.fftshift(np.fft.fft2(channel))
        restored = np.fft.ifft2(np.fft.ifftshift(spectrum * response))
        filtered.append(np.real(restored))

    if image.ndim == 2:
        return np.clip(filtered[0], 0.0, 1.0)
    return np.clip(np.stack(filtered, axis=-1), 0.0, 1.0)


def magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(spectrum))
    return exposure.rescale_intensity(mag, out_range=(0.0, 1.0))


def edge_map(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    return feature.canny(filters.gaussian(gray, sigma=0.6), sigma=1.0).astype(float)


def histogram(image: np.ndarray, bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    gray = to_gray(image)
    values, edges = np.histogram(gray.ravel(), bins=bins, range=(0.0, 1.0), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, values


def reference_metrics(reference: np.ndarray, image: np.ndarray) -> MetricResult:
    reference = np.clip(reference, 0.0, 1.0)
    image = np.clip(image, 0.0, 1.0)
    return MetricResult(
        mse=float(metrics.mean_squared_error(reference, image)),
        psnr=float(metrics.peak_signal_noise_ratio(reference, image, data_range=1.0)),
        ssim=float(
            metrics.structural_similarity(
                reference,
                image,
                data_range=1.0,
                channel_axis=-1 if reference.ndim == 3 else None,
            )
        ),
    )
