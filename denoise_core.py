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
        yy = np.arange(h)[:, None]
        wave = np.sin(2 * np.pi * periodic_frequency * yy / max(h, 1))
        wave = periodic_strength * wave[..., None]
        result = result + wave

    return np.clip(result, 0.0, 1.0)


def mean_filter(image: np.ndarray, size: int) -> np.ndarray:
    size = max(1, int(size))
    if image.ndim == 3:
        result = np.stack(
            [ndimage.uniform_filter(image[..., c], size=size) for c in range(image.shape[-1])],
            axis=-1,
        )
    else:
        result = ndimage.uniform_filter(image, size=size)
    return np.clip(result, 0.0, 1.0)


def median_filter(image: np.ndarray, size: int) -> np.ndarray:
    size = max(1, int(size))
    if image.ndim == 3:
        result = np.stack(
            [ndimage.median_filter(image[..., c], size=size) for c in range(image.shape[-1])],
            axis=-1,
        )
    else:
        result = ndimage.median_filter(image, size=size)
    return np.clip(result, 0.0, 1.0)


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
    result = restoration.denoise_nl_means(
        image,
        h=max(strength * sigma, 0.01),
        patch_size=int(patch_size),
        patch_distance=int(patch_distance),
        fast_mode=True,
        channel_axis=-1 if image.ndim == 3 else None,
    )
    return np.clip(result, 0.0, 1.0)


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


def frequency_axis_profiles(image: np.ndarray) -> dict[str, np.ndarray]:
    gray = to_gray(image)
    h, w = gray.shape
    spectrum_x = np.fft.fftshift(np.fft.fft(gray, axis=1), axes=1)
    spectrum_y = np.fft.fftshift(np.fft.fft(gray, axis=0), axes=0)
    profile_x = np.log1p(np.abs(spectrum_x[h // 2, :]))
    profile_y = np.log1p(np.abs(spectrum_y[:, w // 2]))

    def normalize(profile: np.ndarray) -> np.ndarray:
        profile = profile.astype(np.float64)
        span = np.max(profile) - np.min(profile)
        if span <= 1e-12:
            return np.zeros_like(profile)
        return (profile - np.min(profile)) / span

    return {
        "x_frequency": np.linspace(-0.5, 0.5, w) * w,
        "x_profile": normalize(profile_x),
        "y_frequency": np.linspace(-0.5, 0.5, h) * h,
        "y_profile": normalize(profile_y),
    }


def _positive_axis_peaks(profile: np.ndarray, peak_count: int, guard_percent: float) -> list[int]:
    profile = ndimage.gaussian_filter1d(np.asarray(profile, dtype=np.float64), sigma=1.2)
    n = profile.size
    center = n // 2
    guard = max(2, int(n * guard_percent / 100.0))
    work = profile.copy()
    work[: center + guard] = 0

    positive = work[center + guard :]
    if positive.size == 0 or np.max(positive) <= 0:
        return []

    baseline = np.median(positive)
    mad = np.median(np.abs(positive - baseline))
    prominence = max(0.04, 2.5 * mad)
    distance = max(2, int(n * 0.025))
    peaks, props = signal.find_peaks(work, distance=distance, prominence=prominence)

    if peaks.size == 0:
        candidate_count = min(max(peak_count * 3, peak_count), positive.size)
        peaks = np.argpartition(positive, -candidate_count)[-candidate_count:] + center + guard
        scores = work[peaks]
    else:
        scores = props.get("prominences", work[peaks])

    order = np.argsort(scores)[::-1]
    selected: list[int] = []
    for peak in peaks[order]:
        if peak <= center + guard:
            continue
        if any(abs(int(peak) - existing) < distance for existing in selected):
            continue
        selected.append(int(peak))
        if len(selected) >= peak_count:
            break
    return selected


def _detect_bright_line_angles(
    image: np.ndarray,
    line_count: int,
    line_threshold: float,
    low_frequency_guard_percent: float,
) -> list[float]:
    gray = to_gray(image)
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx = w / 2
    cy = h / 2
    x = xx - cx
    y = yy - cy
    radius = np.sqrt(x * x + y * y)
    max_radius = np.sqrt(cx * cx + cy * cy)

    magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    magnitude = exposure.rescale_intensity(magnitude, out_range=(0.0, 1.0))
    magnitude[radius < max_radius * low_frequency_guard_percent / 100.0] = 0

    angles = np.linspace(0, np.pi, 180, endpoint=False)
    sigma = max(min(h, w) * 0.006, 1.0)
    scores = []
    for angle in angles:
        distance = np.abs(x * np.sin(angle) - y * np.cos(angle))
        line_weight = np.exp(-(distance * distance) / (2 * sigma * sigma))
        score = float(np.sum(magnitude * line_weight) / max(np.sum(line_weight), 1e-12))
        scores.append(score)

    scores = ndimage.gaussian_filter1d(np.asarray(scores), sigma=1.0, mode="wrap")
    axis_guard = np.deg2rad(8)
    axis_like = (
        (angles < axis_guard)
        | (np.abs(angles - np.pi / 2) < axis_guard)
        | (angles > np.pi - axis_guard)
    )
    scores[axis_like] = 0
    max_score = float(np.max(scores))
    if max_score <= 0:
        return []
    threshold = max_score * float(np.clip(line_threshold, 0.0, 1.0))
    candidate_count = min(max(line_count * 8, line_count), scores.size)
    valid = np.argpartition(scores, -candidate_count)[-candidate_count:]
    valid = valid[scores[valid] >= threshold]
    if valid.size == 0:
        return []
    peak_scores = scores[valid]

    selected: list[float] = []
    for index, score in sorted(zip(valid, peak_scores), key=lambda item: item[1], reverse=True):
        angle = float(angles[int(index) % scores.size])
        if score <= 0:
            continue
        if any(abs(np.angle(np.exp(1j * 2 * (angle - existing)))) / 2 < np.deg2rad(10) for existing in selected):
            continue
        selected.append(angle)
        if len(selected) >= line_count:
            break
    return selected


def _line_score(image: np.ndarray, angle: float, low_frequency_guard_percent: float) -> float:
    gray = to_gray(image)
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx = w / 2
    cy = h / 2
    x = xx - cx
    y = yy - cy
    radius = np.sqrt(x * x + y * y)
    max_radius = np.sqrt(cx * cx + cy * cy)
    magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    magnitude = exposure.rescale_intensity(magnitude, out_range=(0.0, 1.0))
    magnitude[radius < max_radius * low_frequency_guard_percent / 100.0] = 0
    sigma = max(min(h, w) * 0.006, 1.0)
    distance = np.abs(x * np.sin(angle) - y * np.cos(angle))
    line_weight = np.exp(-(distance * distance) / (2 * sigma * sigma))
    return float(np.sum(magnitude * line_weight) / max(np.sum(line_weight), 1e-12))


def auto_band_stop_response(
    image: np.ndarray,
    peak_count: int = 4,
    notch_radius_percent: float = 1.5,
    depth: float = 0.95,
    line_threshold: float = 0.80,
    low_frequency_guard_percent: float = 8.0,
) -> np.ndarray:
    gray = to_gray(image)
    h, w = gray.shape
    profiles = frequency_axis_profiles(gray)
    x_peaks = _positive_axis_peaks(profiles["x_profile"], peak_count, low_frequency_guard_percent)
    y_peaks = _positive_axis_peaks(profiles["y_profile"], peak_count, low_frequency_guard_percent)
    x = np.arange(w)
    y = np.arange(h)
    center_x = w // 2
    center_y = h // 2
    response = np.ones_like(gray, dtype=np.float32)
    depth = float(np.clip(depth, 0.0, 1.0))
    sigma_x = max(w * notch_radius_percent / 100.0, 1.0)
    sigma_y = max(h * notch_radius_percent / 100.0, 1.0)

    for peak in x_peaks:
        mirror = (2 * center_x - peak) % w
        for column in (peak, mirror):
            notch = 1.0 - depth * np.exp(-((x - column) ** 2) / (2 * sigma_x * sigma_x))
            response *= notch[None, :].astype(np.float32)

    for peak in y_peaks:
        mirror = (2 * center_y - peak) % h
        for row in (peak, mirror):
            notch = 1.0 - depth * np.exp(-((y - row) ** 2) / (2 * sigma_y * sigma_y))
            response *= notch[:, None].astype(np.float32)

    yy, xx = np.mgrid[0:h, 0:w]
    centered_x = xx - w / 2
    centered_y = yy - h / 2
    radius = np.sqrt(centered_x * centered_x + centered_y * centered_y)
    max_radius = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
    high_frequency_weight = 1.0 - np.exp(
        -(radius * radius) / (2 * (max_radius * low_frequency_guard_percent / 100.0) ** 2)
    )

    notch = 1.0 - depth * np.exp(-((x - center_x) ** 2) / (2 * sigma_x * sigma_x))
    response *= (1.0 - (1.0 - notch[None, :]) * high_frequency_weight).astype(np.float32)
    notch = 1.0 - depth * np.exp(-((y - center_y) ** 2) / (2 * sigma_y * sigma_y))
    response *= (1.0 - (1.0 - notch[:, None]) * high_frequency_weight).astype(np.float32)

    line_sigma = max(min(h, w) * notch_radius_percent / 100.0, 1.0)
    line_angles = _detect_bright_line_angles(
        gray,
        max(1, min(2, peak_count // 2)),
        line_threshold,
        low_frequency_guard_percent,
    )
    diagonal_angles = [np.pi / 4, 3 * np.pi / 4]
    max_diagonal_score = max(_line_score(gray, angle, low_frequency_guard_percent) for angle in diagonal_angles)
    for angle in diagonal_angles:
        score = _line_score(gray, angle, low_frequency_guard_percent)
        if max_diagonal_score > 0 and score >= max_diagonal_score * float(np.clip(line_threshold, 0.0, 1.0)):
            if not any(abs(np.angle(np.exp(1j * 2 * (angle - existing)))) / 2 < np.deg2rad(10) for existing in line_angles):
                line_angles.append(angle)

    for angle in line_angles:
        distance = np.abs(centered_x * np.sin(angle) - centered_y * np.cos(angle))
        line_notch = 1.0 - depth * np.exp(-(distance * distance) / (2 * line_sigma * line_sigma)) * high_frequency_weight
        response *= line_notch.astype(np.float32)

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
