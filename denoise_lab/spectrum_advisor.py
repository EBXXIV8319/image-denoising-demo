from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage, signal

from .analysis import to_gray


@dataclass(frozen=True)
class SpectrumRecommendation:
    recommended_filter: str
    reason: str
    notch_points: list[dict[str, float]]
    radial_bands: list[dict[str, float]]
    peak_count: int
    radial_peak_count: int
    axial_energy_ratio: float
    high_frequency_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_spectrum(image: np.ndarray) -> SpectrumRecommendation:
    gray = to_gray(image)
    h, w = gray.shape
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(spectrum))
    normalized = (magnitude - magnitude.min()) / max(float(np.ptp(magnitude)), 1e-12)

    yy, xx = np.mgrid[0:h, 0:w]
    center_y = h / 2
    center_x = w / 2
    dy = yy - center_y
    dx = xx - center_x
    radius_px = np.sqrt(dx * dx + dy * dy)
    max_radius = max(float(radius_px.max()), 1.0)
    low_frequency_mask = radius_px <= 0.08 * min(h, w)
    candidate = normalized.copy()
    candidate[low_frequency_mask] = 0.0

    peak_threshold = max(float(np.quantile(candidate, 0.997)), 0.72)
    local_max = candidate == ndimage.maximum_filter(candidate, size=9)
    peak_mask = local_max & (candidate >= peak_threshold)
    peak_coords = np.argwhere(peak_mask)
    if peak_coords.size:
        peak_strength = candidate[peak_mask]
        order = np.argsort(peak_strength)[::-1][:10]
        peak_coords = peak_coords[order]

    notch_points = []
    seen: set[tuple[int, int]] = set()
    for y, x in peak_coords:
        u = int(round(x - center_x))
        v = int(round(y - center_y))
        if abs(u) < 2 and abs(v) < 2:
            continue
        key = tuple(sorted(((u, v), (-u, -v)))[0])
        if key in seen:
            continue
        seen.add(key)
        notch_points.append({"u": float(u), "v": float(v), "radius": 6.0, "order": 2.0, "depth": 0.95})
        if len(notch_points) >= 4:
            break

    bins = np.linspace(0.0, max_radius, 80)
    bin_index = np.clip(np.digitize(radius_px.ravel(), bins) - 1, 0, len(bins) - 2)
    radial_energy = np.bincount(bin_index, weights=candidate.ravel(), minlength=len(bins) - 1)
    radial_count = np.bincount(bin_index, minlength=len(bins) - 1)
    radial_profile = radial_energy / np.maximum(radial_count, 1)
    radial_profile[:5] = 0.0
    radial_threshold = float(np.mean(radial_profile) + 2.2 * np.std(radial_profile))
    radial_peak_indices, _ = signal.find_peaks(radial_profile, height=radial_threshold, distance=4)

    radial_bands = []
    for index in radial_peak_indices[:4]:
        center_percent = 100.0 * ((bins[index] + bins[index + 1]) / 2.0) / max_radius
        radial_bands.append({"center": float(center_percent), "width": 4.0, "order": 2.0, "depth": 0.9})

    axis_width = max(2, min(h, w) // 160)
    horizontal_axis = np.abs(yy - center_y) <= axis_width
    vertical_axis = np.abs(xx - center_x) <= axis_width
    axis_mask = (horizontal_axis | vertical_axis) & ~low_frequency_mask
    axial_energy_ratio = float(candidate[axis_mask].mean() / max(candidate[~low_frequency_mask].mean(), 1e-12))
    high_frequency_ratio = float(candidate[radius_px > 0.28 * max_radius].mean() / max(candidate[~low_frequency_mask].mean(), 1e-12))

    if notch_points:
        recommended_filter = "Butterworth Notch Reject"
        reason = "频谱中存在远离中心的离散亮点，适合使用成对陷波带阻滤波器。"
    elif radial_bands:
        recommended_filter = "Butterworth Radial Band-Stop"
        reason = "频谱能量在某些半径上形成环状峰值，适合使用径向 Band-Stop。"
    elif high_frequency_ratio > 1.15:
        recommended_filter = "Butterworth Low-Pass"
        reason = "高频区域整体能量偏高，优先尝试低通滤波抑制随机噪声。"
    else:
        recommended_filter = "Bilateral / NLM"
        reason = "频谱中没有明显周期峰值，建议优先使用保边空域方法。"

    return SpectrumRecommendation(
        recommended_filter=recommended_filter,
        reason=reason,
        notch_points=notch_points,
        radial_bands=radial_bands,
        peak_count=len(notch_points),
        radial_peak_count=len(radial_bands),
        axial_energy_ratio=axial_energy_ratio,
        high_frequency_ratio=high_frequency_ratio,
    )
