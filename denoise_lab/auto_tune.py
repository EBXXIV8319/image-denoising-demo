from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erf, sqrt
from typing import Callable

import numpy as np
from skimage import transform

from .advanced import bilateral_filter, nlm_filter
from .analysis import reference_metrics
from .frequency import frequency_filter, frequency_response
from .spatial import mean_filter, median_filter
from .spectrum_advisor import SpectrumRecommendation, analyze_spectrum


@dataclass(frozen=True)
class MethodTuneResult:
    method: str
    params: dict
    mse: float
    psnr: float
    ssim: float
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AutoTuneResult:
    params: dict
    method_results: list[MethodTuneResult]
    spectrum_recommendation: dict

    def to_dict(self) -> dict:
        return {
            "params": self.params,
            "method_results": [result.to_dict() for result in self.method_results],
            "spectrum_recommendation": self.spectrum_recommendation,
        }


def tune_parameters(
    reference: np.ndarray,
    noisy: np.ndarray,
    selected_methods: list[str],
    iterations: int,
    seed: int,
) -> AutoTuneResult:
    reference, noisy = _prepare_tuning_images(reference, noisy)
    rng = np.random.default_rng(seed)
    recommendation = analyze_spectrum(noisy)
    tuned_params: dict = {}
    method_results: list[MethodTuneResult] = []

    if "均值滤波" in selected_methods:
        result = _tune_mean(reference, noisy, max(iterations, 4), rng)
        tuned_params.update(result.params)
        method_results.append(result)
    if "中值滤波" in selected_methods:
        result = _tune_median(reference, noisy, max(iterations, 4), rng)
        tuned_params.update(result.params)
        method_results.append(result)
    if "频域滤波" in selected_methods:
        result = _tune_frequency(reference, noisy, recommendation, max(iterations, 6), rng)
        tuned_params.update(result.params)
        method_results.append(result)
    if "双边滤波" in selected_methods:
        result = _tune_bilateral(reference, noisy, max(iterations, 6), rng)
        tuned_params.update(result.params)
        method_results.append(result)
    if "NLM" in selected_methods:
        result = _tune_nlm(reference, noisy, max(iterations, 6), rng)
        tuned_params.update(result.params)
        method_results.append(result)

    return AutoTuneResult(tuned_params, method_results, recommendation.to_dict())


def _prepare_tuning_images(reference: np.ndarray, noisy: np.ndarray, max_side: int = 160) -> tuple[np.ndarray, np.ndarray]:
    h, w = reference.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return reference, noisy
    output_shape = (max(8, int(round(h * scale))), max(8, int(round(w * scale))))
    if reference.ndim == 3:
        output_shape = (*output_shape, reference.shape[-1])
    small_reference = transform.resize(reference, output_shape, preserve_range=True, anti_aliasing=True)
    small_noisy = transform.resize(noisy, output_shape, preserve_range=True, anti_aliasing=True)
    return np.clip(small_reference, 0.0, 1.0), np.clip(small_noisy, 0.0, 1.0)


def _score(reference: np.ndarray, output: np.ndarray) -> tuple[float, float, float, float]:
    metrics = reference_metrics(reference, output)
    psnr = 60.0 if np.isinf(metrics.psnr) else metrics.psnr
    mse_penalty = min(metrics.mse * 25.0, 2.0)
    score = float(metrics.ssim + psnr / 60.0 - mse_penalty)
    return score, metrics.mse, metrics.psnr, metrics.ssim


def _odd(value: float, low: int = 1, high: int = 15) -> int:
    candidate = int(round(value))
    candidate = max(low, min(high, candidate))
    if candidate % 2 == 0:
        candidate = candidate + 1 if candidate < high else candidate - 1
    return max(low, min(high, candidate))


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / sqrt(2.0 * np.pi)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-9)
    improvement = mu - best
    z = improvement / sigma
    return improvement * _normal_cdf(z) + sigma * _normal_pdf(z)


def _gp_predict(x_train: np.ndarray, y_train: np.ndarray, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length_scale = np.full(x_train.shape[1], 0.35)
    diff = (x_train[:, None, :] - x_train[None, :, :]) / length_scale
    kernel = np.exp(-0.5 * np.sum(diff * diff, axis=2)) + np.eye(len(x_train)) * 1e-6
    query_diff = (x_query[:, None, :] - x_train[None, :, :]) / length_scale
    query_kernel = np.exp(-0.5 * np.sum(query_diff * query_diff, axis=2))
    inv_kernel = np.linalg.pinv(kernel)
    centered = y_train - np.mean(y_train)
    mu = np.mean(y_train) + query_kernel @ inv_kernel @ centered
    variance = np.maximum(1.0 - np.sum((query_kernel @ inv_kernel) * query_kernel, axis=1), 1e-9)
    return mu, np.sqrt(variance)


def _bayesian_search(
    evaluate: Callable[[np.ndarray], tuple[float, dict]],
    dimensions: int,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[dict, float]:
    x_train: list[np.ndarray] = []
    y_train: list[float] = []
    payloads: list[dict] = []
    initial = min(max(3, dimensions + 2), iterations)

    for _ in range(initial):
        x = rng.random(dimensions)
        score, payload = evaluate(x)
        x_train.append(x)
        y_train.append(score)
        payloads.append(payload)

    while len(x_train) < iterations:
        candidates = rng.random((160, dimensions))
        mu, sigma = _gp_predict(np.vstack(x_train), np.array(y_train, dtype=float), candidates)
        next_x = candidates[int(np.argmax(_expected_improvement(mu, sigma, max(y_train))))]
        score, payload = evaluate(next_x)
        x_train.append(next_x)
        y_train.append(score)
        payloads.append(payload)

    best_index = int(np.argmax(y_train))
    return payloads[best_index], float(y_train[best_index])


def _method_result(method: str, params: dict, reference: np.ndarray, output: np.ndarray, score: float) -> MethodTuneResult:
    _, mse, psnr, ssim = _score(reference, output)
    return MethodTuneResult(method=method, params=params, mse=mse, psnr=psnr, ssim=ssim, score=score)


def _tune_mean(reference: np.ndarray, noisy: np.ndarray, iterations: int, rng: np.random.Generator) -> MethodTuneResult:
    def evaluate(x: np.ndarray) -> tuple[float, dict]:
        size = _odd(1 + x[0] * 14)
        output = mean_filter(noisy, size)
        score, *_ = _score(reference, output)
        return score, {"mean_size": size, "_output": output}

    payload, score = _bayesian_search(evaluate, 1, iterations, rng)
    output = payload.pop("_output")
    return _method_result("均值滤波", payload, reference, output, score)


def _tune_median(reference: np.ndarray, noisy: np.ndarray, iterations: int, rng: np.random.Generator) -> MethodTuneResult:
    def evaluate(x: np.ndarray) -> tuple[float, dict]:
        size = _odd(1 + x[0] * 14)
        output = median_filter(noisy, size)
        score, *_ = _score(reference, output)
        return score, {"median_size": size, "_output": output}

    payload, score = _bayesian_search(evaluate, 1, iterations, rng)
    output = payload.pop("_output")
    return _method_result("中值滤波", payload, reference, output, score)


def _tune_bilateral(reference: np.ndarray, noisy: np.ndarray, iterations: int, rng: np.random.Generator) -> MethodTuneResult:
    def evaluate(x: np.ndarray) -> tuple[float, dict]:
        sigma_color = 0.02 + x[0] * 0.28
        sigma_spatial = 1.0 + x[1] * 13.0
        output = bilateral_filter(noisy, sigma_color, sigma_spatial)
        score, *_ = _score(reference, output)
        return score, {"bilateral_color": round(float(sigma_color), 3), "bilateral_spatial": round(float(sigma_spatial), 2), "_output": output}

    payload, score = _bayesian_search(evaluate, 2, iterations, rng)
    output = payload.pop("_output")
    return _method_result("双边滤波", payload, reference, output, score)


def _tune_nlm(reference: np.ndarray, noisy: np.ndarray, iterations: int, rng: np.random.Generator) -> MethodTuneResult:
    def evaluate(x: np.ndarray) -> tuple[float, dict]:
        h = 0.35 + x[0] * 1.65
        patch_size = _odd(3 + x[1] * 6, 3, 9)
        patch_distance = int(round(3 + x[2] * 12))
        output = nlm_filter(noisy, h, patch_size, patch_distance)
        score, *_ = _score(reference, output)
        return score, {"nlm_h": round(float(h), 2), "nlm_patch_size": patch_size, "nlm_patch_distance": patch_distance, "_output": output}

    payload, score = _bayesian_search(evaluate, 3, iterations, rng)
    output = payload.pop("_output")
    return _method_result("NLM", payload, reference, output, score)


def _tune_frequency(
    reference: np.ndarray,
    noisy: np.ndarray,
    recommendation: SpectrumRecommendation,
    iterations: int,
    rng: np.random.Generator,
) -> MethodTuneResult:
    candidate_results: list[MethodTuneResult] = []
    for tuner in (_tune_gaussian_lowpass, _tune_butterworth_lowpass, _tune_radial_bandstop):
        candidate_results.append(tuner(reference, noisy, max(4, iterations // 2), rng))
    if recommendation.notch_points:
        candidate_results.append(_tune_notch(reference, noisy, recommendation, max(4, iterations // 2), rng))
    return max(candidate_results, key=lambda result: result.score)


def _tune_gaussian_lowpass(reference: np.ndarray, noisy: np.ndarray, iterations: int, rng: np.random.Generator) -> MethodTuneResult:
    def evaluate(x: np.ndarray) -> tuple[float, dict]:
        cutoff = int(round(5 + x[0] * 55))
        response = frequency_response(noisy.shape[:2], "Gaussian Low-Pass", cutoff_percent=cutoff)
        output = frequency_filter(noisy, response)
        score, *_ = _score(reference, output)
        return score, {"frequency_type": "Gaussian Low-Pass", "cutoff": cutoff, "_output": output}

    payload, score = _bayesian_search(evaluate, 1, iterations, rng)
    output = payload.pop("_output")
    return _method_result("频域滤波", payload, reference, output, score)


def _tune_butterworth_lowpass(reference: np.ndarray, noisy: np.ndarray, iterations: int, rng: np.random.Generator) -> MethodTuneResult:
    def evaluate(x: np.ndarray) -> tuple[float, dict]:
        cutoff = int(round(5 + x[0] * 55))
        order = int(round(1 + x[1] * 7))
        response = frequency_response(noisy.shape[:2], "Butterworth Low-Pass", cutoff_percent=cutoff, order=order)
        output = frequency_filter(noisy, response)
        score, *_ = _score(reference, output)
        return score, {"frequency_type": "Butterworth Low-Pass", "cutoff": cutoff, "order": order, "_output": output}

    payload, score = _bayesian_search(evaluate, 2, iterations, rng)
    output = payload.pop("_output")
    return _method_result("频域滤波", payload, reference, output, score)


def _tune_radial_bandstop(reference: np.ndarray, noisy: np.ndarray, iterations: int, rng: np.random.Generator) -> MethodTuneResult:
    def evaluate(x: np.ndarray) -> tuple[float, dict]:
        center = 6.0 + x[0] * 58.0
        width = 1.0 + x[1] * 13.0
        order = int(round(1 + x[2] * 5))
        depth = 0.55 + x[3] * 0.45
        bands = [{"center": round(float(center), 2), "width": round(float(width), 2), "order": order, "depth": round(float(depth), 2)}]
        response = frequency_response(noisy.shape[:2], "Butterworth Radial Band-Stop", radial_bands=bands)
        output = frequency_filter(noisy, response)
        score, *_ = _score(reference, output)
        return score, {"frequency_type": "Butterworth Radial Band-Stop", "radial_bands": bands, "_output": output}

    payload, score = _bayesian_search(evaluate, 4, iterations, rng)
    output = payload.pop("_output")
    return _method_result("频域滤波", payload, reference, output, score)


def _tune_notch(
    reference: np.ndarray,
    noisy: np.ndarray,
    recommendation: SpectrumRecommendation,
    iterations: int,
    rng: np.random.Generator,
) -> MethodTuneResult:
    base_notches = recommendation.notch_points[:3]

    def evaluate(x: np.ndarray) -> tuple[float, dict]:
        radius = 2.0 + x[0] * 14.0
        depth = 0.65 + x[1] * 0.35
        order = int(round(1 + x[2] * 5))
        notches = [
            {"u": notch["u"], "v": notch["v"], "radius": round(float(radius), 2), "order": order, "depth": round(float(depth), 2)}
            for notch in base_notches
        ]
        response = frequency_response(noisy.shape[:2], "Butterworth Notch Reject", notches=notches)
        output = frequency_filter(noisy, response)
        score, *_ = _score(reference, output)
        return score, {"frequency_type": "Butterworth Notch Reject", "notches": notches, "_output": output}

    payload, score = _bayesian_search(evaluate, 3, iterations, rng)
    output = payload.pop("_output")
    return _method_result("频域滤波", payload, reference, output, score)
