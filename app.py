from __future__ import annotations

import io
import zipfile

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from denoise_core import (
    add_noise,
    auto_band_stop_response,
    bilateral_filter,
    edge_map,
    float_to_uint8,
    frequency_axis_profiles,
    frequency_filter,
    frequency_response,
    histogram,
    magnitude_spectrum,
    mean_filter,
    median_filter,
    nlm_filter,
    pil_to_float_rgb,
    reference_metrics,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


st.set_page_config(
    page_title="图像去噪实时演示",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
      --ink: #172026;
      --muted: #62727f;
      --line: #d8e0e6;
      --panel: #f7f9fb;
      --accent: #0f766e;
      --warm: #b45309;
    }
    .main .block-container { padding-top: 1.5rem; max-width: 1480px; }
    h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
    .hero {
      border-bottom: 1px solid var(--line);
      padding: 0.25rem 0 1rem 0;
      margin-bottom: 1.1rem;
    }
    .hero h1 {
      margin: 0 0 0.25rem 0;
      font-size: 2rem;
      line-height: 1.18;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      font-size: 0.98rem;
    }
    .method-label {
      font-weight: 700;
      color: var(--ink);
      margin: 0.2rem 0 0.4rem 0;
    }
    div[data-testid="stMetric"] {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0.7rem 0.85rem;
    }
    div[data-testid="stImage"] img {
      border-radius: 8px;
      border: 1px solid var(--line);
    }
    .small-note {
      color: var(--muted);
      font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


PRESETS = {
    "手动参数": {
        "noise": "高斯噪声",
        "gaussian_sigma": 0.08,
        "sp_amount": 0.08,
        "speckle_var": 0.04,
        "periodic_strength": 0.18,
        "periodic_frequency": 12,
        "freq_filter": "Gaussian Low-Pass",
        "cutoff": 22,
        "order": 3,
        "ripple": 1.0,
        "attenuation": 40,
        "band_center": 34,
        "band_width": 5,
        "band_depth": 0.9,
        "methods": ["均值滤波", "中值滤波", "频域滤波", "双边滤波", "NLM"],
    },
    "高斯噪声演示": {
        "noise": "高斯噪声",
        "gaussian_sigma": 0.10,
        "sp_amount": 0.06,
        "speckle_var": 0.04,
        "periodic_strength": 0.18,
        "periodic_frequency": 12,
        "freq_filter": "Butterworth Low-Pass",
        "cutoff": 20,
        "order": 3,
        "ripple": 1.0,
        "attenuation": 42,
        "band_center": 34,
        "band_width": 5,
        "band_depth": 0.9,
        "methods": ["均值滤波", "频域滤波", "双边滤波", "NLM"],
    },
    "椒盐噪声演示": {
        "noise": "椒盐噪声",
        "gaussian_sigma": 0.06,
        "sp_amount": 0.12,
        "speckle_var": 0.04,
        "periodic_strength": 0.18,
        "periodic_frequency": 12,
        "freq_filter": "Chebyshev I",
        "cutoff": 24,
        "order": 4,
        "ripple": 1.0,
        "attenuation": 45,
        "band_center": 34,
        "band_width": 5,
        "band_depth": 0.9,
        "methods": ["均值滤波", "中值滤波", "频域滤波", "双边滤波"],
    },
    "混合噪声挑战": {
        "noise": "混合噪声",
        "gaussian_sigma": 0.08,
        "sp_amount": 0.10,
        "speckle_var": 0.04,
        "periodic_strength": 0.18,
        "periodic_frequency": 12,
        "freq_filter": "Elliptic",
        "cutoff": 22,
        "order": 4,
        "ripple": 0.8,
        "attenuation": 55,
        "band_center": 34,
        "band_width": 5,
        "band_depth": 0.9,
        "methods": ["中值滤波", "频域滤波", "双边滤波", "NLM"],
    },
    "周期噪声与带阻": {
        "noise": "周期噪声",
        "gaussian_sigma": 0.06,
        "sp_amount": 0.06,
        "speckle_var": 0.04,
        "periodic_strength": 0.22,
        "periodic_frequency": 14,
        "freq_filter": "Auto Band-Stop",
        "cutoff": 22,
        "order": 3,
        "ripple": 1.0,
        "attenuation": 45,
        "band_center": 22,
        "band_width": 4,
        "band_depth": 0.95,
        "methods": ["均值滤波", "频域滤波", "双边滤波", "NLM"],
    },
}

METHOD_MAP = {
    "均值滤波": "mean",
    "中值滤波": "median",
    "频域滤波": "frequency",
    "双边滤波": "bilateral",
    "NLM": "nlm",
}


def image_download_button(label: str, image, filename: str) -> None:
    buffer = io.BytesIO()
    Image.fromarray(float_to_uint8(image)).save(buffer, format="PNG")
    st.download_button(label, buffer.getvalue(), filename, "image/png", use_container_width=True)


def image_png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(float_to_uint8(image)).save(buffer, format="PNG")
    return buffer.getvalue()


def dataframe_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def figure_download_button(label: str, fig: plt.Figure, filename: str) -> None:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=160)
    st.download_button(label, buffer.getvalue(), filename, "image/png", use_container_width=True)


def figure_png_bytes(fig: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=160)
    return buffer.getvalue()


def safe_filename(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )


def histogram_dataframe(images: dict[str, object]) -> pd.DataFrame:
    rows = []
    for label, image in images.items():
        x, y = histogram(image)
        rows.extend({"灰度": xi, "密度": yi, "图像": label} for xi, yi in zip(x, y))
    return pd.DataFrame(rows)


def histogram_chart(images: dict[str, object]) -> alt.Chart:
    data = histogram_dataframe(images)
    return (
        alt.Chart(data)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("灰度:Q", title="灰度", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("密度:Q", title="密度"),
            color=alt.Color("图像:N", title="图像"),
            tooltip=[
                alt.Tooltip("图像:N", title="图像"),
                alt.Tooltip("灰度:Q", title="灰度", format=".2f"),
                alt.Tooltip("密度:Q", title="密度", format=".2f"),
            ],
        )
        .properties(height=260)
        .interactive()
    )


def frequency_axis_dataframe(image) -> pd.DataFrame:
    profiles = frequency_axis_profiles(image)
    rows = []
    rows.extend(
        {"轴向": "x轴频率", "频率索引": freq, "强度": value}
        for freq, value in zip(profiles["x_frequency"], profiles["x_profile"])
    )
    rows.extend(
        {"轴向": "y轴频率", "频率索引": freq, "强度": value}
        for freq, value in zip(profiles["y_frequency"], profiles["y_profile"])
    )
    return pd.DataFrame(rows)


def frequency_axis_chart(dataframe: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(dataframe)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("频率索引:Q", title="频率索引"),
            y=alt.Y("强度:Q", title="频谱剖面强度"),
            color=alt.Color("轴向:N", title="剖面"),
            tooltip=[
                alt.Tooltip("轴向:N", title="剖面"),
                alt.Tooltip("频率索引:Q", title="频率索引", format=".1f"),
                alt.Tooltip("强度:Q", title="强度", format=".3f"),
            ],
        )
        .properties(height=260)
        .interactive()
    )


def plot_response(response) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=130)
    im = ax.imshow(response, cmap="magma")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def run_methods(noisy, params, selected_methods):
    if params["freq_filter"] == "Auto Band-Stop":
        response = auto_band_stop_response(
            noisy,
            peak_count=params["auto_peak_count"],
            notch_radius_percent=params["auto_notch_radius"],
            depth=params["band_depth"],
            line_threshold=params["auto_line_threshold"],
        )
    else:
        response = frequency_response(
            noisy.shape[:2],
            params["freq_filter"],
            params["cutoff"],
            params["order"],
            params["ripple"],
            params["attenuation"],
            params["band_center"],
            params["band_width"],
            params["band_depth"],
        )
    outputs = {}
    for method in selected_methods:
        key = METHOD_MAP[method]
        if key == "mean":
            outputs[method] = mean_filter(noisy, params["mean_size"])
        elif key == "median":
            outputs[method] = median_filter(noisy, params["median_size"])
        elif key == "frequency":
            outputs[method] = frequency_filter(noisy, response)
        elif key == "bilateral":
            outputs[method] = bilateral_filter(
                noisy,
                params["bilateral_color"],
                params["bilateral_spatial"],
            )
        elif key == "nlm":
            outputs[method] = nlm_filter(
                noisy,
                params["nlm_h"],
                params["nlm_patch_size"],
                params["nlm_patch_distance"],
            )
        outputs[method] = outputs[method].clip(0.0, 1.0)
    return outputs, response


def metric_table(reference, noisy, outputs):
    rows = []
    for name, image in {"含噪图": noisy, **outputs}.items():
        m = reference_metrics(reference, image)
        rows.append(
            {
                "对象": name,
                "MSE": round(m.mse, 6),
                "PSNR(dB)": round(m.psnr, 3),
                "SSIM": round(m.ssim, 4),
            }
        )
    return pd.DataFrame(rows)


def downsample_for_tuning(image, max_side: int = 180):
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / longest
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    resized = Image.fromarray(float_to_uint8(image)).resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def score_candidate(reference, candidate) -> float:
    result = reference_metrics(reference, candidate)
    if result.ssim == result.ssim:
        return result.ssim
    return -result.mse


def best_by_reference(reference, candidates):
    best_score = float("-inf")
    best_params = None
    best_image = None
    for candidate_params, candidate_image in candidates:
        score = score_candidate(reference, candidate_image)
        if score > best_score:
            best_score = score
            best_params = candidate_params
            best_image = candidate_image
    return best_params, best_image, best_score


def auto_tune_parameters(reference, noisy, params, selected_methods, noise_type: str, has_reference: bool):
    tuned = dict(params)
    rows = []

    if not has_reference:
        if noise_type == "周期噪声":
            tuned.update({"freq_filter": "Auto Band-Stop", "auto_peak_count": 4, "auto_notch_radius": 1.5, "auto_line_threshold": 0.80, "band_depth": 0.95})
            rows.append({"方法": "频域滤波", "策略": "无参考启发式", "参数": "Auto Band-Stop, 峰值=4, 半径=1.5%, 斜线阈值=0.80, 抑制=0.95"})
        elif noise_type == "椒盐噪声":
            tuned.update({"median_size": 3 if params["band_depth"] < 0.9 else 5})
            rows.append({"方法": "中值滤波", "策略": "无参考启发式", "参数": f"核大小={tuned['median_size']}"})
        elif noise_type == "高斯噪声":
            tuned.update({"mean_size": 3, "freq_filter": "Gaussian Low-Pass", "cutoff": 28, "bilateral_color": 0.08, "bilateral_spatial": 4.0, "nlm_h": 0.8})
            rows.append({"方法": "通用", "策略": "无参考启发式", "参数": "轻度平滑，优先保留边缘"})
        elif noise_type == "混合噪声":
            tuned.update({"median_size": 3, "bilateral_color": 0.10, "bilateral_spatial": 4.0, "nlm_h": 1.0})
            rows.append({"方法": "通用", "策略": "无参考启发式", "参数": "中值先抗脉冲，保边/NLM 中等强度"})
        else:
            rows.append({"方法": "通用", "策略": "无参考启发式", "参数": "保持当前参数"})
        return tuned, pd.DataFrame(rows)

    ref_small = downsample_for_tuning(reference)
    noisy_small = downsample_for_tuning(noisy)

    if "均值滤波" in selected_methods:
        candidates = [({"mean_size": size}, mean_filter(noisy_small, size)) for size in [3, 5, 7, 9, 11]]
        best, _, score = best_by_reference(ref_small, candidates)
        tuned.update(best)
        rows.append({"方法": "均值滤波", "策略": "SSIM 网格搜索", "参数": f"核大小={best['mean_size']}, SSIM={score:.4f}"})

    if "中值滤波" in selected_methods:
        candidates = [({"median_size": size}, median_filter(noisy_small, size)) for size in [3, 5, 7, 9]]
        best, _, score = best_by_reference(ref_small, candidates)
        tuned.update(best)
        rows.append({"方法": "中值滤波", "策略": "SSIM 网格搜索", "参数": f"核大小={best['median_size']}, SSIM={score:.4f}"})

    if "频域滤波" in selected_methods:
        candidates = []
        if noise_type == "周期噪声" or params["freq_filter"] == "Auto Band-Stop":
            for count in [2, 4, 6]:
                for radius in [1.0, 1.5, 2.5]:
                    for depth in [0.8, 0.95]:
                        for line_threshold in [0.70, 0.85]:
                            response = auto_band_stop_response(noisy_small, count, radius, depth, line_threshold)
                            candidates.append(
                                (
                                    {
                                        "freq_filter": "Auto Band-Stop",
                                        "auto_peak_count": count,
                                        "auto_notch_radius": radius,
                                        "auto_line_threshold": line_threshold,
                                        "band_depth": depth,
                                    },
                                    frequency_filter(noisy_small, response),
                                )
                            )
        else:
            filter_family = params["freq_filter"]
            for cutoff_value in [14, 20, 26, 34, 42]:
                for order_value in [2, 4, 6]:
                    response = frequency_response(
                        noisy_small.shape[:2],
                        filter_family,
                        cutoff_value,
                        order_value,
                        params["ripple"],
                        params["attenuation"],
                        params["band_center"],
                        params["band_width"],
                        params["band_depth"],
                    )
                    candidates.append(
                        (
                            {"freq_filter": filter_family, "cutoff": cutoff_value, "order": order_value},
                            frequency_filter(noisy_small, response),
                        )
                    )
        best, _, score = best_by_reference(ref_small, candidates)
        tuned.update(best)
        rows.append({"方法": "频域滤波", "策略": "SSIM 网格搜索", "参数": f"{best}, SSIM={score:.4f}"})

    if "双边滤波" in selected_methods:
        candidates = []
        for sigma_color in [0.04, 0.08, 0.12, 0.18, 0.25]:
            for sigma_spatial in [2.0, 4.0, 7.0, 10.0]:
                candidates.append(
                    (
                        {"bilateral_color": sigma_color, "bilateral_spatial": sigma_spatial},
                        bilateral_filter(noisy_small, sigma_color, sigma_spatial),
                    )
                )
        best, _, score = best_by_reference(ref_small, candidates)
        tuned.update(best)
        rows.append({"方法": "双边滤波", "策略": "SSIM 网格搜索", "参数": f"颜色sigma={best['bilateral_color']}, 空间sigma={best['bilateral_spatial']}, SSIM={score:.4f}"})

    if "NLM" in selected_methods:
        candidates = []
        for h_value in [0.6, 0.9, 1.2]:
            for patch_size_value in [3, 5]:
                for patch_distance_value in [3, 5]:
                    candidates.append(
                        (
                            {"nlm_h": h_value, "nlm_patch_size": patch_size_value, "nlm_patch_distance": patch_distance_value},
                            nlm_filter(noisy_small, h_value, patch_size_value, patch_distance_value),
                        )
                    )
        best, _, score = best_by_reference(ref_small, candidates)
        tuned.update(best)
        rows.append({"方法": "NLM", "策略": "SSIM 网格搜索", "参数": f"h={best['nlm_h']}, patch={best['nlm_patch_size']}, 搜索={best['nlm_patch_distance']}, SSIM={score:.4f}"})

    return tuned, pd.DataFrame(rows)


def build_download_zip(
    source,
    source_spectrum,
    noisy,
    noisy_spectrum,
    filtered_spectra,
    outputs,
    metrics_df: pd.DataFrame | None,
    response_fig: plt.Figure,
    histogram_df: pd.DataFrame,
    axis_profile_df: pd.DataFrame,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("images/source_image.png", image_png_bytes(source))
        archive.writestr("images/source_spectrum.png", image_png_bytes(source_spectrum))
        archive.writestr("images/noisy_input.png", image_png_bytes(noisy))
        archive.writestr("images/noisy_spectrum.png", image_png_bytes(noisy_spectrum))
        for name, spectrum in filtered_spectra.items():
            archive.writestr(f"spectra/{safe_filename(name)}_spectrum.png", image_png_bytes(spectrum))

        for name, image in outputs.items():
            archive.writestr(f"filtered/{safe_filename(name)}.png", image_png_bytes(image))

        if metrics_df is not None:
            archive.writestr("tables/reference_metrics.csv", dataframe_csv_bytes(metrics_df))
        else:
            archive.writestr(
                "tables/reference_metrics_unavailable.txt",
                "当前模式没有干净参考图，因此不计算 MSE、PSNR、SSIM。\n".encode("utf-8"),
            )

        archive.writestr("analysis/frequency_response.png", figure_png_bytes(response_fig))
        archive.writestr("tables/grayscale_histogram.csv", dataframe_csv_bytes(histogram_df))
        archive.writestr("tables/frequency_axis_profiles.csv", dataframe_csv_bytes(axis_profile_df))
        archive.writestr("edges/noisy_edges.png", image_png_bytes(edge_map(noisy)))
        for name, image in outputs.items():
            archive.writestr(f"edges/{safe_filename(name)}_edges.png", image_png_bytes(edge_map(image)))

    return buffer.getvalue()


st.markdown(
    """
    <div class="hero">
      <h1>图像去噪实时演示</h1>
      <p>上传图片后现场加噪、滤波、显示频谱与指标，比较空域、频域、保边和非局部均值方法。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("输入")
    uploaded = st.file_uploader("上传图像", type=["png", "jpg", "jpeg", "bmp", "webp"])
    preset_name = st.selectbox("演示预设", list(PRESETS.keys()), index=0)
    preset = PRESETS[preset_name]

    st.header("噪声")
    noise_type = st.selectbox(
        "噪声场景",
        ["高斯噪声", "椒盐噪声", "斑点噪声", "混合噪声", "周期噪声", "无：上传图像已含噪"],
        index=["高斯噪声", "椒盐噪声", "斑点噪声", "混合噪声", "周期噪声", "无：上传图像已含噪"].index(preset["noise"]),
        key=f"noise-{preset_name}",
    )
    seed = st.number_input("随机种子", min_value=0, max_value=9999, value=7, step=1)

    gaussian_sigma = preset["gaussian_sigma"]
    sp_amount = preset["sp_amount"]
    speckle_var = preset["speckle_var"]
    periodic_strength = preset["periodic_strength"]
    periodic_frequency = preset["periodic_frequency"]

    if noise_type in {"高斯噪声", "混合噪声"}:
        gaussian_sigma = st.slider("高斯噪声标准差", 0.0, 0.25, gaussian_sigma, 0.01)
    if noise_type in {"椒盐噪声", "混合噪声"}:
        sp_amount = st.slider("椒盐噪声比例", 0.0, 0.35, sp_amount, 0.01)
    if noise_type == "斑点噪声":
        speckle_var = st.slider("斑点噪声方差", 0.0, 0.20, speckle_var, 0.01)
    if noise_type == "周期噪声":
        periodic_strength = st.slider("周期噪声强度", 0.0, 0.40, periodic_strength, 0.01)
        periodic_frequency = st.slider("周期噪声频率", 2, 40, periodic_frequency, 1)
    if noise_type == "无：上传图像已含噪":
        st.caption("当前模式不会向上传图像额外添加噪声。")

    st.header("方法")
    selected_methods = st.multiselect(
        "参与对比的方法",
        list(METHOD_MAP.keys()),
        default=preset["methods"],
        key=f"methods-{preset_name}",
    )

    with st.expander("空域滤波", expanded=True):
        mean_size = st.slider("均值滤波核大小", 1, 15, 5, 2)
        median_size = st.slider("中值滤波核大小", 1, 15, 5, 2)

    with st.expander("频域滤波", expanded=True):
        freq_filter = st.selectbox(
            "频域滤波器",
            [
                "Ideal Low-Pass",
                "Gaussian Low-Pass",
                "Butterworth Low-Pass",
                "Chebyshev I",
                "Chebyshev II",
                "Elliptic",
                "Band-Stop",
                "Auto Band-Stop",
            ],
            index=[
                "Ideal Low-Pass",
                "Gaussian Low-Pass",
                "Butterworth Low-Pass",
                "Chebyshev I",
                "Chebyshev II",
                "Elliptic",
                "Band-Stop",
                "Auto Band-Stop",
            ].index(preset["freq_filter"]),
            key=f"freq-{preset_name}",
        )
        cutoff = st.slider("低通截止半径 (%)", 3, 90, preset["cutoff"], 1)
        order = st.slider("阶数", 1, 8, preset["order"], 1)
        ripple = st.slider("通带纹波 Rp (dB)", 0.1, 5.0, preset["ripple"], 0.1)
        attenuation = st.slider("阻带衰减 Rs (dB)", 10, 90, preset["attenuation"], 1)
        band_center = st.slider("带阻中心半径 (%)", 3, 90, preset["band_center"], 1)
        band_width = st.slider("带阻宽度 (%)", 1, 30, preset["band_width"], 1)
        band_depth = st.slider("带阻抑制强度", 0.0, 1.0, preset["band_depth"], 0.05)
        auto_peak_count = st.slider("自动带阻峰值数量", 2, 12, 4, 2)
        auto_notch_radius = st.slider("自动带阻抑制半径 (%)", 0.5, 5.0, 1.5, 0.5)
        auto_line_threshold = st.slider(
            "斜向亮线检测阈值",
            0.10,
            0.95,
            0.80,
            0.05,
            help="阈值越高，只抑制越明显的斜向亮线；阈值越低，会捕获更多弱斜线但更容易过度滤波。",
        )

    with st.expander("保边与高级滤波", expanded=True):
        bilateral_color = st.slider("双边滤波颜色 sigma", 0.01, 0.35, 0.10, 0.01)
        bilateral_spatial = st.slider("双边滤波空间 sigma", 1.0, 18.0, 5.0, 0.5)
        nlm_h = st.slider("NLM 强度系数", 0.3, 2.2, 0.9, 0.1)
        nlm_patch_size = st.slider("NLM patch 大小", 3, 9, 5, 2)
        nlm_patch_distance = st.slider("NLM 搜索半径", 3, 15, 7, 1)

    auto_tune = st.toggle("自动优化滤波参数", value=False)
    show_demo_matrix = st.toggle("生成一键演示矩阵", value=False)

params = {
    "mean_size": mean_size,
    "median_size": median_size,
    "freq_filter": freq_filter,
    "cutoff": cutoff,
    "order": order,
    "ripple": ripple,
    "attenuation": attenuation,
    "band_center": band_center,
    "band_width": band_width,
    "band_depth": band_depth,
    "auto_peak_count": auto_peak_count,
    "auto_notch_radius": auto_notch_radius,
    "auto_line_threshold": auto_line_threshold,
    "bilateral_color": bilateral_color,
    "bilateral_spatial": bilateral_spatial,
    "nlm_h": nlm_h,
    "nlm_patch_size": nlm_patch_size,
    "nlm_patch_distance": nlm_patch_distance,
}

if uploaded is None:
    st.info("请先上传一张图片。程序不会读取预处理好的示例图，所有结果都会基于当前上传图像实时计算。")
    st.stop()

source = pil_to_float_rgb(Image.open(uploaded))
has_reference = noise_type != "无：上传图像已含噪"
noisy = add_noise(
    source,
    noise_type,
    gaussian_sigma,
    sp_amount,
    speckle_var,
    periodic_strength,
    periodic_frequency,
    seed,
)

tuning_report = None
if auto_tune:
    with st.spinner("正在自动优化参数..."):
        params, tuning_report = auto_tune_parameters(source, noisy, params, selected_methods, noise_type, has_reference)

outputs, response = run_methods(noisy, params, selected_methods)
source_spectrum = magnitude_spectrum(source)
noisy_spectrum = magnitude_spectrum(noisy)
filtered_spectra = {"频域滤波": magnitude_spectrum(outputs["频域滤波"])} if "频域滤波" in outputs else {}
metrics_df = metric_table(source, noisy, outputs) if has_reference and outputs else None
hist_images = {"含噪图": noisy}
first_output = next(iter(outputs.items()), None)
if first_output:
    hist_images[first_output[0]] = first_output[1]
zip_histogram_df = histogram_dataframe({"含噪图": noisy, **outputs})
axis_profile_df = frequency_axis_dataframe(noisy)
response_fig = plot_response(response)

top = st.columns([1, 1, 1, 1])
with top[0]:
    st.markdown('<div class="method-label">原图 / 上传图</div>', unsafe_allow_html=True)
    st.image(source, clamp=True, use_container_width=True)
    image_download_button("下载原图 / 上传图", source, "source_image.png")
with top[1]:
    st.markdown('<div class="method-label">原图频谱</div>', unsafe_allow_html=True)
    st.image(source_spectrum, clamp=True, use_container_width=True)
    image_download_button("下载原图频谱", source_spectrum, "source_spectrum.png")
with top[2]:
    st.markdown('<div class="method-label">当前含噪输入</div>', unsafe_allow_html=True)
    st.image(noisy, clamp=True, use_container_width=True)
    image_download_button("下载含噪输入", noisy, "noisy_input.png")
with top[3]:
    st.markdown('<div class="method-label">含噪图频谱</div>', unsafe_allow_html=True)
    st.image(noisy_spectrum, clamp=True, use_container_width=True)
    image_download_button("下载含噪图频谱", noisy_spectrum, "noisy_spectrum.png")

if has_reference:
    noisy_metrics = reference_metrics(source, noisy)
    mcols = st.columns(3)
    mcols[0].metric("含噪图 MSE", f"{noisy_metrics.mse:.5f}")
    mcols[1].metric("含噪图 PSNR", f"{noisy_metrics.psnr:.2f} dB")
    mcols[2].metric("含噪图 SSIM", f"{noisy_metrics.ssim:.4f}")
else:
    st.markdown(
        '<p class="small-note">当前模式没有干净参考图，因此不显示 PSNR / SSIM / MSE；请使用频谱、直方图和边缘图做无参考比较。</p>',
        unsafe_allow_html=True,
    )

if tuning_report is not None and not tuning_report.empty:
    with st.expander("自动调参结果", expanded=True):
        st.dataframe(tuning_report, hide_index=True, use_container_width=True)

st.subheader("方法对比")
if not outputs:
    st.warning("请至少选择一种去噪方法。")
else:
    cols = st.columns(min(3, len(outputs)))
    for index, (name, image) in enumerate(outputs.items()):
        with cols[index % len(cols)]:
            st.markdown(f'<div class="method-label">{name}</div>', unsafe_allow_html=True)
            st.image(image, clamp=True, use_container_width=True)
            image_download_button("下载结果", image, f"{name}.png")

if has_reference and outputs:
    st.subheader("参考指标")
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

analysis_cols = st.columns([1, 1, 1])
with analysis_cols[0]:
    st.subheader("频域响应")
    st.pyplot(response_fig, clear_figure=False)
    figure_download_button("下载频域响应", response_fig, "frequency_response.png")
with analysis_cols[1]:
    st.subheader("灰度直方图")
    st.altair_chart(histogram_chart(hist_images), use_container_width=True)
with analysis_cols[2]:
    st.subheader("边缘保留观察")
    edge_cols = st.columns(2)
    noisy_edges = edge_map(noisy)
    edge_cols[0].image(noisy_edges, caption="含噪图边缘", clamp=True, use_container_width=True)
    with edge_cols[0]:
        image_download_button("下载含噪图边缘", noisy_edges, "noisy_edges.png")
    if first_output:
        output_edges = edge_map(first_output[1])
        edge_cols[1].image(output_edges, caption=f"{first_output[0]} 边缘", clamp=True, use_container_width=True)
        with edge_cols[1]:
            image_download_button("下载去噪结果边缘", output_edges, f"{first_output[0]}_edges.png")

if filtered_spectra:
    with st.expander("频域滤波后频谱图", expanded="频域滤波" in filtered_spectra):
        spectrum_cols = st.columns(min(3, len(filtered_spectra)))
        for index, (name, spectrum) in enumerate(filtered_spectra.items()):
            with spectrum_cols[index % len(spectrum_cols)]:
                st.image(spectrum, caption=f"{name} 频谱", clamp=True, use_container_width=True)
                image_download_button("下载频谱", spectrum, f"{safe_filename(name)}_spectrum.png")

with st.expander("频域 x / y 轴异常频率剖面", expanded=params["freq_filter"] == "Auto Band-Stop"):
    st.altair_chart(frequency_axis_chart(axis_profile_df), use_container_width=True)
    st.download_button(
        "下载频域 x / y 轴剖面 CSV",
        dataframe_csv_bytes(axis_profile_df),
        "frequency_axis_profiles.csv",
        "text/csv",
        use_container_width=True,
    )

st.subheader("一键下载")
st.download_button(
    "下载当前全部结果 ZIP",
    build_download_zip(source, source_spectrum, noisy, noisy_spectrum, filtered_spectra, outputs, metrics_df, response_fig, zip_histogram_df, axis_profile_df),
    "image_denoising_results.zip",
    "application/zip",
    use_container_width=True,
)
plt.close(response_fig)

if show_demo_matrix:
    st.subheader("一键演示矩阵")
    st.markdown(
        '<p class="small-note">以下三组结果仍然基于当前上传图像现场加噪和计算，用于答辩时快速串联不同噪声与方法。</p>',
        unsafe_allow_html=True,
    )
    matrix_cases = [
        ("高斯噪声", PRESETS["高斯噪声演示"], ["均值滤波", "频域滤波", "双边滤波", "NLM"]),
        ("椒盐噪声", PRESETS["椒盐噪声演示"], ["均值滤波", "中值滤波", "双边滤波"]),
        ("混合噪声", PRESETS["混合噪声挑战"], ["中值滤波", "频域滤波", "NLM"]),
    ]
    for idx, (label, case, methods) in enumerate(matrix_cases):
        st.markdown(f"**{label}**")
        case_noisy = add_noise(
            source,
            case["noise"],
            case["gaussian_sigma"],
            case["sp_amount"],
            case["speckle_var"],
            case["periodic_strength"],
            case["periodic_frequency"],
            seed + idx + 31,
        )
        case_params = {
            **params,
            "freq_filter": case["freq_filter"],
            "cutoff": case["cutoff"],
            "order": case["order"],
            "ripple": case["ripple"],
            "attenuation": case["attenuation"],
            "band_center": case["band_center"],
            "band_width": case["band_width"],
            "band_depth": case["band_depth"],
        }
        case_outputs, _ = run_methods(case_noisy, case_params, methods)
        row = st.columns(1 + len(case_outputs))
        row[0].image(case_noisy, caption="含噪图", clamp=True, use_container_width=True)
        for col, (name, image) in zip(row[1:], case_outputs.items()):
            col.image(image, caption=name, clamp=True, use_container_width=True)
