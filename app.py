from __future__ import annotations

import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

from denoise_core import (
    add_noise,
    bilateral_filter,
    edge_map,
    float_to_uint8,
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
        "freq_filter": "Band-Stop",
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


def plot_histogram(images: dict[str, object]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=130)
    for label, image in images.items():
        x, y = histogram(image)
        ax.plot(x, y, linewidth=1.7, label=label)
    ax.set_xlabel("灰度")
    ax.set_ylabel("密度")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_response(response) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=130)
    im = ax.imshow(response, cmap="magma")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def run_methods(noisy, params, selected_methods):
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
            ],
            index=[
                "Ideal Low-Pass",
                "Gaussian Low-Pass",
                "Butterworth Low-Pass",
                "Chebyshev I",
                "Chebyshev II",
                "Elliptic",
                "Band-Stop",
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

    with st.expander("保边与高级滤波", expanded=True):
        bilateral_color = st.slider("双边滤波颜色 sigma", 0.01, 0.35, 0.10, 0.01)
        bilateral_spatial = st.slider("双边滤波空间 sigma", 1.0, 18.0, 5.0, 0.5)
        nlm_h = st.slider("NLM 强度系数", 0.3, 2.2, 0.9, 0.1)
        nlm_patch_size = st.slider("NLM patch 大小", 3, 9, 5, 2)
        nlm_patch_distance = st.slider("NLM 搜索半径", 3, 15, 7, 1)

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

outputs, response = run_methods(noisy, params, selected_methods)

top = st.columns([1, 1, 1])
with top[0]:
    st.markdown('<div class="method-label">原图 / 上传图</div>', unsafe_allow_html=True)
    st.image(source, use_container_width=True)
with top[1]:
    st.markdown('<div class="method-label">当前含噪输入</div>', unsafe_allow_html=True)
    st.image(noisy, use_container_width=True)
with top[2]:
    st.markdown('<div class="method-label">含噪图频谱</div>', unsafe_allow_html=True)
    st.image(magnitude_spectrum(noisy), clamp=True, use_container_width=True)

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

st.subheader("方法对比")
if not outputs:
    st.warning("请至少选择一种去噪方法。")
else:
    cols = st.columns(min(3, len(outputs)))
    for index, (name, image) in enumerate(outputs.items()):
        with cols[index % len(cols)]:
            st.markdown(f'<div class="method-label">{name}</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            image_download_button("下载结果", image, f"{name}.png")

if has_reference and outputs:
    st.subheader("参考指标")
    st.dataframe(metric_table(source, noisy, outputs), hide_index=True, use_container_width=True)

analysis_cols = st.columns([1, 1, 1])
with analysis_cols[0]:
    st.subheader("频域响应")
    st.pyplot(plot_response(response), clear_figure=True)
with analysis_cols[1]:
    st.subheader("灰度直方图")
    hist_images = {"含噪图": noisy}
    first_output = next(iter(outputs.items()), None)
    if first_output:
        hist_images[first_output[0]] = first_output[1]
    st.pyplot(plot_histogram(hist_images), clear_figure=True)
with analysis_cols[2]:
    st.subheader("边缘保留观察")
    edge_cols = st.columns(2)
    edge_cols[0].image(edge_map(noisy), caption="含噪图边缘", clamp=True, use_container_width=True)
    if first_output:
        edge_cols[1].image(edge_map(first_output[1]), caption=f"{first_output[0]} 边缘", clamp=True, use_container_width=True)

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
        row[0].image(case_noisy, caption="含噪图", use_container_width=True)
        for col, (name, image) in zip(row[1:], case_outputs.items()):
            col.image(image, caption=name, use_container_width=True)
