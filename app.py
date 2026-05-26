from __future__ import annotations

import io
import zipfile

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

from denoise_lab.advanced import bilateral_filter, nlm_filter
from denoise_lab.analysis import edge_map, histogram, magnitude_spectrum, reference_metrics
from denoise_lab.frequency import FREQUENCY_FILTERS, frequency_filter, frequency_response
from denoise_lab.image_io import float_to_uint8, pil_to_float_rgb
from denoise_lab.noise import NOISE_TYPES, add_noise
from denoise_lab.spatial import mean_filter, median_filter


st.set_page_config(page_title="图像去噪实时演示", layout="wide", initial_sidebar_state="expanded")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


METHODS = ["均值滤波", "中值滤波", "频域滤波", "双边滤波", "NLM"]


def image_png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(float_to_uint8(image)).save(buffer, format="PNG")
    return buffer.getvalue()


def dataframe_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def image_download_button(label: str, image, filename: str) -> None:
    st.download_button(label, image_png_bytes(image), filename, "image/png", use_container_width=True)


def figure_png_bytes(fig: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=160)
    return buffer.getvalue()


def figure_download_button(label: str, fig: plt.Figure, filename: str) -> None:
    st.download_button(label, figure_png_bytes(fig), filename, "image/png", use_container_width=True)


def safe_filename(name: str) -> str:
    invalid = '/\\:*?"<>|'
    return "".join("_" if char in invalid else char for char in name)


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


def plot_response(response) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=130)
    im = ax.imshow(response, cmap="magma")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def metric_table(reference, noisy, outputs) -> pd.DataFrame:
    rows = []
    for name, image in {"含噪图": noisy, **outputs}.items():
        result = reference_metrics(reference, image)
        rows.append({"对象": name, "MSE": round(result.mse, 6), "PSNR(dB)": round(result.psnr, 3), "SSIM": round(result.ssim, 4)})
    return pd.DataFrame(rows)


def run_methods(noisy, selected_methods, params):
    outputs = {}
    response = None
    if "均值滤波" in selected_methods:
        outputs["均值滤波"] = mean_filter(noisy, params["mean_size"])
    if "中值滤波" in selected_methods:
        outputs["中值滤波"] = median_filter(noisy, params["median_size"])
    if "频域滤波" in selected_methods:
        response = frequency_response(
            noisy.shape[:2],
            params["frequency_type"],
            params.get("cutoff", 24),
            params.get("order", 3),
            params.get("band_center", 22),
            params.get("band_width", 4),
            params.get("band_depth", 0.95),
        )
        outputs["频域滤波"] = frequency_filter(noisy, response)
    if "双边滤波" in selected_methods:
        outputs["双边滤波"] = bilateral_filter(noisy, params["bilateral_color"], params["bilateral_spatial"])
    if "NLM" in selected_methods:
        outputs["NLM"] = nlm_filter(noisy, params["nlm_h"], params["nlm_patch_size"], params["nlm_patch_distance"])
    return outputs, response


def build_zip(source, source_spectrum, noisy, noisy_spectrum, outputs, response_fig, metrics_df, histogram_df, filtered_spectrum):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("images/source_image.png", image_png_bytes(source))
        archive.writestr("images/source_spectrum.png", image_png_bytes(source_spectrum))
        archive.writestr("images/noisy_input.png", image_png_bytes(noisy))
        archive.writestr("images/noisy_spectrum.png", image_png_bytes(noisy_spectrum))
        for name, image in outputs.items():
            archive.writestr(f"filtered/{safe_filename(name)}.png", image_png_bytes(image))
        if filtered_spectrum is not None:
            archive.writestr("spectra/频域滤波_spectrum.png", image_png_bytes(filtered_spectrum))
        if response_fig is not None:
            archive.writestr("analysis/frequency_response.png", figure_png_bytes(response_fig))
        if metrics_df is not None:
            archive.writestr("tables/reference_metrics.csv", dataframe_csv_bytes(metrics_df))
        else:
            archive.writestr("tables/reference_metrics_unavailable.txt", "当前模式没有干净参考图，因此不计算参考指标。\n".encode("utf-8"))
        archive.writestr("tables/grayscale_histogram.csv", dataframe_csv_bytes(histogram_df))
        archive.writestr("edges/noisy_edges.png", image_png_bytes(edge_map(noisy)))
        for name, image in outputs.items():
            archive.writestr(f"edges/{safe_filename(name)}_edges.png", image_png_bytes(edge_map(image)))
    return buffer.getvalue()


st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.5rem; max-width: 1480px; }
    h1, h2, h3 { color: #172026; letter-spacing: 0; }
    .hero { border-bottom: 1px solid #d8e0e6; padding-bottom: 1rem; margin-bottom: 1rem; }
    .hero h1 { margin: 0 0 .25rem 0; font-size: 2rem; line-height: 1.18; }
    .hero p { margin: 0; color: #62727f; }
    .method-label { font-weight: 700; color: #172026; margin: .2rem 0 .4rem 0; }
    div[data-testid="stImage"] img { border-radius: 8px; border: 1px solid #d8e0e6; }
    div[data-testid="stMetric"] { background: #f7f9fb; border: 1px solid #d8e0e6; border-radius: 8px; padding: .7rem .85rem; }
    .small-note { color: #62727f; font-size: .9rem; }
    </style>
    <div class="hero">
      <h1>图像去噪实时演示</h1>
      <p>手动上传图像、添加噪声、配置滤波参数，并实时比较空域、频域、保边和 NLM 去噪效果。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("输入")
    uploaded = st.file_uploader("上传图像", type=["png", "jpg", "jpeg", "bmp", "webp"])
    st.caption("演示预设：手动参数")

    st.header("噪声")
    noise_type = st.selectbox("噪声场景", NOISE_TYPES, index=0)
    seed = st.number_input("随机种子", min_value=0, max_value=9999, value=7, step=1)
    gaussian_sigma = sp_amount = speckle_var = periodic_strength = 0.0
    periodic_frequency = 12
    if noise_type in {"高斯噪声", "混合噪声"}:
        gaussian_sigma = st.slider("高斯噪声标准差", 0.0, 0.25, 0.08, 0.01)
    if noise_type in {"椒盐噪声", "混合噪声"}:
        sp_amount = st.slider("椒盐噪声比例", 0.0, 0.35, 0.08, 0.01)
    if noise_type == "斑点噪声":
        speckle_var = st.slider("斑点噪声方差", 0.0, 0.20, 0.04, 0.01)
    if noise_type == "周期噪声":
        periodic_strength = st.slider("横向周期噪声强度", 0.0, 0.40, 0.22, 0.01)
        periodic_frequency = st.slider("横向周期噪声频率（周期数）", 2, 40, 14, 1)
    if noise_type == "无：上传图像已含噪":
        st.caption("当前模式不会向上传图像额外添加噪声。")

    st.header("方法")
    selected_methods = st.multiselect("参与对比的方法", METHODS, default=METHODS)
    params = {}

    if "均值滤波" in selected_methods:
        with st.expander("均值滤波配置", expanded=True):
            params["mean_size"] = st.slider("均值滤波核大小（像素）", 1, 15, 5, 2)
    if "中值滤波" in selected_methods:
        with st.expander("中值滤波配置", expanded=True):
            params["median_size"] = st.slider("中值滤波核大小（像素）", 1, 15, 5, 2)
    if "频域滤波" in selected_methods:
        with st.expander("频域滤波配置", expanded=True):
            params["frequency_type"] = st.selectbox("频域滤波器类型", FREQUENCY_FILTERS, index=2)
            if params["frequency_type"] in {"Gaussian Low-Pass", "Butterworth Low-Pass"}:
                params["cutoff"] = st.slider("低通截止半径（%）", 3, 90, 24, 1)
            if params["frequency_type"] == "Butterworth Low-Pass":
                params["order"] = st.slider("巴特沃斯阶数（阶）", 1, 8, 3, 1)
            if params["frequency_type"] == "Band-Stop":
                params["band_center"] = st.slider("带阻中心半径（%）", 3, 90, 22, 1)
                params["band_width"] = st.slider("带阻宽度（%）", 1, 30, 4, 1)
                params["band_depth"] = st.slider("带阻抑制强度（比例）", 0.0, 1.0, 0.95, 0.05)
    if "双边滤波" in selected_methods:
        with st.expander("双边滤波配置", expanded=True):
            params["bilateral_color"] = st.slider("颜色 sigma（灰度差）", 0.01, 0.35, 0.10, 0.01)
            params["bilateral_spatial"] = st.slider("空间 sigma（像素）", 1.0, 18.0, 5.0, 0.5)
    if "NLM" in selected_methods:
        with st.expander("NLM 滤波配置", expanded=True):
            params["nlm_h"] = st.slider("NLM 强度系数（倍）", 0.3, 2.2, 0.9, 0.1)
            params["nlm_patch_size"] = st.slider("NLM patch 大小（像素）", 3, 9, 5, 2)
            params["nlm_patch_distance"] = st.slider("NLM 搜索半径（像素）", 3, 15, 7, 1)

if uploaded is None:
    st.info("请先上传一张图片。")
    st.stop()

source = pil_to_float_rgb(Image.open(uploaded))
has_reference = noise_type != "无：上传图像已含噪"
noisy = add_noise(source, noise_type, gaussian_sigma, sp_amount, speckle_var, periodic_strength, periodic_frequency, seed)
outputs, response = run_methods(noisy, selected_methods, params)

source_spectrum = magnitude_spectrum(source)
noisy_spectrum = magnitude_spectrum(noisy)
filtered_spectrum = magnitude_spectrum(outputs["频域滤波"]) if "频域滤波" in outputs else None
hist_images = {"含噪图": noisy}
first_output = next(iter(outputs.items()), None)
if first_output:
    hist_images[first_output[0]] = first_output[1]
zip_histogram_df = histogram_dataframe({"含噪图": noisy, **outputs})
metrics_df = metric_table(source, noisy, outputs) if has_reference and outputs else None
response_fig = plot_response(response) if response is not None else None

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
    cols = st.columns(3)
    cols[0].metric("含噪图 MSE", f"{noisy_metrics.mse:.5f}")
    cols[1].metric("含噪图 PSNR", f"{noisy_metrics.psnr:.2f} dB")
    cols[2].metric("含噪图 SSIM", f"{noisy_metrics.ssim:.4f}")
else:
    st.markdown('<p class="small-note">当前模式没有干净参考图，因此不显示 PSNR / SSIM / MSE。</p>', unsafe_allow_html=True)

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

if metrics_df is not None:
    st.subheader("参考指标")
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

analysis_cols = st.columns([1, 1, 1])
with analysis_cols[0]:
    st.subheader("频域响应")
    if response_fig is not None:
        st.pyplot(response_fig, clear_figure=False)
        figure_download_button("下载频域响应", response_fig, "frequency_response.png")
    else:
        st.caption("未选择频域滤波。")
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

if filtered_spectrum is not None:
    with st.expander("频域滤波后频谱图", expanded=True):
        st.image(filtered_spectrum, caption="频域滤波频谱", clamp=True, use_container_width=True)
        image_download_button("下载频域滤波频谱", filtered_spectrum, "频域滤波_spectrum.png")

st.subheader("一键下载")
st.download_button(
    "下载当前全部结果 ZIP",
    build_zip(source, source_spectrum, noisy, noisy_spectrum, outputs, response_fig, metrics_df, zip_histogram_df, filtered_spectrum),
    "image_denoising_results.zip",
    "application/zip",
    use_container_width=True,
)

if response_fig is not None:
    plt.close(response_fig)
