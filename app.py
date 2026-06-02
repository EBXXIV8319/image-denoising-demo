from __future__ import annotations

import io
import json
import zipfile

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

from denoise_lab.advanced import bilateral_filter, nlm_filter
from denoise_lab.analysis import edge_map, histogram, magnitude_spectrum, reference_metrics
from denoise_lab.auto_tune import tune_parameters
from denoise_lab.frequency import FREQUENCY_FILTERS, frequency_filter, frequency_response
from denoise_lab.image_io import float_to_uint8, pil_to_float_rgb
from denoise_lab.noise import NOISE_TYPES, add_noise
from denoise_lab.sharpening import unsharp_mask
from denoise_lab.spectrum_advisor import analyze_spectrum
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


def json_bytes(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


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


def int_param(
    label: str,
    min_value: int,
    max_value: int,
    value: int,
    step: int,
    key: str,
    odd: bool = False,
    disabled: bool = False,
) -> int:
    widget_key = f"{key}_auto_locked" if disabled else key
    raw_value = int(
        st.number_input(label, min_value=min_value, max_value=max_value, value=value, step=step, key=widget_key, disabled=disabled)
    )
    if odd and raw_value % 2 == 0:
        adjusted = raw_value + 1 if raw_value < max_value else raw_value - 1
        st.caption(f"已按算法要求使用奇数：{adjusted}")
        return adjusted
    return raw_value


def float_param(
    label: str,
    min_value: float,
    max_value: float,
    value: float,
    step: float,
    key: str,
    fmt: str = "%.3f",
    disabled: bool = False,
) -> float:
    widget_key = f"{key}_auto_locked" if disabled else key
    return float(
        st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            value=value,
            step=step,
            format=fmt,
            key=widget_key,
            disabled=disabled,
        )
    )


def apply_tuned_params_to_state(tuned_params: dict) -> None:
    direct_keys = {
        "mean_size",
        "median_size",
        "cutoff",
        "bilateral_color",
        "bilateral_spatial",
        "nlm_h",
        "nlm_patch_size",
        "nlm_patch_distance",
    }
    for key in direct_keys:
        if key in tuned_params:
            st.session_state[key] = tuned_params[key]

    if "frequency_type" in tuned_params:
        st.session_state["frequency_type"] = tuned_params["frequency_type"]
    if "order" in tuned_params:
        st.session_state["butterworth_order"] = tuned_params["order"]

    radial_bands = tuned_params.get("radial_bands") or []
    if radial_bands:
        st.session_state["radial_band_count"] = len(radial_bands)
        for index, band in enumerate(radial_bands):
            st.session_state[f"radial_center_{index}"] = band.get("center", 22.0)
            st.session_state[f"radial_width_{index}"] = band.get("width", 4.0)
            st.session_state[f"radial_order_{index}"] = band.get("order", 2)
            st.session_state[f"radial_depth_{index}"] = band.get("depth", 0.95)

    notches = tuned_params.get("notches") or []
    if notches:
        st.session_state["notch_count"] = len(notches)
        for index, notch in enumerate(notches):
            st.session_state[f"notch_u_{index}"] = int(round(float(notch.get("u", 0))))
            st.session_state[f"notch_v_{index}"] = int(round(float(notch.get("v", 81 * (index + 1)))))
            st.session_state[f"notch_radius_{index}"] = notch.get("radius", 8.0)
            st.session_state[f"notch_order_{index}"] = notch.get("order", 2)
            st.session_state[f"notch_depth_{index}"] = notch.get("depth", 0.95)


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


def auto_tune_search_table(auto_tune_result: dict | None) -> pd.DataFrame:
    if not auto_tune_result:
        return pd.DataFrame()
    rows = []
    for item in auto_tune_result.get("method_results", []):
        row = {
            "方法": item["method"],
            "调优参数": json.dumps(item["params"], ensure_ascii=False),
            "目标分数": round(item["score"], 4),
        }
        if item.get("niqe_like") is None:
            row.update(
                {
                    "MSE": round(item["mse"], 6),
                    "PSNR(dB)": round(item["psnr"], 3),
                    "SSIM": round(item["ssim"], 4),
                }
            )
        else:
            row["NIQE-like"] = round(item["niqe_like"], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def auto_tune_applied_table(auto_tune_result: dict | None, reference, outputs) -> pd.DataFrame:
    if not auto_tune_result or reference is None:
        return pd.DataFrame()
    rows = []
    for item in auto_tune_result.get("method_results", []):
        method = item["method"]
        if method not in outputs:
            continue
        result = reference_metrics(reference, outputs[method])
        rows.append(
            {
                "方法": method,
                "调优参数": json.dumps(item["params"], ensure_ascii=False),
                "MSE": round(result.mse, 6),
                "PSNR(dB)": round(result.psnr, 3),
                "SSIM": round(result.ssim, 4),
                "搜索目标分数": round(item["score"], 4),
            }
        )
    return pd.DataFrame(rows)


def run_methods(noisy, selected_methods, params):
    outputs = {}
    response = None
    filter_input = params.get("filter_input", noisy)
    if "均值滤波" in selected_methods:
        outputs["均值滤波"] = mean_filter(filter_input, params["mean_size"])
    if "中值滤波" in selected_methods:
        outputs["中值滤波"] = median_filter(noisy, params["median_size"])
    if "频域滤波" in selected_methods:
        response = frequency_response(
            shape=filter_input.shape[:2],
            filter_type=params["frequency_type"],
            cutoff_percent=params.get("cutoff", 24),
            order=params.get("order", 3),
            band_center_percent=params.get("band_center", 22),
            band_width_percent=params.get("band_width", 4),
            band_depth=params.get("band_depth", 0.95),
            notch_u=params.get("notch_u", 0),
            notch_v=params.get("notch_v", 81),
            notch_radius=params.get("notch_radius", 8.0),
            notches=params.get("notches"),
            radial_bands=params.get("radial_bands"),
        )
        outputs["频域滤波"] = frequency_filter(filter_input, response)
    if "双边滤波" in selected_methods:
        outputs["双边滤波"] = bilateral_filter(filter_input, params["bilateral_color"], params["bilateral_spatial"])
    if "NLM" in selected_methods:
        outputs["NLM"] = nlm_filter(filter_input, params["nlm_h"], params["nlm_patch_size"], params["nlm_patch_distance"])
    return outputs, response


def build_zip(
    source,
    source_spectrum,
    noisy,
    noisy_spectrum,
    outputs,
    response_fig,
    metrics_df,
    histogram_df,
    filtered_spectrum,
    parameter_export,
):
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
        archive.writestr("config/current_parameters.json", json_bytes(parameter_export))
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
    gaussian_sigma = sp_amount = periodic_strength = 0.0
    periodic_frequency = 12
    if noise_type in {"高斯噪声", "混合噪声"}:
        gaussian_sigma = float_param("高斯噪声标准差", 0.0, 0.25, 0.08, 0.01, "gaussian_sigma", "%.2f")
    if noise_type in {"椒盐噪声", "混合噪声"}:
        sp_amount = float_param("椒盐噪声比例", 0.0, 0.35, 0.08, 0.01, "sp_amount", "%.2f")
    if noise_type == "周期噪声":
        periodic_strength = float_param("横向周期噪声强度", 0.0, 0.40, 0.22, 0.01, "periodic_strength", "%.2f")
        periodic_frequency = int_param("横向周期噪声频率（周期数）", 2, 40, 14, 1, "periodic_frequency")
    if noise_type == "无：上传图像已含噪":
        st.caption("当前模式不会向上传图像额外添加噪声。")

    source = pil_to_float_rgb(Image.open(uploaded)) if uploaded is not None else None
    has_reference = noise_type != "无：上传图像已含噪"
    noisy = (
        add_noise(source, noise_type, gaussian_sigma, sp_amount, periodic_strength, periodic_frequency, seed)
        if source is not None
        else None
    )
    spectrum_recommendation = analyze_spectrum(noisy).to_dict() if noisy is not None else None

    st.header("方法")
    selected_methods = st.multiselect("参与对比的方法", METHODS, default=METHODS)
    params = {}
    use_mixed_preprocess = False
    mixed_preprocess_size = 3
    if noise_type == "混合噪声":
        with st.expander("混合噪声预处理配置", expanded=True):
            use_mixed_preprocess = st.checkbox("先使用中值滤波预处理后再交给其他滤波器", value=True)
            mixed_preprocess_size = int_param("预处理中值滤波核大小（像素）", 1, 15, 3, 2, "mixed_preprocess_size", odd=True)

    st.header("自动调参")
    auto_allowed = source is not None and bool(selected_methods)
    auto_iterations = int_param("BO 迭代次数", 4, 24, 10, 1, "auto_iterations", disabled=not auto_allowed)
    auto_enabled = st.checkbox("启用自动调参", value=False, disabled=not auto_allowed)
    auto_tune_result = None
    auto_locked = False
    if source is None:
        st.caption("请先上传原图后再启用自动调参。")
    elif not selected_methods:
        st.caption("请至少选择一种参与对比的方法。")
    elif not has_reference:
        st.caption("当前上传图像已含噪，将使用 NIQE-like 无参考指标自动调参，参数不会锁定。")
    else:
        st.caption("当前由程序加噪，将使用 MSE、PSNR、SSIM 自动调参，参数会锁定。")

    if auto_enabled and auto_allowed:
        signature = json.dumps(
            {
                "uploaded_file": uploaded.name,
                "noise_type": noise_type,
                "seed": int(seed),
                "gaussian_sigma": gaussian_sigma,
                "sp_amount": sp_amount,
                "periodic_strength": periodic_strength,
                "periodic_frequency": periodic_frequency,
                "selected_methods": selected_methods,
                "mixed_preprocess": use_mixed_preprocess,
                "mixed_preprocess_size": mixed_preprocess_size,
                "iterations": auto_iterations,
                "metric_mode": "full_reference" if has_reference else "niqe_like_no_reference",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if st.session_state.get("auto_tune_signature") != signature:
            with st.spinner("正在使用贝叶斯优化进行自动调参..."):
                tuning_input = median_filter(noisy, mixed_preprocess_size) if noise_type == "混合噪声" and use_mixed_preprocess else noisy
                reference_for_tuning = source if has_reference else None
                result = tune_parameters(reference_for_tuning, noisy, selected_methods, auto_iterations, int(seed), filter_input=tuning_input)
            st.session_state["auto_tune_signature"] = signature
            st.session_state["auto_tune_result"] = result.to_dict()
            st.session_state["auto_tune_state_applied"] = False
        auto_tune_result = st.session_state.get("auto_tune_result")
        auto_locked = bool(auto_tune_result and has_reference)
        if auto_tune_result:
            if not st.session_state.get("auto_tune_state_applied", False):
                apply_tuned_params_to_state(auto_tune_result["params"])
                st.session_state["auto_tune_state_applied"] = True
            if auto_locked:
                st.success("已应用 BO 自动调参结果，参数框已锁定。")
            else:
                st.success("已应用 NIQE-like 自动调参建议，参数框仍可继续手动修改。")
            st.dataframe(auto_tune_search_table(auto_tune_result), hide_index=True, use_container_width=True)
    tuned_params = auto_tune_result["params"] if auto_tune_result else {}

    if "均值滤波" in selected_methods:
        with st.expander("均值滤波配置", expanded=True):
            params["mean_size"] = int_param(
                "均值滤波核大小（像素）",
                1,
                15,
                int(tuned_params.get("mean_size", 5)),
                2,
                "mean_size",
                odd=True,
                disabled=auto_locked and "mean_size" in tuned_params,
            )
    if "中值滤波" in selected_methods:
        with st.expander("中值滤波配置", expanded=True):
            params["median_size"] = int_param(
                "中值滤波核大小（像素）",
                1,
                15,
                int(tuned_params.get("median_size", 5)),
                2,
                "median_size",
                odd=True,
                disabled=auto_locked and "median_size" in tuned_params,
            )
    if "频域滤波" in selected_methods:
        with st.expander("频域滤波配置", expanded=True):
            tuned_frequency_type = tuned_params.get("frequency_type", "Butterworth Notch Reject")
            frequency_index = FREQUENCY_FILTERS.index(tuned_frequency_type) if tuned_frequency_type in FREQUENCY_FILTERS else 3
            params["frequency_type"] = st.selectbox(
                "频域滤波器类型",
                FREQUENCY_FILTERS,
                index=frequency_index,
                disabled=auto_locked and "frequency_type" in tuned_params,
                key="frequency_type_auto_locked" if auto_locked and "frequency_type" in tuned_params else "frequency_type",
            )
            if params["frequency_type"] in {"Gaussian Low-Pass", "Butterworth Low-Pass"}:
                params["cutoff"] = int_param(
                    "低通截止半径（%）",
                    3,
                    90,
                    int(tuned_params.get("cutoff", 24)),
                    1,
                    "cutoff",
                    disabled=auto_locked and "cutoff" in tuned_params,
                )
            if params["frequency_type"] == "Butterworth Low-Pass":
                params["order"] = int_param(
                    "巴特沃斯阶数（阶）",
                    1,
                    8,
                    int(tuned_params.get("order", 3)),
                    1,
                    "butterworth_order",
                    disabled=auto_locked and "order" in tuned_params,
                )
            if params["frequency_type"] == "Butterworth Radial Band-Stop":
                tuned_radial_bands = tuned_params.get("radial_bands", [])
                band_count = int_param(
                    "径向 Band-Stop 组数（组）",
                    1,
                    6,
                    max(1, len(tuned_radial_bands)) if tuned_radial_bands else 1,
                    1,
                    "radial_band_count",
                    disabled=auto_locked and bool(tuned_radial_bands),
                )
                radial_bands = []
                for index in range(band_count):
                    tuned_band = tuned_radial_bands[index] if index < len(tuned_radial_bands) else {}
                    st.markdown(f'<div class="method-label">径向 Band-Stop 组 {index + 1}</div>', unsafe_allow_html=True)
                    col_center, col_width = st.columns(2)
                    with col_center:
                        band_center = float_param(
                            f"组 {index + 1} 中心半径（%）",
                            1.0,
                            98.0,
                            float(tuned_band.get("center", 22.0 + 8.0 * index)),
                            1.0,
                            f"radial_center_{index}",
                            "%.1f",
                            disabled=auto_locked and bool(tuned_radial_bands),
                        )
                    with col_width:
                        band_width = float_param(
                            f"组 {index + 1} 带宽（%）",
                            0.5,
                            80.0,
                            float(tuned_band.get("width", 4.0)),
                            0.5,
                            f"radial_width_{index}",
                            "%.1f",
                            disabled=auto_locked and bool(tuned_radial_bands),
                        )
                    radial_bands.append(
                        {
                            "center": band_center,
                            "width": band_width,
                            "order": int_param(
                                f"组 {index + 1} 巴特沃斯阶数（阶）",
                                1,
                                8,
                                int(tuned_band.get("order", 2)),
                                1,
                                f"radial_order_{index}",
                                disabled=auto_locked and bool(tuned_radial_bands),
                            ),
                            "depth": float_param(
                                f"组 {index + 1} 抑制强度（比例）",
                                0.0,
                                1.0,
                                float(tuned_band.get("depth", 0.95)),
                                0.05,
                                f"radial_depth_{index}",
                                "%.2f",
                                disabled=auto_locked and bool(tuned_radial_bands),
                            ),
                        }
                    )
                params["radial_bands"] = radial_bands
            if params["frequency_type"] == "Butterworth Notch Reject":
                tuned_notches = tuned_params.get("notches", [])
                notch_count = int_param(
                    "陷波点组数（组）",
                    1,
                    6,
                    max(1, len(tuned_notches)) if tuned_notches else 1,
                    1,
                    "notch_count",
                    disabled=auto_locked and bool(tuned_notches),
                )
                notches = []
                for index in range(notch_count):
                    tuned_notch = tuned_notches[index] if index < len(tuned_notches) else {}
                    default_v = 81 * (index + 1)
                    st.markdown(f'<div class="method-label">陷波点组 {index + 1}</div>', unsafe_allow_html=True)
                    col_u, col_v = st.columns(2)
                    with col_u:
                        notch_u = int_param(
                            f"组 {index + 1} Δu",
                            -512,
                            512,
                            int(round(float(tuned_notch.get("u", 0)))),
                            1,
                            f"notch_u_{index}",
                            disabled=auto_locked and bool(tuned_notches),
                        )
                    with col_v:
                        notch_v = int_param(
                            f"组 {index + 1} Δv",
                            -512,
                            512,
                            int(round(float(tuned_notch.get("v", default_v)))),
                            1,
                            f"notch_v_{index}",
                            disabled=auto_locked and bool(tuned_notches),
                        )
                    notches.append(
                        {
                            "u": notch_u,
                            "v": notch_v,
                            "radius": float_param(
                                f"组 {index + 1} 陷波半径 D0（像素）",
                                1.0,
                                80.0,
                                float(tuned_notch.get("radius", 8.0)),
                                1.0,
                                f"notch_radius_{index}",
                                "%.1f",
                                disabled=auto_locked and bool(tuned_notches),
                            ),
                            "order": int_param(
                                f"组 {index + 1} 巴特沃斯阶数（阶）",
                                1,
                                8,
                                int(tuned_notch.get("order", 2)),
                                1,
                                f"notch_order_{index}",
                                disabled=auto_locked and bool(tuned_notches),
                            ),
                            "depth": float_param(
                                f"组 {index + 1} 抑制强度（比例）",
                                0.0,
                                1.0,
                                float(tuned_notch.get("depth", 0.95)),
                                0.05,
                                f"notch_depth_{index}",
                                "%.2f",
                                disabled=auto_locked and bool(tuned_notches),
                            ),
                        }
                    )
                params["notches"] = notches
    if "双边滤波" in selected_methods:
        with st.expander("双边滤波配置", expanded=True):
            params["bilateral_color"] = float_param(
                "颜色 sigma（灰度差）",
                0.01,
                0.35,
                float(tuned_params.get("bilateral_color", 0.10)),
                0.01,
                "bilateral_color",
                "%.2f",
                disabled=auto_locked and "bilateral_color" in tuned_params,
            )
            params["bilateral_spatial"] = float_param(
                "空间 sigma（像素）",
                1.0,
                18.0,
                float(tuned_params.get("bilateral_spatial", 5.0)),
                0.5,
                "bilateral_spatial",
                "%.1f",
                disabled=auto_locked and "bilateral_spatial" in tuned_params,
            )
    if "NLM" in selected_methods:
        with st.expander("NLM 滤波配置", expanded=True):
            params["nlm_h"] = float_param(
                "NLM 强度系数（倍）",
                0.3,
                2.2,
                float(tuned_params.get("nlm_h", 0.9)),
                0.1,
                "nlm_h",
                "%.1f",
                disabled=auto_locked and "nlm_h" in tuned_params,
            )
            params["nlm_patch_size"] = int_param(
                "NLM patch 大小（像素）",
                3,
                9,
                int(tuned_params.get("nlm_patch_size", 5)),
                2,
                "nlm_patch_size",
                odd=True,
                disabled=auto_locked and "nlm_patch_size" in tuned_params,
            )
            params["nlm_patch_distance"] = int_param(
                "NLM 搜索半径（像素）",
                3,
                15,
                int(tuned_params.get("nlm_patch_distance", 7)),
                1,
                "nlm_patch_distance",
                disabled=auto_locked and "nlm_patch_distance" in tuned_params,
            )

    parameter_export = {
        "uploaded_file": uploaded.name if uploaded is not None else None,
        "noise": {
            "type": noise_type,
            "seed": int(seed),
            "gaussian_sigma": gaussian_sigma,
            "salt_pepper_amount": sp_amount,
            "periodic_strength": periodic_strength,
            "periodic_frequency": periodic_frequency,
        },
        "selected_methods": selected_methods,
        "mixed_noise_preprocess": {
            "enabled": bool(noise_type == "混合噪声" and use_mixed_preprocess),
            "median_size": mixed_preprocess_size,
        },
        "auto_tuning": {
            "enabled": bool(auto_enabled and auto_allowed),
            "locked": auto_locked,
            "iterations": auto_iterations,
            "metric_scope": "full_resolution",
            "result": auto_tune_result,
        },
        "spectrum_recommendation": spectrum_recommendation,
        "method_parameters": dict(params),
    }
    st.header("参数导出")
    st.download_button(
        "一键导出当前参数 JSON",
        json_bytes(parameter_export),
        "current_parameters.json",
        "application/json",
        use_container_width=True,
    )

if source is None:
    st.info("请先上传一张图片。")
    st.stop()

filter_input = median_filter(noisy, mixed_preprocess_size) if noise_type == "混合噪声" and use_mixed_preprocess else noisy
params["filter_input"] = filter_input
outputs, response = run_methods(noisy, selected_methods, params)

source_spectrum = magnitude_spectrum(source)
noisy_spectrum = magnitude_spectrum(noisy)
filtered_spectrum = magnitude_spectrum(outputs["频域滤波"]) if "频域滤波" in outputs else None
hist_images = {"含噪图": noisy}
hist_images.update(outputs)
zip_histogram_df = histogram_dataframe({"含噪图": noisy, **outputs})
metrics_df = metric_table(source, noisy, outputs) if has_reference and outputs else None
auto_tune_applied_df = auto_tune_applied_table(auto_tune_result, source, outputs) if has_reference and outputs else pd.DataFrame()
response_fig = plot_response(response) if response is not None else None

image_tab, spectrum_tab, histogram_tab, edge_tab, sharpen_tab = st.tabs(["图像", "频谱", "灰度直方图", "边缘", "后处理锐化"])

with image_tab:
    top = st.columns([1, 1])
    with top[0]:
        st.markdown('<div class="method-label">原图 / 上传图</div>', unsafe_allow_html=True)
        st.image(source, clamp=True, use_container_width=True)
        image_download_button("下载原图 / 上传图", source, "source_image.png")
    with top[1]:
        st.markdown('<div class="method-label">当前含噪输入</div>', unsafe_allow_html=True)
        st.image(noisy, clamp=True, use_container_width=True)
        image_download_button("下载含噪输入", noisy, "noisy_input.png")

    st.subheader("含噪图数据")
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
        cols = st.columns([1, 1.1, 1]) if len(outputs) == 1 else st.columns(min(3, len(outputs)))
        content_cols = [cols[1]] if len(outputs) == 1 else cols
        for index, (name, image) in enumerate(outputs.items()):
            with content_cols[index % len(content_cols)]:
                st.markdown(f'<div class="method-label">{name}</div>', unsafe_allow_html=True)
                st.image(image, clamp=True, use_container_width=True)
                image_download_button("下载结果", image, f"{name}.png")

    if metrics_df is not None:
        st.subheader("参考指标")
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    if auto_tune_result:
        st.subheader("BO 调参结果")
        if has_reference:
            st.caption("下表按原尺寸最终输出计算，与上方参考指标使用同一口径。")
            st.dataframe(auto_tune_applied_df, hide_index=True, use_container_width=True)
        else:
            st.caption("上传图像已含噪时无参考图，使用 NIQE-like 无参考指标；分数越低越好。")
            st.dataframe(auto_tune_search_table(auto_tune_result), hide_index=True, use_container_width=True)

    st.subheader("一键下载")
    st.download_button(
        "下载当前全部结果 ZIP",
        build_zip(
            source,
            source_spectrum,
            noisy,
            noisy_spectrum,
            outputs,
            response_fig,
            metrics_df,
            zip_histogram_df,
            filtered_spectrum,
            parameter_export,
        ),
        "image_denoising_results.zip",
        "application/zip",
        use_container_width=True,
    )

with spectrum_tab:
    if spectrum_recommendation is not None:
        st.subheader("频谱分析推荐")
        st.info(f"{spectrum_recommendation['recommended_filter']}：{spectrum_recommendation['reason']}")
        rec_cols = st.columns(4)
        rec_cols[0].metric("离散峰组数", spectrum_recommendation["peak_count"])
        rec_cols[1].metric("径向峰组数", spectrum_recommendation["radial_peak_count"])
        rec_cols[2].metric("轴向能量比", f"{spectrum_recommendation['axial_energy_ratio']:.2f}")
        rec_cols[3].metric("高频能量比", f"{spectrum_recommendation['high_frequency_ratio']:.2f}")
        if spectrum_recommendation["notch_points"]:
            st.caption(f"建议陷波点：{spectrum_recommendation['notch_points']}")
        if spectrum_recommendation["radial_bands"]:
            st.caption(f"建议径向 Band-Stop：{spectrum_recommendation['radial_bands']}")

    spectrum_cols = st.columns([1, 1, 1])
    with spectrum_cols[0]:
        st.subheader("原图频谱")
        st.image(source_spectrum, clamp=True, use_container_width=True)
        image_download_button("下载原图频谱", source_spectrum, "source_spectrum.png")
    with spectrum_cols[1]:
        st.subheader("含噪图频谱")
        st.image(noisy_spectrum, clamp=True, use_container_width=True)
        image_download_button("下载含噪图频谱", noisy_spectrum, "noisy_spectrum.png")
    with spectrum_cols[2]:
        st.subheader("频域响应")
        if response_fig is not None:
            st.pyplot(response_fig, clear_figure=False)
            figure_download_button("下载频域响应", response_fig, "frequency_response.png")
        else:
            st.caption("未选择频域滤波。")

    if filtered_spectrum is not None:
        small_cols = st.columns([1, 1, 1])
        with small_cols[1]:
            st.subheader("频域滤波后频谱")
            st.image(filtered_spectrum, caption="频域滤波频谱", clamp=True, use_container_width=True)
            image_download_button("下载频域滤波频谱", filtered_spectrum, "频域滤波_spectrum.png")

with histogram_tab:
    st.subheader("灰度直方图")
    st.altair_chart(histogram_chart(hist_images), use_container_width=True)

with edge_tab:
    st.subheader("边缘保留观察")
    edge_images = {"含噪图": noisy, **outputs}
    cols = st.columns(min(3, len(edge_images)))
    for index, (name, image) in enumerate(edge_images.items()):
        edges = edge_map(image)
        with cols[index % len(cols)]:
            st.image(edges, caption=f"{name} 边缘", clamp=True, use_container_width=True)
            image_download_button("下载边缘图", edges, f"{safe_filename(name)}_edges.png")

with sharpen_tab:
    st.subheader("后处理锐化")
    if not outputs:
        st.warning("请先至少选择并运行一种滤波方法，再从其输出中选择图片进行锐化。")
    else:
        sharpen_source_name = st.selectbox("选择一张滤波结果", list(outputs.keys()))
        sharpen_radius = float_param("USM 模糊半径 sigma（像素）", 0.1, 8.0, 1.2, 0.1, "sharpen_radius", "%.1f")
        sharpen_amount = float_param("USM 锐化强度", 0.0, 3.0, 0.8, 0.1, "sharpen_amount", "%.1f")
        sharpen_threshold = float_param("细节阈值", 0.0, 0.2, 0.0, 0.01, "sharpen_threshold", "%.2f")

        if st.button("执行后处理锐化", use_container_width=True):
            source_for_sharpening = outputs[sharpen_source_name]
            st.session_state["sharpened_result"] = unsharp_mask(
                source_for_sharpening,
                sharpen_radius,
                sharpen_amount,
                sharpen_threshold,
            )
            st.session_state["sharpened_source_name"] = sharpen_source_name
            st.session_state["sharpened_params"] = {
                "radius": sharpen_radius,
                "amount": sharpen_amount,
                "threshold": sharpen_threshold,
            }

        if "sharpened_result" in st.session_state:
            displayed_source_name = st.session_state.get("sharpened_source_name", sharpen_source_name)
            if displayed_source_name not in outputs:
                displayed_source_name = sharpen_source_name
            st.caption(
                f"当前锐化来源：{displayed_source_name}；"
                f"参数：{st.session_state.get('sharpened_params')}"
            )
            cols = st.columns([1, 1])
            with cols[0]:
                st.markdown('<div class="method-label">锐化前</div>', unsafe_allow_html=True)
                st.image(outputs[displayed_source_name], clamp=True, use_container_width=True)
            with cols[1]:
                st.markdown('<div class="method-label">USM 后处理锐化</div>', unsafe_allow_html=True)
                st.image(st.session_state["sharpened_result"], clamp=True, use_container_width=True)
                image_download_button("下载锐化结果", st.session_state["sharpened_result"], "usm_sharpened.png")
        else:
            st.info("设置参数后点击“执行后处理锐化”才会生成锐化结果。")

if response_fig is not None:
    plt.close(response_fig)
