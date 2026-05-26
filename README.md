# 图像去噪实时演示

这是一个面向《信号与系统》课程展示的本地交互式程序。用户上传图片后，程序会实时添加噪声、执行去噪算法、展示 DFT 频谱与滤波响应，并在有干净参考图时计算 MSE、PSNR、SSIM。

## 功能

- 上传任意图片作为原图，或上传已含噪图片并跳过加噪
- 实时添加高斯噪声、椒盐噪声、混合噪声、横向周期噪声
- 空域滤波：均值滤波、中值滤波
- 频域滤波：高斯低通、巴特沃斯低通、巴特沃斯成对陷波带阻滤波器
- 保边滤波：双边滤波
- 高级去噪：非局部均值 NLM
- 展示原图频谱、含噪图频谱、频域滤波后频谱、频域响应、灰度直方图、边缘图
- 频域滤波参数完全手动配置，成对陷波带阻滤波器支持多组手动陷波点
- 所有数值参数均可直接键入，便于按频谱亮点位置精确设置参数
- 混合噪声支持先用中值滤波预处理，再交给其他滤波器
- 有参考图时显示 MSE、PSNR、SSIM；无参考图时改用视觉分析
- 顶部导航栏分为图像、频谱、灰度直方图、边缘四个视图，切换视图不会影响侧边栏参数

## 运行

建议使用 Python 3.12。Streamlit Community Cloud 部署时，在 Advanced settings 里选择 Python 3.12，避免 Python 3.14 与 SciPy / scikit-image 等图像科学依赖出现二进制导入兼容问题。

```powershell
streamlit run app.py --browser.gatherUsageStats=false
```

如果缺少依赖：

```powershell
pip install -r requirements.txt
```

## 展示建议

1. 先上传一张边缘和纹理比较明显的图片。
2. 选择“高斯噪声”，手动配置均值滤波、频域低通、双边滤波和 NLM，说明不同方法的平滑与保边差异。
3. 选择“椒盐噪声”，手动配置中值滤波，突出中值滤波对脉冲噪声的优势。
4. 选择“周期噪声”和“Butterworth Notch Reject”，根据频谱中关于中心成对出现的亮点，增加一组或多组陷波点，手动调整每组 Δu、Δv、陷波半径、阶数和抑制强度，观察 DFT 频谱中周期条纹对应异常分布的变化。

## 文件结构

- `app.py`：Streamlit 界面、参数组织、结果展示和下载
- `denoise_lab/noise.py`：噪声生成
- `denoise_lab/spatial.py`：均值滤波和中值滤波
- `denoise_lab/frequency.py`：频域响应和频域滤波
- `denoise_lab/advanced.py`：双边滤波和 NLM
- `denoise_lab/analysis.py`：频谱、边缘、直方图和参考指标
- `denoise_lab/image_io.py`：图像格式转换
