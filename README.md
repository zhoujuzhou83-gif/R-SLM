# R-SLM
面向 12 导联 ECG 自动分析与临床报告生成，构建了基于 SE-ResNet 风格 1D ECG 编码器、Perceiver Resampler 和 Qwen3-1.7B 的多模态生成模型。模型首先从原始心电信号中提取时序特征，通过带位置编码、多尺度局部卷积和交叉注意力的 Perceiver Resampler 将 ECG 特征压缩为 32 个心电 Token，再映射到 Qwen 隐空间，与文本 Token 拼接后输入因果语言模型，实现 ECG 条件下的诊断报告生成。
