# TruSynapse Documentation

TruSynapse 是一个为TruthSouth芯片开发的编程框架，旨在为类脑计算提供高效的开发和部署平台。

## 文档结构

```
source/
├── introduction/           # 框架介绍与基础概念
├── getting_started/        # 快速入门指南
│   ├── installation.rst    # 安装说明
│   ├── data_preparation.rst # 数据准备
│   └── run_a_net.rst       # 运行网络
├── models/                 # 神经网络模型示例
│   ├── mlp.rst             # 多层感知机
│   ├── cnn.rst             # 卷积神经网络
│   ├── lnn.rst             # 液态神经网络
│   ├── resnet.rst          # 残差网络
│   └── transformer.rst     # Transformer
├── applications/           # 应用场景
│   ├── image_classification.rst
│   ├── speech_recognition.rst
│   ├── autonomous_driving.rst
│   └── brain_simulation.rst
├── higher_level_use/       # 高级用法
└── api_datastruct_documentation/  # API 参考
```

## 在线文档

文档发布在 GitHub Pages 上：

**https://which197.github.io/TruSynapse_doc/**

## 本地构建

### 环境要求

- Python 3.10+
- Sphinx 5.0+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 构建文档

```bash
# Linux/macOS
make html

# Windows
.\make.bat html
```

构建完成后，打开 `build/html/index.html` 查看文档。

## 支持的网络模型

| 模型 | 说明 |
|------|------|
| MLP | 多层感知机，适用于 MNIST 手写数字识别 |
| CNN | 卷积神经网络，适用于图像分类任务 |
| LNN | 液态神经网络，适用于时序数据处理 |


## 依赖框架

- [snntorch](https://github.com/jeshraghian/snntorch) - 脉冲神经网络训练框架
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！
