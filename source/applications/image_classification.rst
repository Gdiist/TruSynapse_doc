图像分类
========

应用概述
--------
图像分类任务的目标是将输入图像映射到预定义类别标签。本章节展示 SNN 网络的推理结果，不再包含 ANN 对照。

- MNIST：手写数字分类数据集，10 个类别，训练集 60,000 张，测试集 10,000 张，输入尺寸 1x28x28。
- CIFAR10：通用目标分类数据集，10 个类别，训练集 50,000 张，测试集 10,000 张，输入尺寸 3x32x32。

本章节包含四个 SNN 实现方案：

- `mnist-MLP`：全连接脉冲网络（784-512-256-10）。
- `mnist-sparse`：基于稀疏连接策略的 MLP 脉冲网络。
- `mnist-conv`：MNIST 卷积脉冲网络。
- `cifar10-conv`：CIFAR10 卷积脉冲网络。

实现方案一 ：mnist-MLP
----------------------

数据输入演示
^^^^^^^^^^^^

1. 加载和可视化MNIST图像：

.. code-block:: python
    :linenos:

    import torch
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. 加载10张MNIST图像
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root='../../data', train=True, download=True, transform=transform)
    images = []
    labels = []
    for i in range(10):
        image, label = dataset[i]
        images.append(image)
        labels.append(label)
    
    print(f"加载了{len(images)}张图像，标签为: {labels}")

    # 2. 可视化第一张图像
    plt.figure(figsize=(4, 4))
    plt.imshow(images[0].squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()

执行上述代码后，将显示如下MNIST数字图像：

.. image:: ../_static/images/mnist_sample_demo.png
   :width: 400px
   :align: center
   :alt: MNIST样本演示

2. 转换为脉冲向量：

.. code-block:: python
    :linenos:

    # 3. 将10张图像转换为1D向量并拼接
    all_vectors = []
    for i, image in enumerate(images):
        vector_1d = image.flatten()  # 28x28 -> 784
        all_vectors.append(vector_1d)
    
    # 拼接所有向量为一个大向量 (10 * 784 = 7840)
    combined_vector = torch.cat(all_vectors, dim=0)

    # 4. 转换为脉冲向量 (阈值化为0或1)
    threshold = 0.5  # 阈值设置为0.5
    spike_vector = (combined_vector > threshold).int()  # 转换为整数类型

    # 5. 保存脉冲向量到input.txt
    np.savetxt('result/input.txt', spike_vector.numpy(), fmt='%d')
    print(f"10张图像的脉冲向量已保存到: result/input.txt")

执行输出结果：

.. code-block:: text

    10张图像的脉冲向量已保存到: result/input.txt

网络结构定义
^^^^^^^^^^^^

- `mnist-MLP` 网络结构（SNN）

.. code-block:: python
    :linenos:

    class MnistSNN(nn.Module):
        def __init__(self, input_neuron_num=784, hidden1=512, hidden2=256, output_neuron_num=10, beta=0.9):
            super().__init__()
            self.fc1 = nn.Linear(input_neuron_num, hidden1, bias=False)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

            self.fc2 = nn.Linear(hidden1, hidden2, bias=False)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

            self.fc3 = nn.Linear(hidden2, output_neuron_num, bias=False)
            self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        def forward(self, x):
            if x.dim() == 4:
                x = x.view(x.size(0), -1)

            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()

            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
            return spk3, mem3



使用NFU测试推理结果
^^^^^^^^^^^^^^^^^^

NFU推理过程详细输出：

.. code-block:: text

    Time(ns) | Raw_Hex  | Timestamp | GNC | Neuron | Note
    ---------|----------|-----------|-----|--------|----------
    11905390 | 0002c017 |     1     |  6  |   23   | Data
    25396850 | 0004c017 |     2     |  6  |   23   | Data  
    25396890 | 0004c012 |     2     |  6  |   18   | Data
    31632410 | 0006c012 |     3     |  6  |   18   | Data
    112331090| 00000001 |     -     |  -  |    -   | Finish Flag
    220061150| 00000001 |     -     |  -  |    -   | Finish Flag
    327831230| 00000001 |     -     |  -  |    -   | Finish Flag
    435538630| 00000001 |     -     |  -  |    -   | Finish Flag
    543221930| 00000001 |     -     |  -  |    -   | Finish Flag
    650929350| 00000001 |     -     |  -  |    -   | Finish Flag
    758702230| 00000001 |     -     |  -  |    -   | Finish Flag
    866410990| 00000001 |     -     |  -  |    -   | Finish Flag

.. **推理结果分析：**

.. - **神经元激活**: 检测到第6组神经元(GNC=6)的18号和23号神经元产生输出脉冲
.. - **时间戳序列**: 1-3个时间步内完成主要计算
.. - **完成标志**: 8个Finish Flag表明推理过程正常完成
.. - **输出神经元**: 18号和23号神经元可能对应不同的分类结果

实现方案二 ：mnist-sparse
-------------------------

- `mnist-sparse` 采用稀疏训练策略，参考文献：`Rigging the Lottery: Making All Tickets Winners <https://ojs.aaai.org/index.php/AAAI/article/view/25079>`_。
- 与 `mnist-MLP` 保持相同层级拓扑（784-512-256-10），但在线性层引入稀疏连接掩码。
- 训练阶段采用剪枝-再生长（prune-regrow）策略，目标稀疏度可配置（当前配置为 90%）。

.. code-block:: python
    :linenos:

    class StaticSparseSNNLinear(nn.Module):
        def __init__(self, in_features, out_features, sparsity=0.9, beta=0.9):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
            self.register_buffer("mask", torch.ones_like(self.weight))
            self.lif = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
            self.sparsity = sparsity
            self._initialize_sparse_connections()

        def _initialize_sparse_connections(self):
            num_connections = int(self.weight.numel() * (1 - self.sparsity))
            flat_mask = torch.zeros(self.weight.numel())
            indices = torch.randperm(self.weight.numel())[:num_connections]
            flat_mask[indices] = 1
            self.mask.copy_(flat_mask.view_as(self.weight))

        def forward(self, x, mem):
            sparse_weight = self.weight * self.mask
            cur = F.linear(x, sparse_weight, None)
            spk, mem = self.lif(cur, mem)
            return spk, mem

    class SparseMLPSNN(nn.Module):
        def __init__(self, input_size=784, hidden_sizes=(512, 256), output_size=10, sparsity=0.9, beta=0.9):
            super().__init__()
            sizes = [input_size] + list(hidden_sizes) + [output_size]
            self.layers = nn.ModuleList(
                [StaticSparseSNNLinear(sizes[i], sizes[i + 1], sparsity=sparsity, beta=beta)
                 for i in range(len(sizes) - 1)]
            )

实现方案三 ：mnist-conv
-----------------------

- `mnist-conv` 网络结构（SNN）

.. code-block:: python
    :linenos:

    class ConvMnistSNN(nn.Module):
        def __init__(self, beta=0.9):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

            self.fc = nn.Linear(16 * 14 * 14, 10, bias=False)
            self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()

            spk1, mem1 = self.lif1(self.conv1(x), mem1)
            spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
            spk2_flat = spk2.view(spk2.size(0), -1)
            spk3, mem3 = self.lif3(self.fc(spk2_flat), mem3)
            return spk3, mem3

实现方案四 ：cifar10-conv
-------------------------

- `cifar10-conv` 网络结构（SNN）

.. code-block:: python
    :linenos:

    class Cifar10ConvSNN(nn.Module):
        def __init__(self, beta=0.9):
            super().__init__()

            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

            self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

            self.fc = nn.Linear(32 * 4 * 4, 10, bias=False)
            self.lif4 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
            mem4 = self.lif4.init_leaky()

            spk1, mem1 = self.lif1(self.conv1(x), mem1)
            spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
            spk3, mem3 = self.lif3(self.conv3(spk2), mem3)
            spk3_flat = spk3.view(spk3.size(0), -1)
            spk4, mem4 = self.lif4(self.fc(spk3_flat), mem4)
            return spk4, mem4

推理结果
--------

.. list-table:: SNN 推理时间（TruSynapse）
   :header-rows: 1
   :widths: 14 12 12 12 16 14

   * - 任务
     - 模型类型
     - input_size
     - 数据量
     - 推理设备
     - 推理时间
   * - mnist-MLP
     - SNN
     - 1x28x28
     - 100 批
     - TruSynapse
     - 0.0744 s
   * - mnist-sparse
     - SNN
     - 1x28x28
     - 100 批
     - TruSynapse
     - 待补充
   * - mnist-conv
     - SNN
     - 1x28x28
     - 100 批
     - TruSynapse
     - 待补充
   * - cifar10-conv
     - SNN
     - 3x32x32
     - 10 批
     - TruSynapse
     - 待补充


