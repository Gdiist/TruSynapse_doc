图像分类
========

应用概述
--------
图像分类任务的目标是将输入图像映射到预定义类别标签。本章节仅比较 SNN 网络在
CPU 2.5GHz 与 TruSynapse 平台上的精度与推理时间，不再包含 ANN 对照。

- MNIST：手写数字分类数据集，10 个类别，训练集 60,000 张，测试集 10,000 张，输入尺寸 1x28x28。
- CIFAR10：通用目标分类数据集，10 个类别，训练集 50,000 张，测试集 10,000 张，输入尺寸 3x32x32。

本章节包含四个 SNN 实现方案：

- `mnist-MLP`：全连接脉冲网络（784-512-256-10）。
- `mnist-sparse`：基于稀疏连接策略的 MLP 脉冲网络。
- `mnist-conv`：MNIST 卷积脉冲网络。
- `cifar10-conv`：CIFAR10 卷积脉冲网络。

实现方案一 ：mnist-MLP
--------
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

实现方案二 ：mnist-sparse
--------
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
--------
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
--------
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

性能评估
--------
本节仅比较 SNN 在 CPU 2.5GHz 与 TruSynapse 两类推理平台上的精度与推理时间。
同一模型在两类设备上的测试数据规模应保持一致。

.. list-table:: SNN 精度与推理时间对比（CPU 2.5GHz vs TruSynapse）
   :header-rows: 1
   :widths: 14 12 12 12 16 14 14

   * - 任务
     - 模型类型
     - input_size
     - 数据量
     - 推理设备
     - 准确率
     - 推理时间
   * - mnist-MLP
     - SNN
     - 1x28x28
     - 100 批
     - CPU 2.5GHz
     - 待补充
     - 待补充
   * - mnist-MLP
     - SNN
     - 1x28x28
     - 100 批
     - TruSynapse 1.5GHz
     - 0.97
     - 0.0744 s
   * - mnist-sparse
     - SNN
     - 1x28x28
     - 待补充
     - CPU 2.5GHz
     - 待补充
     - 待补充
   * - mnist-sparse
     - SNN
     - 1x28x28
     - 待补充
     - TruSynapse
     - 待补充
     - 待补充
   * - mnist-conv
     - SNN
     - 1x28x28
     - 待补充
     - CPU 2.5GHz
     - 0.96
     - 待补充
   * - mnist-conv
     - SNN
     - 1x28x28
     - 待补充
     - TruSynapse
     - 待补充
     - 待补充
   * - cifar10-conv
     - SNN
     - 3x32x32
     - 待补充
     - CPU 2.5GHz
     - 待补充
     - 待补充
   * - cifar10-conv
     - SNN
     - 3x32x32
     - 待补充
     - TruSynapse
     - 待补充
     - 待补充


