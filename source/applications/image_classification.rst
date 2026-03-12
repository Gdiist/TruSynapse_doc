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

实现方案一：mnist-MLP
----------------------

数据输入演示
^^^^^^^^^^^^^^^^^^

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

.. image:: ../_static/images/mnist_sample_demo15.png
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
^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^

NFU推理过程详细输出：

.. code-block:: text

    输出层神经元到GNC的映射:
    {1552: 0, 1553: 0, 1554: 0, 1555: 0, 1556: 0, 1557: 0, 1558: 0, 1559: 0, 1560: 0, 1561: 0}

    Time(ns) | Raw_Hex  | Timestamp | GNC | Neuron | Note
    ---------|----------|-----------|-----|--------|----------
    28917630 | 00080617 |     4     |  0  |  1559  | Data

**推理结果分析：**

- **神经元映射**: 输出层神经元1552-1561对应数字0-9，都映射到GNC=0
- **神经元激活**: 检测到GNC=0的1559号神经元产生输出脉冲
- **分类结果**: 1559号神经元对应数字7


实现方案二：mnist-sparse
-------------------------

- `mnist-sparse` 采用稀疏训练策略，参考文献：`Rigging the Lottery: Making All Tickets Winners <https://ojs.aaai.org/index.php/AAAI/article/view/25079>`_。
- 与 `mnist-MLP` 保持相同层级拓扑（784-512-256-10），但在线性层引入稀疏连接掩码。
- 训练阶段采用剪枝-再生长（prune-regrow）策略，目标稀疏度可配置（当前配置为 90%）。

数据输入说明
^^^^^^^^^^^^^^^^^^

**输入数据与mnist-MLP相同**：使用相同的10张MNIST图像和相同的脉冲向量转换方法，输入数据文件为 `result/input.txt`。

网络结构定义
^^^^^^^^^^^^^^^^^^^

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



使用NFU测试推理结果
^^^^^^^^^^^^^^^^^^^^^^^^

NFU推理过程详细输出：

.. code-block:: text

    Time(ns) | Raw_Hex  | Timestamp | GNC | Neuron | Note
    ---------|----------|-----------|-----|--------|----------
    8316150  | 00080617 |     4     |  0  |  1559  | Data


**推理结果分析（稀疏网络）：**

- **神经元激活**: 检测到GNC=0的1559号神经元产生输出脉冲
- **分类结果**: 1559号神经元对应数字7



实现方案三：mnist-conv
-----------------------

- `mnist-conv` 网络结构（SNN）

数据输入说明
^^^^^^^^^^^^^^^^^^^

**输入数据与mnist-MLP相同**：使用相同的10张MNIST图像和相同的脉冲向量转换方法，输入数据文件为 `result/input.txt`。

网络结构定义
^^^^^^^^^^^^^^^^^^^

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

使用NFU测试推理结果
^^^^^^^^^^^^^^^^^^^^^^^^^

NFU推理过程详细输出：

.. code-block:: text

   输出层神经元到GNC的映射:
    ['1_2000', '1_2001', '1_2002', '1_2003', '1_2004', '1_2005', '1_2006', '1_2007', '1_2008', '1_2009']

    Time(ns) | Raw_Hex  | Timestamp | GNC | Neuron | Note
    ---------|----------|-----------|-----|--------|----------
    9151970  | 000627d7 |     3     |  1  |  2007  | Data


**推理结果分析（卷积网络）：**

- **神经元激活**: 检测到GNC=1的2007号神经元产生输出脉冲
- **分类结果**: 2007号神经元对应数字7



实现方案四：cifar10-conv
-------------------------

数据输入演示
^^^^^^^^^^^^^^^^^^^

1. 加载和可视化CIFAR-10图像：

.. code-block:: python
    :linenos:

    import torch
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import random

    # CIFAR-10类别标签
    CIFAR10_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 创建result目录
    os.makedirs('result', exist_ok=True)

    # 1. 加载CIFAR-10数据集并随机选择1张图片
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root='../data/CIFAR10', train=True, download=True, transform=transform)
    
    random.seed(42)
    indices = random.sample(range(len(dataset)), 1)
    images = []
    labels = []
    
    for idx in indices:
        image, label = dataset[idx]
        images.append(image)
        labels.append(label)

    # 2. 创建单张图片显示
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # 显示图片
    ax.imshow(images[0].permute(1, 2, 0))
    ax.set_title(f'CIFAR-10 Sample: {CIFAR10_CLASSES[labels[0]]}\n(Label: {labels[0]})', fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('result/cifar10_single_image.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"加载了{len(images)}张图像，标签为: {labels}")

执行上述代码后，将显示如下单张CIFAR-10图像：

.. image:: ../_static/images/summary_dog_image.png
   :width: 400px
   :align: center
   :alt: CIFAR-10单张样本演示

2. 转换为二进制脉冲向量：

.. code-block:: python
    :linenos:

    # 泊松编码转换为二进制脉冲向量
    def poisson_encoding(images, threshold=0.5, time_steps=1, seed=42):
        """
        将图像转换为二进制脉冲向量 (类似MNIST binary flat格式)
        
        Args:
            images: 图像列表
            threshold: 二值化阈值
            time_steps: 时间步数 (设为1生成二进制flat格式)
            seed: 随机种子
        
        Returns:
            binary_vectors: 二进制向量列表
        """
        np.random.seed(seed)
        all_binary_vectors = []
        
        for i, image in enumerate(images):
            # 将图像展平为1维 (3072维: 3×32×32)
            flat_image = image.flatten().numpy()
            
            # 使用泊松过程生成二进制向量
            # 像素值作为发放概率
            binary_vector = (np.random.random(len(flat_image)) < flat_image).astype(int)
            
            all_binary_vectors.append(binary_vector)
            print(f"图片 ({CIFAR10_CLASSES[labels[i]]}): {len(binary_vector)}维, {binary_vector.sum()}个脉冲")
        
        return all_binary_vectors

    # 生成二进制脉冲向量
    print("\n=== 生成CIFAR-10单张图片二进制脉冲向量 ===")
    binary_vectors = poisson_encoding(images, threshold=0.5, time_steps=1)

    # 保存为binary flat格式 (每行一个值)
    output_file = '../test_data/cifar10_single_binary_flat.txt'
    os.makedirs('../test_data', exist_ok=True)

    with open(output_file, 'w') as f:
        binary_vector = binary_vectors[0]
        # 每行一个值，不包含注释行
        for value in binary_vector:
            f.write(f'{value}\n')

    print(f"单张CIFAR-10图像的脉冲向量已保存到: {output_file}")

执行输出结果：

.. code-block:: text
   
    === 生成CIFAR-10单张图片二进制脉冲向量 ===
    图片 (frog): 3072维, 893个脉冲
    单张CIFAR-10图像的脉冲向量已保存到: ../test_data/cifar10_single_binary_flat.txt

网络结构定义
^^^^^^^^^^^^^^^^^^^

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

使用NFU测试推理结果
^^^^^^^^^^^^^^^^^^^^^^^

NFU推理过程详细输出：

.. code-block:: text

    输出层神经元到GNC的映射:
    {6656: 0, 6657: 0, 6658: 0, 6659: 0, 6660: 0, 6661: 0, 6662: 0, 6663: 0, 6664: 0, 6665: 0}

    Time(ns) | Raw_Hex  | Timestamp | GNC | Neuron | Note
    ---------|----------|-----------|-----|--------|----------
    7598810  | 000a1a05 |     5     |  0  |  6661  | Data



**推理结果分析（CIFAR-10卷积网络）：**

- **神经元映射**: 输出层神经元6656-6665对应CIFAR-10类别0-9
- **神经元激活**: 检测到GNC=0的6661号神经元产生输出脉冲
- **分类结果**: 6661号神经元对应类别 **dog**