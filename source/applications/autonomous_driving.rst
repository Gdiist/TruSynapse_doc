自动驾驶
========

应用概述
--------
自动驾驶转向角预测任务的目标是将车载摄像头采集的道路图像映射为车辆实时转向角数值。本章节采用 `PilotNet` 作为基础网络，面向端到端转向角回归任务。

- 数据集：`Udacity Mini Challenge 2 <https://github.com/udacity/self-driving-car/tree/master/datasets>`_。
- Udacity Mini Challenge 2 数据集是评估自动驾驶端到端转向预测算法性能的轻量化公开数据集。该数据集包含 5614 条样本，每条样本由车载中心摄像头采集的 RGB 道路图像（覆盖城市 / 高速道路的直道、弯道场景）和人类驾驶员操作的实时转向角标签组成。

- 网络模型：`PilotNet <https://github.com/lhzlhz/PilotNet>`_。
- PilotNet 是 NVIDIA 提出的经典轻量化卷积神经网络，专为自动驾驶端到端转向角预测设计，具备结构简洁、推理速度快的特点，适配车载嵌入式设备的部署需求。

数据输入演示
----------------

1. 加载和可视化自动驾驶图像：

.. code-block:: python
    :linenos:

    import os
    import cv2
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    from decimal import Decimal, InvalidOperation
    import torch

    # 1. 加载自动驾驶数据集
    DATASET_PATH = r".\data\Ch2_001"
    INPUT_SIZE = (33, 100)  # (H, W)

    df = pd.read_csv(
        os.path.join(DATASET_PATH, "final_example.csv"),
        dtype={"frame_id": "string"},
    )
    center_dir = os.path.join(DATASET_PATH, "center")

    # 2. 随机选择10张图片
    random.seed(42)
    indices = random.sample(range(min(len(df), 1000)), 10)
    images_original = []
    images_processed = []
    steerings = []
    frame_ids = []

    for idx in indices:
        frame_id = str(int(float(str(df.iloc[idx]["frame_id"]).strip())))
        img_files = [f for f in os.listdir(center_dir) if f.startswith(frame_id + "_")]
        
        if img_files:
            img_path = os.path.join(center_dir, img_files[0])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            images_original.append(img.copy())
            processed_img = cv2.resize(img, (INPUT_SIZE[1], INPUT_SIZE[0])) / 255.0
            images_processed.append(torch.from_numpy(processed_img).permute(2, 0, 1).float())
            steerings.append(df.iloc[idx]["steering_angle"])
            frame_ids.append(frame_id)

    print(f"成功加载{len(images_original)}张自动驾驶图像")

执行上述代码后，输出：

.. code-block:: text

    成功加载10张自动驾驶图像

2. 可视化图像网格：

.. code-block:: python
    :linenos:

    # 3. 创建2x5的网格显示图片
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(len(images_original)):
        axes[i].imshow(images_original[i])
        axes[i].set_title(f'Sample {i}\nFrame: {frame_ids[i]}\nSteering: {steerings[i]:.3f}', fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('result/autonomous_driving_10_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

执行上述代码后，将显示如下自动驾驶图像网格：

.. image:: ../_static/images/autonomous_driving_10_samples.png
   :width: 800px
   :align: center
   :alt: 自动驾驶样本演示

3. 转换为二进制脉冲向量：

.. code-block:: python
    :linenos:

    # 4. 转换为脉冲向量
    binary_vectors = []
    np.random.seed(42)

    for i, img_tensor in enumerate(images_processed):
        # 将图像展平为1维 (3*33*100 = 9900维)
        flat_image = img_tensor.flatten().numpy()
        
        # 使用泊松过程生成二进制向量
        binary_vector = (np.random.random(len(flat_image)) < flat_image).astype(int)
        binary_vectors.append(binary_vector)
        
        print(f"图像{i}: {len(binary_vector)}维, {binary_vector.sum()}个脉冲, 转向角度: {steerings[i]:.3f}")

    # 5. 保存为binary flat格式
    output_file = '../test_data/autonomous_driving_10_binary_flat.txt'
    os.makedirs('../test_data', exist_ok=True)

    with open(output_file, 'w') as f:
        for i, binary_vector in enumerate(binary_vectors):
            binary_str = ' '.join(map(str, binary_vector))
            f.write(f"# Sample {i}: Frame {frame_ids[i]}, Steering {steerings[i]:.3f}\n")
            f.write(binary_str + '\n')

    print(f"10张自动驾驶图像的脉冲向量已保存到: {output_file}")

执行输出结果：

.. code-block:: text

    图像0: 9900维, 4832个脉冲, 转向角度: 0.247
    图像1: 9900维, 4756个脉冲, 转向角度: -0.156
    图像2: 9900维, 4891个脉冲, 转向角度: 0.089
    图像3: 9900维, 4623个脉冲, 转向角度: -0.012
    图像4: 9900维, 4677个脉冲, 转向角度: 0.134
    图像5: 9900维, 4834个脉冲, 转向角度: -0.067
    图像6: 9900维, 4598个脉冲, 转向角度: 0.201
    图像7: 9900维, 4712个脉冲, 转向角度: -0.089
    图像8: 9900维, 4856个脉冲, 转向角度: 0.178
    图像9: 9900维, 4743个脉冲, 转向角度: -0.023
    
    10张自动驾驶图像的脉冲向量已保存到: ../test_data/autonomous_driving_10_binary_flat.txt

实现方案 ：PilotNet（SNN）
----------------------------------
- `PilotNet` 是端到端自动驾驶网络，用于将前视图像直接映射为转向角。
- 原始实现以卷积特征提取 + 全连接回归为核心结构。
- 本章节使用的网络输入尺寸配置为 `3x33x100`。

.. code-block:: python
    :linenos:
      
    class SingleStepSNN_PilotNet(nn.Module):
        def __init__(self):
            super().__init__()
            spike_grad = surrogate.fast_sigmoid()

            self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, bias=False)
            self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, bias=False)
            self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, bias=False)
            self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

            self.fc1 = nn.Linear(64 * 1 * 9, 100, bias=False)
            self.fc2 = nn.Linear(100, 50, bias=False)
            self.fc3 = nn.Linear(50, 10, bias=False)
            self.fc4 = nn.Linear(10, 1, bias=False)

            self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad, reset_mechanism="zero")
            self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad, reset_mechanism="zero")
            self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad, reset_mechanism="zero")
            self.lif4 = snn.Leaky(beta=0.9, spike_grad=spike_grad, reset_mechanism="zero")
            self.lif5 = snn.Leaky(beta=0.9, spike_grad=spike_grad, reset_mechanism="zero")
            self.lif6 = snn.Leaky(beta=0.9, spike_grad=spike_grad, reset_mechanism="zero")
            self.lif7 = snn.Leaky(beta=0.9, spike_grad=spike_grad, reset_mechanism="zero")
            self.lif8 = snn.Leaky(beta=0.9, spike_grad=spike_grad, reset_mechanism="zero")

        def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
            mem4 = self.lif4.init_leaky()
            mem5 = self.lif5.init_leaky()
            mem6 = self.lif6.init_leaky()
            mem7 = self.lif7.init_leaky()
            mem8 = self.lif8.init_leaky()

            spk1, mem1 = self.lif1(self.conv1(x), mem1)
            spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
            spk3, mem3 = self.lif3(self.conv3(spk2), mem3)
            spk4, mem4 = self.lif4(self.conv4(spk3), mem4)
            spk5, mem5 = self.lif5(self.conv5(spk4), mem5)
            spk5_flat = spk5.view(spk5.size(0), -1)
            spk6, mem6 = self.lif6(self.fc1(spk5_flat), mem6)
            spk7, mem7 = self.lif7(self.fc2(spk6), mem7)
            spk8, mem8 = self.lif8(self.fc3(spk7), mem8)
            out = self.fc4(spk8)
            return out.squeeze()


.. 使用NFU测试推理结果
.. ------------------------

.. NFU推理过程详细输出：
.. 待补充..