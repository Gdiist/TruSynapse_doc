自动驾驶
========

应用概述
--------
自动驾驶转向角预测任务的目标是将车载摄像头采集的道路图像映射为车辆实时转向角数值。本章节采用 `PilotNet` 作为基础网络，面向端到端转向角回归任务。

- 数据集：`Udacity Mini Challenge 2 <https://github.com/udacity/self-driving-car/tree/master/datasets>`_。
- Udacity Mini Challenge 2 数据集是评估自动驾驶端到端转向预测算法性能的轻量化公开数据集。该数据集包含 5614 条样本，每条样本由车载中心摄像头采集的 RGB 道路图像（覆盖城市 / 高速道路的直道、弯道场景）和人类驾驶员操作的实时转向角标签组成。

- 网络模型：`PilotNet <https://github.com/lhzlhz/PilotNet>`_。
- PilotNet 是 NVIDIA 提出的经典轻量化卷积神经网络，专为自动驾驶端到端转向角预测设计，具备结构简洁、推理速度快的特点，适配车载嵌入式设备的部署需求。

实现方案 ：PilotNet（SNN）
--------------------------
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


推理结果
--------
.. list-table:: SNN 推理时间（TruSynapse）
   :header-rows: 1
   :widths: 14 12 14 12 16 14

   * - 任务
     - 模型类型
     - input_size
     - 数据量
     - 推理设备
     - 推理时间
   * - Udacity Mini Challenge 2 / PilotNet
     - SNN
     - 3*33*100
     - 10 批
     - TruSynapse
     - 待补充
