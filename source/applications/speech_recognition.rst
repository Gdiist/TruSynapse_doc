语音识别
========

应用概述
--------
语音识别任务的目标是将输入语音信号映射到预定义类别标签。

- 数据集：Speech Commands（关键词识别）。

Google语音命令数据集（V2）是评估关键词检测算法性能的常用数据集。该数据集包含2,618位不同说话者说的35个不同单词的105,829段1秒长的语音片段。数据以16 kHz采样频率进行编码，采用线性16位单通道脉冲编码调制格式。

本章节当前采用一个 SNN 实现方案：

- `gsc-MLP`：面向 Speech Commands 的全连接脉冲网络。

实现方案 ：gsc-MLP
-------------------
- `gsc-MLP` 网络结构（SNN）
- 输入为预处理后的语音特征向量（示例配置：`input_neuron_num=4000`）

.. code-block:: python
    :linenos:

    class SpeechCommandsSNN(nn.Module):
        def __init__(
            self,
            input_neuron_num=4000,
            hidden_dim=256,
            output_neuron_num=35,
            beta=0.9,
        ):
            super().__init__()

            spike_grad = surrogate.fast_sigmoid()

            self.fc1 = nn.Linear(input_neuron_num, hidden_dim, bias=False)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

            self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

            self.fc3 = nn.Linear(hidden_dim, output_neuron_num, bias=False)
            self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

        def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()

            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
            return spk3, mem3

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
   * - speech_commands / gsc-MLP
     - SNN
     - 4000
     - 1 批
     - TruSynapse
     - 待补充
