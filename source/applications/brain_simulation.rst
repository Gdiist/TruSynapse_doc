脑仿真
======

应用概述
--------

本示例模拟的眼线虫颈部神经，网络按照神经元类型不同可分为三层：

第一层：包含I1L、I1R、I2L、I2R、I3、I4、I5、I6共8个神经元。

第二层：包含M1、M2L、M2R、M3L、M3R、M4、M5、MCL、MCR、MI共10个神经元。

第三层：包含NSML、NSMR共2个神经元。

每层都与包括其自身的全部三层网络连接，共9组连接关系：

第1组：连接第一层及其自身，稀疏连接。

第2组：连接第一层及第二层，稀疏连接。

第3组：连接第一层及第三层，稀疏连接。

第4组：连接第二层及第一层，稀疏连接。

第5组：连接第二层及其自身，稀疏连接。

第6组：连接第二层及第三层，稀疏连接。

第7组：连接第三层及第一层，稀疏连接。

第8组：连接第三层及第二层，稀疏连接。

第9组：连接第三层及其自身，稀疏连接。

.. figure:: ../_static/images/net_exm_brain.png
   :alt: LNN 
   :width: 50%
   :align: center

   眼线虫颈部神经网络结构示例

实现方案
--------

.. code-block:: python
    :linenos:

    from snntorch import functional

    import snntorch as snn

    import torch.nn as nn

    import pickle

    #网络与连接关系均可由用户自定义，包括神经网络层数、各层神经元数以及连接方式，示例为眼线虫颈部神经模拟

    betaI = 0.8

    betaM = 0.9

    betaN = 1.0

    threshold = 1.0

    num_input = 10

    num_I = 8

    num_M = 10

    num_N = 2

    class WormNet(nn.Module):

       def __init__(self):

          super().__init__()  # 初始化父类

          # 神经元

          self.neuron_I = snn.Leaky(beta=betaI, threshold=threshold)

          self.neuron_M = snn.Leaky(beta=betaM, threshold=threshold)

          self.neuron_N = snn.Leaky(beta=betaN, threshold=threshold)
        
          # 化学+电连接

          self.input_to_I = nn.Linear(num_input, num_I, bias=False)  # 输入到I

          self.I_to_M = nn.Linear(num_I, num_M, bias=False)           # I到M

          self.M_to_N = nn.Linear(num_M, num_N, bias=False)           # M到N

      def forward(self, spike_input):  # 前向推理

          time_steps = spike_input.shape[0]  # 时间步长
        
          mem_I = self.neuron_I.init_leaky()  # I神经元膜电位

          mem_M = self.neuron_M.init_leaky()  # M神经元膜电位

          mem_N = self.neuron_N.init_leaky()  # N神经元膜电位
        
          spk_I_prev = torch.zeros(num_I)  # I神经元上一个时间步的脉冲

          spk_M_prev = torch.zeros(num_M)  # M神经元上一个时间步的脉冲

          spk_N_prev = torch.zeros(num_N)  # N神经元上一个时间步的脉冲
        
          spk_I_rec = []  # I神经元脉冲记录

          spk_M_rec = []  # M神经元脉冲记录

          spk_N_rec = []  # N神经元脉冲记录

          for t in range(time_steps):  # 按时间步推理

              # I神经元的输入: 来自外部输入

              input_I = self.input_to_I(spike_input[t])

              spk_I, mem_I = self.neuron_I(input_I, mem_I)
            
              # M神经元的输入: 来自上一个时间步的I神经元

              input_M = self.I_to_M(spk_I_prev)

              spk_M, mem_M = self.neuron_M(input_M, mem_M)
            
              # N神经元的输入: 来自上一个时间步的M神经元

              input_N = self.M_to_N(spk_M_prev)

              spk_N, mem_N = self.neuron_N(input_N, mem_N)
            
              # 记录脉冲

              spk_I_rec.append(spk_I)

              spk_M_rec.append(spk_M)

              spk_N_rec.append(spk_N)
            
              # 更新上一个时间步的脉冲

              spk_I_prev = spk_I

              spk_M_prev = spk_M

              spk_N_prev = spk_N
        
          return torch.stack(spk_I_rec), torch.stack(spk_M_rec), torch.stack(spk_N_rec)

    connections = linear_connections_trans(weight_matrix, input_size, input_neuron_ids, output_neuron_ids)

    with open("/your/path/to/input.txt") as file:

         inputdata = file.readlines()

    def main():

         data = functional.framework(net,connections,inputdata)

         net_output = functional.run(data)
    
    if __name__ == "__main__":

         main()

使用NFU测试推理结果
^^^^^^^^^^^^^^^^^^^^^^^

NFU推理过程详细输出：

.. code-block:: text

    输出层神经元到GNC的映射:
    {26: 0, 27: 0}

    Time(ns) | Raw_Hex  | Timestamp | GNC | Neuron | Note
    ---------|----------|-----------|-----|--------|----------
    943490   | 0006001b |       3   |  0  |    27  | Data
    943590   | 0006001a |       3   |  0  |    26  | Data
    990490   | 0008001b |       4   |  0  |    27  | Data
    990730   | 0008001a |       4   |  0  |    26  | Data
    1040930  | 000a001b |       5   |  0  |    27  | Data
    1041170  | 000a001a |       5   |  0  |    26  | Data
    1091370  | 000c001b |       6   |  0  |    27  | Data
    1091610  | 000c001a |       6   |  0  |    26  | Data
    1141850  | 000e001b |       7   |  0  |    27  | Data
    1142090  | 000e001a |       7   |  0  |    26  | Data
    1193650  | 0010001b |       8   |  0  |    27  | Data
    1193890  | 0010001a |       8   |  0  |    26  | Data
    1244290  | 0012001b |       9   |  0  |    27  | Data
    1244530  | 0012001a |       9   |  0  |    26  | Data
    1297450  | 0014001b |      10   |  0  |    27  | Data
    1297690  | 0014001a |      10   |  0  |    26  | Data
    1347930  | 0016001b |      11   |  0  |    27  | Data
    1348170  | 0016001a |      11   |  0  |    26  | Data
    1399730  | 0018001b |      12   |  0  |    27  | Data
    1399970  | 0018001a |      12   |  0  |    26  | Data
    1450370  | 001a001b |      13   |  0  |    27  | Data
    1450610  | 001a001a |      13   |  0  |    26  | Data
    1504990  | 001c001b |      14   |  0  |    27  | Data
    1505230  | 001c001a |      14   |  0  |    26  | Data
    1556830  | 001e001b |      15   |  0  |    27  | Data
    1557070  | 001e001a |      15   |  0  |    26  | Data
    1610030  | 0020001b |      16   |  0  |    27  | Data
    1610270  | 0020001a |      16   |  0  |    26  | Data
    1661830  | 0022001b |      17   |  0  |    27  | Data
    1662070  | 0022001a |      17   |  0  |    26  | Data
    1712370  | 0024001b |      18   |  0  |    27  | Data
    1712610  | 0024001a |      18   |  0  |    26  | Data

**推理结果分析：**

- **神经元映射**: 输出层神经元共有两个，分别映射至0号GNC的26、27号神经元
- **神经元激活**: 在输入数据批次对应的时间步内，输出层神经元均进行了发放


