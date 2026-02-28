液态神经网络（LNN）
=========

模型概述
--------

液态神经网络（LNN）是一种新型的神经网络架构，旨在通过模拟液态物质的动态特性来提高信息处理能力。与传统神经网络不同，LNN 通过引入时间和空间的连续性，使得网络能够在更高维度上进行学习和推理。这种方法特别适用于处理复杂的时序数据和动态系统。

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

   图一：LNN网络结构示例



实现代码
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

              # I神经元的输入：来自外部输入

              input_I = self.input_to_I(spike_input[t])

              spk_I, mem_I = self.neuron_I(input_I, mem_I)
            
              # M神经元的输入：来自上一个时间步的I神经元

              input_M = self.I_to_M(spk_I_prev)

              spk_M, mem_M = self.neuron_M(input_M, mem_M)
            
              # N神经元的输入：来自上一个时间步的M神经元

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

    connections = connection_trans(weight_matrix, input_size, input_neuron_ids, output_neuron_ids,stride=(1, 1), padding=(0, 0), dilation=(1, 1))

    with open("/your/path/to/input.txt") as file:

         inputdata = file.readlines()

    def main():

         data = functional.framework(net,connections,inputdata)

         net_output = functional.run(data)
    
    if __name__ == "__main__":

         main()

运行结果
--------

TODO: 添加运行结果
