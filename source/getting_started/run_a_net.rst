==============
开始使用
==============
这里将介绍三种常见的使用场景

一、从头搭建神经网络
====================

概述
----
第一种使用场景是从头搭建一个神经网络，并将其映射到 NFU 上执行。
本示例展示如何使用 TruSynapse 框架将一个三层前向传播全连接神经网络映射到 NFU 并启动计算。
文档按流程分为网络构建、连接转换、输入准备、构造子网执行体与执行五个部分。

1. 构建神经网络（net）
----------------------
描述网络结构（层次、每层神经元数、必要的神经元/层参数等），并将网络信息保存为变量（例如 `net`），供后续映射使用。

示例：

.. code-block:: python
    :linenos:

    import snntorch as snn
    import torch.nn as nn
    from snntorch import surrogate
    class SNNMLP(nn.Module):
        def __init__(self, input_neuron_num=784, hidden1=512, hidden2=256, output_neuron_num=10, beta=0.9):
            super(SNNMLP, self).__init__()
            self.fc1 = nn.Linear(input_neuron_num, hidden1, bias=False)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

            self.fc2 = nn.Linear(hidden1, hidden2, bias=False)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

            self.fc3 = nn.Linear(hidden2, output_neuron_num, bias=False)
            self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()

            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
            return spk3, mem3

    net = SNNMLP()


2. 转换连接关系数据（connections）
----------------------------------
使用辅助函数将二维连接矩阵/张量转换为框架需要的三元组格式（src_id, dst_id, weight）。

示例函数：

.. code-block:: python
    :linenos:

    import numpy as np

    def connections_trans(connection_tensor, start1layer_ID, block_num):
        """
        将二维连接张量转换为三元组列表。
        connection_tensor: 2D tensor，形状为 (neuron_in_src_block, neuron_in_dst_block)
        start1layer_ID: 源层第一个神经元的全局 ID
        block_num: 子模块数量（重复块数）
        返回: triplets 列表，元素为 (src_id, dst_id, weight_float32)
        """
        triplets = []
        neuron_in_src_block, neuron_in_dst_block = connection_tensor.shape
        start2layer_ID = start1layer_ID + block_num * neuron_in_src_block
        for k in range(block_num):
            for i in range(neuron_in_src_block):
                for j in range(neuron_in_dst_block):
                    src_id = i + start1layer_ID + k * neuron_in_src_block
                    dst_id = j + start2layer_ID + k * neuron_in_dst_block
                    weight = np.float32(connection_tensor[i, j].item())
                    triplets.append((src_id, dst_id, weight))
        return triplets


    connection_file = "./mnist_snn.pkl"
    connection_origin_value = []
    with open(connection_file, 'rb') as pk:
        connection_value = pickle.load(pk)
    
    for key, value in connection_value.items():
        if 'weight' in key:
            connection_origin_value.append(value.T)
    
    for i in connection_origin_value:
        print(i.shape)
    
    connection_input = connections_trans(connection_origin_value[0], 0, 1)
    connection_hidden1 = connections_trans(connection_origin_value[1], 784, 1)
    connection_output = connections_trans(connection_origin_value[2], 1296, 1)
    connections = connection_input + connection_hidden1 + connection_output


3. 准备输入脉冲（inputdata）
-----------------------------
输入应为一维列表，元素为 0 或 1，长度等于输入层神经元数。


示例：

.. code-block:: python
    :linenos:
    
    from torchvision import datasets, transforms
    import numpy as np
    
    def convert_mnist_to_spike(n=100, thr=0.5, datapath='./data'):

        """Convert MNIST images to binary (spike) representation.
        :param n: int
        脉冲转化的样本数量，默认处理100个样本。
        :param thr: float
        脉冲转化的阈值，像素值大于该阈值将被视为脉冲（1），否则为非脉冲（0）（默认：0.5）。
        """
        test = datasets.MNIST(root=datapath, train=False, download=True,
                              transform=transforms.ToTensor())
        # 脉冲转化
        spike = [(test[i][0].view(-1) > thr).int().numpy() 
                  for i in range(min(n, len(test)))]

        inputdata = np.concatenate(spike)
        np.savetxt('inputdata.txt', inputdata, fmt='%d')
        return inputdata

    inputdata = convert_mnist_to_spike()  


4. 构造 NFU 子网执行体
-----------------------
使用框架接口将网络、连接和输入组合成可执行的数据结构（示例接口名使用 `functional.framework`）。

示例：

.. code-block:: python
    :linenos:

    from snntorch import functional

    data = functional.framework(net, connections, inputdata) 


5. 执行 NFU 子网并获取输出
---------------------------
调用运行接口执行 NFU 子网，并读取返回结果。

示例：

.. code-block:: python
    :linenos:

    net_output = functional.run(data) 

6. 处理输出结果
----------------
NFU的输出结果保存在输出spike空间，用户可以直接读取该空间的数据，也可以使用框架的工具进行转换。

NFU 直接输出结果以 32 位无符号整数表示，各字段含义如下：

.. list-table:: NFU 输出数据格式（32位）
   :header-rows: 1
   :align: center
   :widths: 20 20 60

   * - 位范围
     - 字段名
     - 说明
   * - [31:17] (15bit)
     - timestep
     - 时间步信息，表示该神经元输出是在哪个时间步产生的
   * - [16:13] (4bit)
     - GNC号
     - 输出层神经元所在的 GNC 编号
   * - [12:0] (13bit)
     - 逻辑ID
     - 该神经元的逻辑编号

我们提供了转换工具对输出结果进行整理，转换后的输出数据为一个二维数组，每个元素包含了所有输出层神经元信息，数组的索引代表 timestep 信息。

转换工具会统计该 timestep 所有输出层神经元发放脉冲的情况，若该 timestep 有发放则该神经元所在的位置为1，否则为 0。

示例：

.. code-block:: text

    假设一个神经网络有4个输出层神经元：

    timestep0时：2号神经元与4号神经元有输出

    timestep1时：一号神经元有输出

    timestep2时：所有输出层神经元均发放脉冲

    timestep3时：所有输出层神经元均不发放脉冲

    timestep4时：神经元1、2、4均发放脉冲

则转换后的输出结果为：

.. code-block:: text

    [[0,1,0,1],  # timestep0
     [1,0,0,0],  # timestep1
     [1,1,1,1],  # timestep2
     [0,0,0,0],  # timestep3
     [1,1,0,1]]  # timestep4

该输出会保存为outputdata供用户调用，用户可根据网络用途对NFU的输出进行处理。


二、直接导入已有网络
=====================
概述
----
对于已经训练好的神经网络，用户可以直接从多个文件中加载参数构造子网执行体，并调用NFU驱动进行执行。此方法无需网络处理步骤。

文件说明
----------
需要加载的文件说明如下:

.. list-table::
    :align: center

    * - 示例文件路径
      - 类型
      - 内容
    * - ./snn_data/inputspike.txt
      - 文本文件
      - 输入脉冲数据
    * - ./snn_data/connections.pkl
      -	| pickle生成的二进制文件
        | (也支持Pytorch的.pth文件)
      - 用户的SNN网络结构参数
    * - ./snn_data/neuron.data
      - 数据文件
      - 神经元模型参数数据
    * - ./snn_data/1_0.hdf5
      - HDF5文件
      - 处理后的SNN网络参数

其中，HDF5的文件来源有两种，一种是调用本框架的 ``net_process`` 模块生成自有格式的HDF5文件；第二种是外源的HDF5文件经过工具函数转换后得到（待实现）。

示例流程
---------
下面的流程图将展示如何调用 ``net_process`` 模块生成自有格式的HDF5文件，随后再加载该HDF5文件中的数据，并构造子网执行体进行计算。

.. figure:: ../_static/images/workflow.png
   :align: center
   :width: 80%
   :alt: 整体流程示意图

示例代码
---------

保存参数至自有格式的HDF5文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下面演示了一个完整的脉冲神经网络（SNN）处理流程，此流程会从文件中提取参数并进行处理，随后存至自有格式的HDF5文件中，主要包含以下两个步骤：

1. 定义一个三层前馈SNN网络（MnistSNN），包含两个全连接层和LIF神经元；
2. 用net_process()函数，将网络结构、连接权重和输入数据转换为HDF5格式的参数文件。

.. code-block:: python
    :linenos:

    import torch.nn as nn
    import snntorch as snn
    from snntorch import surrogate
    from net_process import net_process
    class MnistSNN(nn.Module):
        def __init__(self, input_neuron_num=4, hidden1=2, hidden2=2, output_neuron_num=3, beta=0.9):
            super(MnistSNN, self).__init__()
            self.fc1 = nn.Linear(input_neuron_num, hidden1, bias=False)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

            self.fc2 = nn.Linear(hidden1, hidden2, bias=False)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

            self.fc3 = nn.Linear(hidden2, output_neuron_num, bias=False)
            self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()

            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
            return spk3, mem3
    def main():

        # 实例化网络
        SNN_net = MnistSNN()
        # 如果想要存入hdf5的参数，可用变量获取net_process的返回值，如 paras = net_process(...)
        # 这里hdf5文件名经过路径校验后，会从"1.hdf5"变为"1_0.hdf5"
        net_process(SNN_net,connection_path="./snn_data/connections.pkl",inputdata_path="./snn_data/inputspike.txt",output_file_path="./snn_data/1.hdf5")

    if __name__ == "__main__":
        main()

从HDF5文件中加载参数并执行计算
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下面演示了从HDF5文件及其他文件中读取数据，并调用NFU驱动执行计算的流程，主要包含以下两个步骤：

1. 实例化 ``paras_process`` 类
2. 调用类中的 ``execute_computing`` 函数执行计算

输出脉冲说明
 - 注意: 输出脉冲列表中的首个“1”表示存在输出，此为标志位，并非实际的输出脉冲，实际输出脉冲应从列表第二个元素开始计算；
 - 输出脉冲的格式: 一个输出脉冲数据共32位，其中0~13为物理神经元号，14~17为GNC号，17~31为时间步数。

.. code-block:: python
    :linenos:

    from net_to_run import paras_process
    def main():
        
        # 从文件中解析参数并执行计算
        process = paras_process()
        # 获取计算结果
        source_results = process.execute_computing(spikes_in_path="./snn_data/inputspike.txt",
                                                neurondata_in_path="./snn_data/neuron.data",
                                                subnetsandparas_in_path = "./snn_data/1_0.hdf5",
                                                subnet_num = 1)
        # 打印计算结果
        print(source_results)
        
    if __name__ == "__main__":
        main()


三、搭建混合神经网络
======================
作为一款类脑CPU的框架，TruSynapse 除了支持常规的脉冲神经网络外，还能支持ANN/SNN混合神经网络。
用户可以将部分子网部署在 NFU 上执行，而其他子网继续在 CPU 上运行，从而实现性能与灵活性的平衡。

.. figure:: ../_static/images/hybrid_network.png
   :alt: Hybrid Neural Network
   :width: 80%
   :align: center

   ANN/SNN混合神经网络示例

应用场景示例
------------

1. 边缘智能监控
    - SNN：处理事件相机（DVS）流，提供毫秒级或更低延迟的实时检测与触发，功耗极低，适合全天候运行。
    - ANN：仅在触发时启动，进行高精度识别与行为/人脸分析，节省能耗。

2. 自动驾驶与机器人感知
    - SNN：处理激光雷达或事件相机的时序数据，快速响应紧急避障。
    - ANN：负责交通标志识别、路径规划与场景理解，承担复杂推理任务。

3. 脑机接口（BCI）
    - SNN：实时解码神经脉冲，实现超低延迟的反馈控制。
    - ANN：执行意图识别与高级指令映射。

4. 工业质检
    - SNN：在高速流水线上实时检测缺陷并触发剔除，延迟极低。
    - ANN：对缺陷进行分类与严重度评估，通常离线或异步运行以保证准确性。

5. 低功耗语音唤醒
    - SNN：常时监听唤醒词，功耗极低，提供即时唤醒信号。
    - ANN：在唤醒后开展语音识别与自然语言理解，完成复杂交互。

下面给出一个简单的示例，演示如何在 Trusynapse 中搭建一个混合神经网络。

.. code-block:: python
    :linenos:


    def basic_fork_example(self):
        try:
            pid = os.fork()

            if pid < 0:
                print("Fork失败!")
                return

            if pid == 0:
                # 子进程
                # 执行子进程任务
                print("子进程: 计数(1,1)")
                snn_data = save_all_parse_collect_to_struct()
                driver = SNNDriver()
                driver.execute(types.byte_snn_data)
                print("子进程: 任务完成")
                os._exit(0)  # 子进程退出
            else:
                matrix(a,b)
                # 父进程
                # 等待子进程
                pid_done, status = os.waitpid(pid, 0)



