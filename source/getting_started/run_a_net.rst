从头搭建神经网络
=================

概述
----
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

    def connection_trans(connection_tensor, start1layer_ID, block_num):
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
    
    connection_input = connection_trans(connection_origin_value[0], 0, 1)
    connection_hidden1 = connection_trans(connection_origin_value[1], 784, 1)
    connection_output = connection_trans(connection_origin_value[2], 1296, 1)
    connections = connection_input + connection_hidden1 + connection_output


3. 准备输入脉冲（inputdata）
-----------------------------
输入应为一维列表，元素为 0 或 1，长度等于输入层神经元数。


示例：

.. code-block:: python
    :linenos:
    

    def convert_mnist_to_spike(n=100, thr=0.5, datapath='./data'):

        """Convert MNIST images to binary (spike) representation.
        :param n: int
        数据批次大小，指定要处理的样本数量或每个输出批次中的样本数（默认：100）。
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
根据网络设计和应用需求，对输出结果进行处理。计算完成后的输出文件包括：

- `outputdata.txt`：保存网络输出的脉冲数据
- `performance_metrics.txt`：保存网络性能指标，如运行时间、延迟等

outputdata.txt文件为 ``[batch, output_neuron_num]`` 的二维文本文件，每列对应一个输入样本的输出脉冲序列。

用户需要进一步对输出数据进行解析和处理，以获得最终的推理结果或性能评估指标。


导入已训练神经网络
=================
概述
----
对于已训练好的神经网络（例如 Deepseek、千问 等），网络结构和参数来自外部资源，需构建 NFU 子网执行体（SNNData）并调用底层驱动执行。
下面示例按典型的 5 个步骤给出参考实现。

步骤（示例代码）
----------------

1. 实例化 NFU 子网执行体

.. code-block:: python
    :linenos:

    # 实例化一个 SNNData 子网执行体
    snn_data = SNNData()

2. 填充子网执行体各字段（示例）

.. code-block:: python
    :linenos:

    # 填充 27 个配置参数
    snn_data.params = (ctypes.c_uint64 * 27)(*[ctypes.c_uint64(x) for x in params])

    # 填充 connection_data
    snn_data.connection_data = (ctypes.c_uint64 * len(connection_data))(
        *[ctypes.c_uint64(x) for x in connection_data]
    )
    snn_data.connection_len = len(connection_data)

    # 填充 inputneuronlist_data
    snn_data.inputneuronlist_data = (ctypes.c_uint32 * len(inputneuronlist_data))(
        *[ctypes.c_uint32(x) for x in inputneuronlist_data]
    )
    snn_data.inputneuronlist_len = len(inputneuronlist_data)

    # 填充 inputspike_data
    snn_data.inputspike_data = (ctypes.c_uint32 * len(inputspike_data))(
        *[ctypes.c_uint32(x) for x in inputspike_data]
    )
    snn_data.inputspike_len = len(inputspike_data)

    # 填充 neuronbase_data
    snn_data.neuronbase_data = (ctypes.c_uint32 * len(neuronbase_data))(
        *[ctypes.c_uint32(x) for x in neuronbase_data]
    )
    snn_data.neuronbase_len = len(neuronbase_data)

    # 填充 neuron_data
    snn_data.neuron_data = (ctypes.c_uint32 * len(neuron_data))(
        *[ctypes.c_uint32(x) for x in neuron_data]
    )
    snn_data.neuron_data_len = len(neuron_data)

    # 初始化输出字段
    snn_data.output_data = None
    snn_data.output_len = 0

3. 执行子网

.. code-block:: python
    :linenos:

    driver = SNNDriver()
    driver.execute(ctypes.byref(snn_data))

4. 处理子网输出

.. code-block:: python
    :linenos:

    if snn_data.output_data and snn_data.output_len > 0:
        output_array = ctypes.cast(snn_data.output_data, POINTER(c_uint32 * snn_data.output_len))
        output_results = [output_array.contents[i] for i in range(snn_data.output_len)]

5. 回收子网执行体资源

.. code-block:: python
    :linenos:

    driver.free_output(ctypes.byref(snn_data))


搭建混合神经网络
===============
作为一款类脑CPU的框架，TruSynapse 除了支持常规的脉冲神经网络外，还能支持ANN/SNN混合神经网络。
用户可以将部分子网部署在 NFU 上执行，而其他子网继续在 CPU 上运行，从而实现性能与灵活性的平衡。

.. figure:: ../../_static/images/hybrid_network.png
   :alt: Hybrid Neural Network
   :width: 80%
   :align: center

   ANN/SNN混合神经网络示例

下面给出一个简单的示例，演示如何在 Trusynapse 中搭建一个混合神经网络。

1. 定义混合神经网络结构

.. code-block:: python
    :linenos:

    import snntorch as snn
    import torch.nn as nn

    class HybridNN(nn.Module):
        def __init__(self):
            super(HybridNN, self).__init__()
            self.ann = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )
            self.snn = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

        def forward(self, x):
            x = self.ann(x)
            x = self.snn(x)
            return x

2. 将 ANN 子网部署到 CPU 上执行，SNN 子网部署到 NFU 上执行

.. code-block:: python
    :linenos:

    to be continued...

