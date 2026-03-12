数据准备
========

概述
-------

需要准备的数据根据使用方式有一定差异。

* 对于从头搭建神经网络的方式，需要准备的数据包括:

.. list-table::
    :align: center

    * - 示例名
      - 类型
      - 内容
    * - net
      - python对象
      - 神经网络结构
    * - connections.pkl
      -	| pickle生成的二进制文件
        | (也支持Pytorch的.pth文件)
      - 用户的SNN网络的权重文件
    * - inputspike.txt
      - 文本文件
      - 输入脉冲数据
    * - neuron.data
      - 数据文件
      - 神经元模型参数数据

.. _label_InputFilesIntro:

* 如果直接导入已有网络，则需要准备以下文件：

.. list-table::
    :align: center

    * - 示例文件名
      - 类型
      - 内容
    * - connections.pkl
      -	| pickle生成的二进制文件
        | (也支持Pytorch的.pth文件)
      - 用户的SNN网络的权重文件
    * - inputspike.txt
      - 文本文件
      - 输入脉冲数据
    * - neuron.data
      - 数据文件
      - 神经元模型参数数据
    * - subnet_data_0.hdf5
      - HDF5文件
      - 处理后的SNN网络参数

以下是对表格内所涉及的数据/文件的详细介绍。

神经网络定义 (net对象)
---------------------------

首先需要对神经网络模型进行定义，目前主要支持使用snntorch定义网络，实例化的网络文件包含了每层的类型、参数以及连接关系等信息，供后续映射使用。其他框架的网络可以通过转换为NIR格式后使用。
下面是一个SNN_MLP网络的定义示例:

.. code-block:: python
    :linenos:

    import snntorch as snn
    import torch.nn as nn
    class SNN_MLP(nn.Module):
        def __init__(self, input_neuron_num=784, hidden1=512, hidden2=256, output_neuron_num=10, beta=0.9):
            super(SNN_MLP, self).__init__()
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
    
    # 实例化网络
    net = SNN_MLP()

.. 其他框架的网络通过NIR转换后的使用示例如下：
.. 
.. .. code-block:: python
..     :linenos:
.. 
..     import nir
..     import nirtorch
.. 
..     to be continued...


权重文件 (connections.pkl)
----------------------------------

权重文件通常是训练好的模型参数，保存为特定格式（如 .pkl 或 .pth 文件）。
这些文件包含了神经网络中每层的权重信息，供后续映射使用。可以通过加载权重文件并进行必要的转换来获取连接关系数据（connections），以适配 TruSynapse 框架的输入要求。

.. code-block:: python
    :linenos:

    connection_file = "./connection.pkl"
    
    def load_connection(connection_file):
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
        return connections

    load_connection(connection_file)

输入脉冲数据 (inputspike.txt)
-------------------------------------

由于类脑芯片只支持0，1两种状态的输入，因此需要将原始输入数据（如图像、文本等）转换为脉冲化格式，并以一维数组的形式进行保存。
常用的方式包括率编码、时间编码等，具体方法取决于输入数据的特性和应用需求。也可以使用 ``snntorch.spikegen`` 中内置的编码器将其转换为脉冲序列。

输入脉冲数据需存储为纯文本文件（.txt 格式），且需满足以下格式要求：
 - 每行仅包含一个脉冲信号值（0 或 1），无任何标点符号、分隔符、空白字符；
 - 文件首部、尾部均无空行，行内无前置 / 后置空格、制表符等无效字符；
 - 文本编码采用 UTF-8，避免特殊字符导致解析异常。

下面的脚本介绍了一种直接进行阈值判断将 MNIST 数据集中的图像转换为二值化格式的做法，最后输出的 ``inputdata.txt`` 文件作为后续输入直接加载和使用。

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



神经元模型 (neuron.data)
-----------------------------

神经元模型文件 ``neuron.data`` 是 NFU 执行神经元计算所需的程序指令。该文件由汇编指令转换生成，定义了神经元的更新逻辑、事件处理和脉冲生成等行为。
本框架中默认提供了 LIF 神经元模型的机器码，用户也可以根据需要自定义神经元模型并生成对应的 ``neuron.data`` 文件。具体可参考 :ref:`高级使用中的神经元模型定义 <neuron_model_definition>` 和文件生成流程部分。


以下是系统默认的神经元模型汇编指令 ``neuron.txt`` 示例：

.. code-block:: text
   :linenos:

    LUI a11 0
    ST a11 a0 1
    FLD fa1 a0 1        //v_reset
    LUI a11 16256
    ST a11 a0 2
    FLD fa2 a0 2        //v_th
    LUI a1 16
    LUI a2 16
    LUI a3 0            //spike 
    LUI a4 1            //local timestep
    LUI a5 3            //global timestep
    LUI a11 16248
    ST a11 a0 3
    FLD fa3 a0 3        //decay
    BRE a1 a2
    LUI a0 0
    BEQ 24 a16 a3       //only spike
    BEQ 26 a16 a4       //local timestep
    BEQ 21 a16 a5       //global timestep
    FMUL fa27 fa17 fa3  //Decay
    BRE a1 a2
    ADDI a26 a0 0       //global timestep generate spike
    FADD fa27 fa1 fa0   //eventcompl
    BRE a1 a2
    FADD fa27 fa16 fa17 //only spike事件处理 v=v+w&&eventcompl
    BRE a1 a2           
    FADD fa6 fa16 fa17  //local timestep
    FLT 32 fa6 fa2      
    ADDI a16 a3 0       //spike
    ADDI a26 a0 0       //generate spike
    FADD fa27 fa1 fa0   //v=v_reset&&eventcompl
    BRE a1 a2
    FADD fa27 fa6 fa0   //v=v+w&&eventcompl
    BRE a1 a2
    LUI a0 0
    LUI a0 0
    LUI a0 0
    LUI a0 0


可以通过调用 API 将汇编指令转换为 ``neuron.data`` 二进制机器码文件：

.. code-block:: python
    :linenos:

    from asm2bin import assemble_file

    #文件转换
    assemble_file("neuron.txt", "neuron.data")

相关 API 参考：:ref:`assemble_file <api_assemble_file>`



.. _label_自有格式HDF5文件:

自有格式HDF5 (subnet_data.hdf5)
----------------------------------------

自有格式HDF5文件存储了子网执行体的相关信息，按子网划分，每个子网均存储了NFU进行计算时所需的多种参数。这些参数大体可分为子网执行体参数和NFU硬件参数，具体如下：

* ``connection_data``: SNN网络的连接权重数据，存储了神经网络中所有神经元之间的连接关系和权重信息；
* ``inputneuronlist_data``: 输入神经元映射列表数据，用于定义输入数据如何映射到NFU的输入神经元；
* ``neuronbase_data``: 神经元状态空间数据，存储SNN网络中每个神经元的状态；
* ``register_paras``: NFU的寄存器参数，为硬件参数，如下：

    * ``neuronbase_param``: 神经元状态空间区域的偏移量，表示神经元基地址相对于物理内存起始地址的偏移；
    * ``connection_param``: 连接权重数据区域的偏移量，表示连接数据区域相对于物理内存起始地址的偏移；
    * ``inputspike_param``: 输入脉冲数据的偏移量，表示输入脉冲数据区域相对于物理内存起始地址的偏移；
    * ``outputspike_param``: 输出脉冲数据的偏移量，表示输出脉冲数据区域相对于物理内存起始地址的偏移；
    * ``inputneulist_param``: 输入神经元列表的偏移量，表示输入神经元列表相对于物理内存起始地址的偏移；
    * ``hibase_addr_data``: 高基地址数据，用于设置NFU的高位地址基准；
    * ``startpc_addr_data``: 启动程序计数器的地址，用于设置NFU的程序执行起始地址；
    * ``gnc_layercnt_addr_data``: 是一个包含16个元素的列表，每个元素对应一个GNC的相关配置信息。这16个参数用于配置每个GNC的最后一层神经元计数（该GNC输出层的神经元数量）、输出指针信息（数据输出的位置信息）、父节点数量（有多少个上级GNC会向此GNC发送数据）、GNC内部神经元计数（该GNC内部的神经元总数）；
    * ``mode_addr_data``: 模式配置数据，用于设置NFU的工作模式和衰减同步模式，其综合了输入层配置、衰减模式和同步控制多种配置；
    * ``regbroadcast_addr_data0``: 寄存器广播配置数据，用于控制NFU的寄存器广播操作，为NFU的底层硬件控制参数，其确保各个组件能够正确接收和执行配置指令；
    * ``vth_batch_addr_data``: 阈值和批处理配置数据，用于设置NFU的激活阈值和批处理参数，其直接影响处理器的计算精度和处理能力；
    * ``runstep_addr_data``: 运行步骤配置数据，用于控制NFU的运行时序和步骤参数，其综合了运行时的关键控制信息，确保NFU能够按照预定的时序和配置正确执行计算任务。

.. list-table::
  :align: center
  :class: top-align-table excel-style-table
  
  * - 自有格式HDF5文件参数存储形式        
    - 自有格式HDF5文件实际文件结构
  * -
    -
  * - .. figure:: /_static/images/hdf5_structure.png
          :alt: hdf5_structure
    - .. figure:: /_static/images/hdf5_file.png
          :alt: hdf5_file

