数据准备
========

需要准备的数据包括：神经网络、网络训练权重及输入数据

神经网络定义
------------

首先需要对神经网络模型进行定义，目前主要支持使用snntorch定义网络。其他框架的网络可以通过转换为NIR格式后使用。
下面是一个MLP网络的定义示例：

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

其他框架的网络通过NIR转换后的使用示例如下：

.. code-block:: python
    :linenos:

    import nir
    import nirtorch

    to be continued...



权重文件
----------

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

        connection_input = connection_trans(connection_origin_value[0], 0, 1)
        connection_hidden1 = connection_trans(connection_origin_value[1], 784, 1)
        connection_output = connection_trans(connection_origin_value[2], 1296, 1)
        connections = connection_input + connection_hidden1 + connection_output
    return connections

    load_connection(connection_file)

输入数据
--------

由于类脑芯片只支持0，1两种状态的输入，因此需要将原始输入数据（如图像、文本等）转换为脉冲化格式，并以一维数组的形式进行保存。
常用的方式包括率编码、时间编码等，具体方法取决于输入数据的特性和应用需求。可以使用 ``snntorch.spikegen`` 中内置的编码器将其转换为脉冲序列。

下面的脚本展示了如何将 MNIST 数据集中的图像转换为二值化格式，并保存为 ``inputdata.txt``文件，方便后续加载和使用。

.. code-block:: python
    :linenos:

    import numpy as np
    from torchvision import datasets, transforms

    mport torch
    rom torchvision import datasets, transforms
    mport numpy as np
    rom snntorch import spikegen

    def gen_mnist_spike(n=100, datapath='./data', filename='inputdata.txt'):
        """
        使用 spikegen.rate 将 MNIST 转换为脉冲并保存
        :param n: 处理的样本数量
        """
        test = datasets.MNIST(root=datapath, train=False, download=True,
                              transform=transforms.ToTensor())

        data_loader = torch.utils.data.DataLoader(test, batch_size=n)
        images, _ = next(iter(data_loader))

        spike_data = spikegen.rate(images, num_steps=1) 

        final_data = spike_data.view(-1).numpy().astype(int)

        np.savetxt(filename, final_data, fmt='%d')

        return final_data

    inputdata = gen_mnist_spike(n=100)


