数据准备
========

需要准备的数据包括：神经网络、网络训练权重及输入数据

神经网络定义
------------

首先需要对神经网络模型进行定义，比如一个MLP网络，定义示例如下：

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

权重文件
----------

权重文件通常是训练好的模型参数，保存为特定格式（如 .pkl 或 .pth 文件）。
这些文件包含了神经网络中每层的权重和偏置信息，供后续映射使用。



输入数据
--------

由于类脑芯片只支持0，1两种状态的输入，因此需要将原始输入数据（如图像、文本等）转换为二值化格式。
例如，对于图像数据，可以使用以下方法进行二值化： 

.. code-block:: python
    :linenos:

    import numpy as np

    def binarize_image(image, threshold=0.5):
        """将图像数据二值化。
        :param image: 输入图像，形状为 (C, H, W)
        :param threshold: 二值化阈值
        :return: 二值化后的图像，形状为 (C, H, W)
        """
        return (image > threshold).astype(np.uint8)

我们也提供了函数 ``convert_mnist_to_spike`` 来生成二值化的 MNIST 数据集，供用户直接使用。
该脚本会将 MNIST 数据集中的图像转换为二值化格式，并保存为 ``inputdata.txt``文件，方便后续加载和使用。

.. code-block:: python
    :linenos:

    import numpy as np
    from torchvision import datasets, transforms

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


