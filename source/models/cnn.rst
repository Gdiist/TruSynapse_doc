卷积神经网络 (CNN)
==================

模型概述
--------

本示例是用于MNIST数据集手写数字识别，示例包含4层网络，各层包含的神经元数量与连接关系为：

输入层：示例中包含784个神经元,表示一张MNIST数据集图片的输入。

隐藏层1:示例中输入通道数为1，特征图尺寸为28*28，输出通道数为16，特征图为28*28，与输入层之间为卷积连接，卷积核为3*3，步长为1，包含12544个神经元.

隐藏层2:示例中输入通道数为16，特征图为28*28，输出通道数为32，特征图为14*14，与隐藏层1之间为卷积连接，卷积核为3*3，步长为2，包含6272个神经元.

输出层：示例中包含10个神经元，表明将图像分为10类，与隐藏层2之间的连接方式为全连接。

所以：本示例总共包含19610个神经元，包含1030976条连接关系

实现代码
--------
.. code-block:: python
    :linenos:

    from snntorch import functional

    import snntorch as snn

    import torch.nn as nn

    import pickle

    #该示例中网络结构用户可以自定义，包括通道数、卷积核、步长等信息，以下为示例代码

    class MnistConvSNN(nn.Module):

        def __init__(self, beta=0.9):

            super(MnistConvSNN, self).__init__()

            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)

            self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)

            self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

            self.fc = nn.Linear(32 * 14 * 14, 10, bias=False)

            self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        def forward(self, x):

            mem1 = self.lif1.init_leaky()

            mem2 = self.lif2.init_leaky()

            mem3 = self.lif3.init_leaky()

            spk1, mem1 = self.lif1(self.conv1(x), mem1)          # [B,16,28,28]

            spk2, mem2 = self.lif2(self.conv2(spk1), mem2)       # [B,32,14,14]

            spk2_flat = spk2.view(spk2.size(0), -1)              # [B,6272]

            spk3, mem3 = self.lif3(self.fc(spk2_flat), mem3)     # [B,10]

            # 返回所有层 spk（以及最后一层 mem 方便算 pred）

            return spk1, spk2, spk3, mem3

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
