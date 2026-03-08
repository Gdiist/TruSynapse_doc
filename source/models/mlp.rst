多层感知机 (MLP)
================

模型概述
--------

该示例任务是对MNIST手写数字数据集图片进行分类，模型总共分为四层，层与层之间的连接为全连接的方式：

输入层：本示例包含784个神经元，表示一张MNIST数据集图片的输入

隐藏层1:本示例包含512个神经元，与输入层之间为全连接的方式进行连接

隐藏层2:本示例包含256个神经元，与隐藏层1之间为全连接的方式进行连接

输出层：本示例包含10个神经元，与隐藏层2之间为全连接的方式进行连接

所以：该示例总共包含1562个神经元，总连接数为535040条

实现代码
--------
.. code-block:: python
    :linenos:

    from snntorch import functional

    import snntorch as snn

    import torch.nn as nn

    import pickle

    #网络包含的网络层数以及各层神经元数量均为用户自定义，此示例为四层网络，各层神经元数量为784，512，256，10.
    
    class MnistSNN(nn.Module):

        def __init__(self, input_neuron_num=784, hidden1=512, hidden2=256, output_neuron_num=10, beta=0.9):

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

    net = MnistSNN(input_neuron_num=784, hidden1=512, hidden2=256, output_neuron_num=10, beta=0.9)

    connections1 = linear_connections_trans(weight_matrix, input_size, input_neuron_ids, output_neuron_ids)

    connections2 = linear_connections_trans(weight_matrix, input_size, input_neuron_ids, output_neuron_ids)

    connections3 = linear_connections_trans(weight_matrix, input_size, input_neuron_ids, output_neuron_ids)

    connections = connections1 + connections2 + connections3
    
    with open("/your/path/to/inputdata.txt") as file:

        inputdata = file.readlines()

    def main():

        data = functional.framework(net,connections,inputdata)

        net_output = functional.run(data)
    
    if __name__ == "__main__":

        main()

运行结果
--------

TODO: 添加运行结果
