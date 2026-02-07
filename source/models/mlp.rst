多层感知机 (MLP)
================

模型概述
--------

该示例任务是对MNIST手写数字数据集图片进行分类，模型总共分为四层：

输入层：包含784个神经元，一个批次接收784个神经脉冲

隐藏层1:包含512个神经元，与输入层之间为全连接的方式进行连接

隐藏层2:包含256个神经元，与隐藏层1之间为全连接的方式进行连接

输出层：包含10个神经元，与隐藏层2之间为全连接的方式进行连接

所以：该示例总共包含1562个神经元，总连接数为535040条

实现代码
--------

1、网络描述

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

2、连接关系导入

连接关系以pickle文件形式给出，其内包含多个字典，表征层与层之间的连接关系与连接权重：

    connection_file = "/your/path/to/connection.pkl"

    connection_origin_value = []

    with open(connection_file, 'rb') as pk:

        connection_value = pickle.load(pk)
    
    for key, value in connection_value.items():

        if 'weight' in key:

            connection_origin_value.append(value.T)

    connection_input = connection_trans(connection_origin_value[0], 0, 1)

    connection_hidden1 = connection_trans(connection_origin_value[1], 784, 1)

    connection_output = connection_trans(connection_origin_value[2], 1296, 1)

    connections = connection_input + connection_hidden1 + connection_output

3、输入数据准备

输入数据只能包含0或1，以txt文件给出：

    with open("/your/path/to/inputdata.txt") as file:

        inputdata = file.readlines()

4、调用框架启动NFU

    data = functional.framework(net,connections,inputdata)

    net_output = functional.run(data)

总体代码：

    from snntorch import functional

    import snntorch as snn

    import torch.nn as nn

    import pickle
    
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
    
    connection_file = "/your/path/to/connection.pkl"

    connection_origin_value = []

    with open(connection_file, 'rb') as pk:

        connection_value = pickle.load(pk)

    for key, value in connection_value.items():

        if 'weight' in key:

            connection_origin_value.append(value.T)

    connection_input = connection_trans(connection_origin_value[0], 0, 1)

    connection_hidden1 = connection_trans(connection_origin_value[1], 784, 1)

    connection_output = connection_trans(connection_origin_value[2], 1296, 1)

    connections = connection_input + connection_hidden1 + connection_output
    
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
