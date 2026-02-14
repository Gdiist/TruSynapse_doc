SNN网络处理流程示例
====================================

本示例展示了如何使用Python通过调库完成神经网络提取、计算、结果处理的完整流程。

1 SNN网络约束规范
===========================

.. code-block:: python

    class SNN_Gesture_Model(nn.Module):
        def __init__(self, n_inputs, n_hidden1, n_hidden2, n_outputs):
            super().__init__()
            # 第一层：全连接 + LIF
            self.fc1 = nn.Linear(n_inputs, n_hidden1)
            self.lif1 = snn.Leaky(beta=BETA_HIDDEN, spike_grad=SPIKE_GRAD, learn_threshold=True)
            # 第二层：全连接 + LIF
            self.fc2 = nn.Linear(n_hidden1, n_hidden2)
            self.lif2 = snn.Leaky(beta=BETA_HIDDEN, spike_grad=SPIKE_GRAD, learn_threshold=True)
            # 输出层：全连接 + LIF（无重置机制），输出层不重置，累加膜电位
            self.fc3 = nn.Linear(n_hidden2, n_outputs)
            self.lif3 = snn.Leaky(beta=BETA_OUTPUT, spike_grad=SPIKE_GRAD, learn_threshold=True, reset_mechanism='none')

        def forward(self, x):
            # 初始化LIF神经元的膜电位
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
            mem3_rec = 0 # 用于累加输出层膜电位

            # 对每个时间步进行处理
            for step in range(x.size(0)):
                # 第一层：全连接 + LIF
                cur1 = self.fc1(x[step])  # [batch_size, n_hidden1]
                spk1, mem1 = self.lif1(cur1, mem1)  # 输出脉冲和更新后的膜电位
                # 第二层：全连接 + LIF
                cur2 = self.fc2(spk1)  # [batch_size, n_hidden2]
                spk2, mem2 = self.lif2(cur2, mem2)
                # 输出层：全连接 + LIF（累加膜电位）
                cur3 = self.fc3(spk2)  # [batch_size, n_outputs]
                spk3, mem3 = self.lif3(cur3, mem3)
                # 累加输出层膜电位
                mem3_rec += mem3

            return mem3_rec


1.1 网络实例化方式约束
---------------------------

网络结构仅支持通过顺序式手动声明的方式创建（即逐行显式定义各层结构，并按前向传播顺序完成网络拼接）；暂不支持基于 ``nn.ModuleList`` （或同类模块化列表容器）的动态组网方式。


1.2 偏置（Bias）配置约束
---------------------------

该 SNN 框架对神经元层偏置参数的支持范围严格限定为以下两类场景：

- 无偏置模式：网络内所有神经元层均不配置偏置参数；
- 层级统一偏置模式：若某神经元层启用偏置，则该层内所有神经元共享同一标量偏置值（即偏置参数维度为标量，而非与神经元数量匹配的向量）；

暂不支持 “每层神经元独立配置不同偏置值” 的场景。


1.3 神经元层类型约束
---------------------------

网络中仅允许集成以下两类神经元层，不支持其他层类型：

- **线性层（Linear Layer）**：全连接类型的神经元层，需与神经元计算逻辑强关联，暂不支持无神经元关联的纯线性变换层（如动作头、价值头等，二者仅承担数值映射功能，无 SNN 神经元核心计算逻辑）；
- **卷积层（Convolutional Layer）**：包含 1D/2D/3D 卷积层（根据框架适配范围）；

暂不支持池化层、激活层、循环层、归一化层等其他类型的神经元层。


1.4 输入数据规范
---------------------------

***************************
1.4.1 输入值类型约束
***************************

输入脉冲信号仅支持二值离散型数据，取值范围严格限定为 {0, 1}；不支持连续数值型输入（如浮点型、整型非 0/1 值、复数型等）。

***************************
1.4.2 输入脉冲文件存储格式约束
***************************

输入脉冲信号需存储为纯文本文件（``.txt`` 格式），且需满足以下格式要求：

- 每行仅包含一个脉冲信号值（0 或 1），无任何标点符号、分隔符、空白字符；
- 文件首部、尾部均无空行，行内无前置 / 后置空格、制表符等无效字符；
- 文本编码采用 UTF-8，避免特殊字符导致解析异常。


2 调用流程
===========================

.. figure:: ../_static/workflow.png
   :align: center
   :alt: SNN执行流程示意图

   

2.1 定义 SNN 网络结构
---------------------------

- 配置参数：根据用户的网络结构，完成对应配置参数的定义与赋值；
- 实例化网络：基于已完成配置的网络参数，完成 SNN 网络对象的实例化构建。

---------------------------
2.2 数据提取与转换
---------------------------

***************************
2.2.1 调用框架函数提取并处理网络数据
***************************

.. code-block:: python

    framwork( net,
              connection_path="./snn_data/snn_policy.pth",
              inputdata_path="./snn_data/inputspike.txt",
              output_file_path="./snn_data/navigation.hdf5")

- ``net`` : 输入，为已完成实例化的 SNN 网络对象。
- ``snn_policy.pth`` : 输入，存储 SNN 网络结构信息（含 ``.weight`` 类权重数据）的文件，支持 pkl、pth 格式；
- ``inputspike.txt`` : 输入，为SNN网络的输入脉冲文件，每行代表一个脉冲事件；
- ``navigation.hdf5`` : 输出，存储 NFU所需数据的文件（包含子网参数、结构信息等），供后续 C 库解析，默认采用 hdf5 格式存储。

调用执行 ``framwork(...)``，该函数会从已定义的 SNN 网络中提取权重、阈值、连接关系等参数，处理后生成NFU所需的数据，并保存在hdf5文件中。

***************************
2.2.2 参数加载与数据转换
***************************

.. code-block:: python

    convert = paras_process()    
    snn_data = convert.parse_collect_to_struct(
                                spikes_in_path="./snn_data/inputspike.txt",
                                neurondata_in_path="./snn_data/neuron.data",
                                subnetsandparas_in_path = "./snn_data/navigation.hdf5",
                                subnet_num = -1,
                                subnet_paras_name = "all")    

- ``paras_process()``: 创建参数处理类实例
- ``parse_collect_to_struct()``: 从文件中加载SNN数据并转换成C结构体：
  - ``spikes_in_path``: 输入，为SNN网络的输入脉冲文件，每行代表一个脉冲事件
  - ``neurondata_in_path``: 输入，为存储神经元模型数据的文件
  - ``subnetsandparas_in_path``: 输入，存储 NFU所需数据的文件
  - ``subnet_num``: 需要调用的子网编号（-1表示所有子网）
  - ``subnet_paras_name``: 子网内需要调用的参数名称（默认全部）

调用 ``paras_process().parse_collect_to_struct()``，读取上述文件，将其转换为 ``snn_data`` 结构体（C 兼容的内存布局），该结构体将直接传递给 SNN 驱动执行计算。

---------------------------
2.3 SNN计算执行
---------------------------

.. code-block:: python

    driver = SNNDriver(lib_path='./libsnndriver.so')    
    driver.execute(ctypes.byref(snn_data))    

- ``SNNDriver()``: 初始化SNN驱动，加载C语言动态链接库（``libsnndriver.so``）
- ``execute()``: 执行SNN计算，传入数据结构指针
- ``snn_data``: 转换后的结构体参数
- 结果输出：nfu计算返回的原始结果保存在 ``snndata.outputdata`` 数组里  

===========================
3 数据文件说明
===========================

+-------------------------------+-------------+----------------------------------------+
| 文件路径                       | 类型        | 内容                                   |
+===============================+=============+========================================+
| ./snn_data/policy.pth         | torch文件   | 用户的SNN网络结构参数                   |
|                               | (也支持.pkl)|                                        |
+-------------------------------+-------------+----------------------------------------+
| ./snn_data/inputspike.txt     | 文本文件    | 输入脉冲数据                            |
+-------------------------------+-------------+----------------------------------------+
| ./snn_data/neuron.data        | 数据文件    | 神经元模型参数数据                       |
+-------------------------------+-------------+----------------------------------------+
| ./snn_data/navigation.hdf5    | HDF5文件    | 处理后的SNN网络结构参数                  |
+-------------------------------+-------------+----------------------------------------+

===========================
4 输出说明
===========================

- **输出格式**: 32位无符号整数列表
- **控制台显示示例**: 

.. code-block:: text

    [警告] 目前仅支持一次性计算一个子网! 若文件内有多个子网，选择导出全部子网, 则会默认导出第一个子网!
    [提示] 该hdf5文件里根目录下Group数量: 4
    [提示] hdf5读取成功, 从 ./snn_data/1_0.hdf5 加载 1 个子网的变量
    执行SNN计算...
    === SNN硬件初始化 ===
    已打开驱动设备: /dev/accelerator
    ✅ Hugepage file opened
    Hugepage file size: 1073741824 bytes
    Mapping 1GB hugepage...
    ✅ Successfully mapped 1GB HugePage at virtual: 0xffff40000000
    First-touch读取数据: 0x3f00000000000000
    硬件初始化完成
    connection数据写入完成
    inputneuronlist数据写入完成
    inputspike数据写入完成
    neuronbase数据写入完成
    neuron数据写入完成
    开始配置寄存器...
    物理地址: 0x0000000c00000000
    物理地址: 0xc00000000
    开始执行SNN计算...
    SNN计算完成
    开始探测输出结果长度...
    输出基地址: 0x200100
    探测到输出结果数量: 10
    复制数据中...
    成功读取 10 个输出结果
    复制后的前10个输出结果: 1 655369 917514 917513 786442 786441 786440 1179658 1179657 1179656
    清理硬件资源...
    硬件资源清理完成
    处理输出结果...
       输出脉冲: 15 个
       全部输出: [1, 655369, 917514, 917513, 786442, 786441, 786440, 1179658, 1179657, 1179656, 13246849, 13246848, 13377926, 13377921, 13377920]
       输出内存已释放
    === SNN测试完成 ===

- **输出脉冲说明**: 

  - 注意: 输出脉冲列表中的首个“1”表示存在输出，此为标志位，并非实际的输出脉冲，实际输出脉冲应从列表第二个元素开始计算
  - 输出脉冲的格式: 一个输出脉冲数据共32位，其中0~13为物理神经元号，14~17为GNC号，17~31为时间步数
  - 输出结果转换参考代码

.. code-block:: python

    def fun1(lista: list):
        print("*"*10,f"开始统计:","*"*10)
        for num in lista:
            physical_num = (num & 0x1fff)
            GNC_num = (num >> 13) & 0xf
            time_step = (num >> 17)
            if(lista[0] == 1)and(physical_num == 1)and(GNC_num == 0)and(time_step == 0):
                print("有输出！")
            else:
                print(f"{time_step, GNC_num,physical_num}, 在第{time_step}时间步，{GNC_num}号GNC的{physical_num}号神经元发放了一次")
        print("*"*10,f"总共有{len(lista)}个输出","*"*10)

    def main():
        h = [1, 524298, 524297, 524296, 786442, 786441, 786440, 1179658, 1179657, 1179656]
        fun1(h)

    if __name__ == '__main__':
        main()