API 文档
========

本部分提供详细的 API 参考文档。

.. toctree::
   :maxdepth: 2

核心 API
--------

(1) address(length,output_layer_length,input_layer_length,BATCH_SIZE,BATCH) :

.. code-block:: bash
    :linenos:

    功能：为连接关系数据、输出数据、输入数据、输出层神经元数据、输入层神经元数据分配空间用于后续保存相关数据
    输入：
	 'length' :连接关系数据规模
	 'output_layer_length' :输出层数据规模
	 'input_layer_length' :输入层数据规模
	 'BATCH_SIZE' :每批次输入数据长度
	 'BATCH' :输入数据批次
    输出：各段数据空间的起始与结束地址（基地址为0）

(2) address_list_init(address_data_list) :

.. code-block:: bash
    :linenos:

    功能:将数据列表依据地址空间分配列表进行初始化
    输入:
    'address_data_list' :地址空间分配列表
    输出:
    'mem_file_content' :初始化后的数据空间

(3) assign_neuron_ids_1d(net,input_size) :

.. code-block:: bash
    :linenos:

    功能：为—维神经网络中的每个神经元分配唯—的全局ID
    输入：
    'net' :脉冲神经网络模型
    'input_size' :代表输入纬度的元组（通道数， 长度）
    输出：
    'neuron_id_map' :神经元ID，各层神经元映射
    'total_neurons' :神经网络中所有神经元数量

(4) assign_neuron_ids_2d(net,input_size) :

.. code-block:: bash
    :linenos:

    功能：为二维神经网络中的每个神经元分配唯—的全局ID
    输入：
    'net' :脉冲神经网络模型
    'input_size' :代表输入纬度的元组（通道数， 高度， 宽度）
    输出：
    'neuron_id_map' :神经元ID，各层神经元映射
    'total_neurons' :神经网络中所有神经元数量

(4) assign_neuron_ids_2d(net,input_size) :

.. code-block:: bash
    :linenos:

    功能：为三维神经网络中的每个神经元分配唯—的全局ID
    输入：
    'net' :脉冲神经网络模型
    'input_size' :代表输入纬度的元组（通道数， 深度， 高度， 宽度）
    输出：
    'neuron_id_map' :神经元ID，各层神经元映射
    'total_neurons' :神经网络中所有神经元数量

(5) assign_select(net,input_size) :

.. code-block:: bash
    :linenos:

    功能：依据输入尺寸选择合适的assign_neuron_ids函数
    输入：
    'net' :脉冲神经网络模型
    'input_size' :代表输入纬度的元组
    输出：调用对应的assign_neuron_ids函数，输出与对应函数输出—致

(6) build_connections(net,neuron_id_map,input_size) :

.. code-block:: bash
    :linenos:

    功能：构建神经网络内各个神经元之间的连接关系
    输入：
    'net' :脉冲神经网络模型
    'neuron_id_map' :神经元映射ID列表
    'input_size' :代表输入纬度的元组
    输出：
    'connections' :描述连接关系的三元组

(7) ComplexSNN(input_neu,hidden_neu,output_neu) :

.. code-block:: bash
    :linenos:

    功能：依据各层神经元的数量组成脉冲神经网络，用于后续构建连接关系以及神经元映射
    输入：
    'input_neu' :输入层神经元数量
    'hidden_neu' :隐藏层神经元数量
    'output_neu' :输出层神经元数量
    输出：
    'net' :根据输入要求形成的脉冲神经网络

(8) class GNC(gnc_id,neuron_id) :

.. code-block:: bash
    :linenos:

    功能 ：GNC类，用于创建GNC和放置神经元
    参数：
    'gnc_id' :gnc ID号
    'neuron_id' :神经元ID号




工具函数
--------

TODO: 添加工具函数文档

配置选项
--------

TODO: 添加配置选项文档
