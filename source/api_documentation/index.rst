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

(5) assign_neuron_ids_2d(net,input_size) :

.. code-block:: bash
    :linenos:

    功能：为三维神经网络中的每个神经元分配唯—的全局ID
    输入：
    'net' :脉冲神经网络模型
    'input_size' :代表输入纬度的元组（通道数， 深度， 高度， 宽度）
    输出：
    'neuron_id_map' :神经元ID，各层神经元映射
    'total_neurons' :神经网络中所有神经元数量

(6) assign_select(net,input_size) :

.. code-block:: bash
    :linenos:

    功能：依据输入尺寸选择合适的assign_neuron_ids函数
    输入：
    'net' :脉冲神经网络模型
    'input_size' :代表输入纬度的元组
    输出：调用对应的assign_neuron_ids函数，输出与对应函数输出—致

(7) build_connections(net,neuron_id_map,input_size) :

.. code-block:: bash
    :linenos:

    功能：构建神经网络内各个神经元之间的连接关系
    输入：
    'net' :脉冲神经网络模型
    'neuron_id_map' :神经元映射ID列表
    'input_size' :代表输入纬度的元组
    输出：
    'connections' :描述连接关系的三元组

(8) ComplexSNN(input_neu,hidden_neu,output_neu) :

.. code-block:: bash
    :linenos:

    功能：依据各层神经元的数量组成脉冲神经网络，用于后续构建连接关系以及神经元映射
    输入：
    'input_neu' :输入层神经元数量
    'hidden_neu' :隐藏层神经元数量
    'output_neu' :输出层神经元数量
    输出：
    'net' :根据输入要求形成的脉冲神经网络

(9) class GNC(gnc_id,neuron_id) :

.. code-block:: bash
    :linenos:

    功能 ：GNC类，用于创建GNC和放置神经元
    参数：
    'gnc_id' :gnc ID号
    'neuron_id' :神经元ID号

(10) cluster(neuron_id_map, connections, relation=1) :

.. code-block:: bash
    :linenos:

	功能：基于连接关系将神经元聚类映射到NFU的GNC中（直接映射法）
	输入：
	'neuron_id_map' :神经元ID映射列表
	'connections'   :连接关系
	'relation'      :关系值,默认为1
	输出:
	'input_mapping'  :输入层神经元映射结果
	'output_mapping' :输出层神经元映射结果
	'nfu' :NFU使用率
	'longest_time_expression' ：网络中spike传输的最长时间

(11) cluster_input2center(neuron_id_map, connections, relation=1) :

.. code-block:: bash
    :linenos:

	功能 ：基于神经元的连接关系及预设的4x4网格结构，将其聚类映射至NFU的GNC
	输入：
	'neuron_id_map' :神经元ID映射列表
	'connections'   :连接关系
	'relation'      :关系值,默认为1
	输出:
	'input_mapping'  :输入层神经元映射结果
	'output_mapping' :输出层神经元映射结果
	'nfu' :NFU使用率
	'longest_time_expression' ：网络中spike传输的最长时间

(12) cluster2complex_single(neuron_id_map, connections) :

.. code-block:: bash
    :linenos:

	功能 ：基于连接关系将神经元聚类映射到NFU的GNC中（模拟退火法）
	输入：
	'neuron_id_map' :神经元ID映射列表
	'connections' :连接关系
	输出：
	'input_mapping' :输入层神经元映射结果
	'output_mapping' :输出层神经元映射结果
	'nfu' :NFU使用率
	'longest_time_expression' ：网络中spike传输的最长时间

(13) check_save_eq(save_file_name, net, sample_input_size) :

.. code-block:: bash
    :linenos:

	功能：检查是否存在与输入文件名称相同的文件，如果存在文件则尝试加载。 加载成功则对比文件内神经网络、输入尺寸与输入是否—致， —致则返回所有加载内容，不—致或其他异常则返回None
	输入：
	'save_file_name' :保存的文件名
	'net' :脉冲神经网络
	'sample_input_size' :网络输入尺寸
	输出：
	'save_data' :如果文件存在， 返回保存的数据
	'None' ：没有文件存在， 返回None

(14) check_64bit_align(mem_file_content,address_data_list) :

.. code-block:: bash
    :linenos:

	功能：检查分配的地址空间是否为64位对齐，若是则不变，若否，则将地址转换为64位对齐数据后存入地址分配空间
	输入：
	'mem_file_content' :数据空间列表
	'address_data_list' :地址分配列表
	输出：
	'addresses' :修改后的地址空间分配列表

(15) connection_data_load(connection_list,data_list) :

.. code-block:: bash
    :linenos:

	功能：将连接关系列表中的数据按照—定的规则放入数据列表
	输入：
	'connection_list' :连接关系列表
	'data_list' :整体数据列表
	输出：
	'data_list' :加入了连接关系数据的整体数据列表

(16) conv_connections_trans(weight_matrix, input_size, input_neuron_ids, output_neuron_ids,stride=(1, 1), padding=(0, 0), dilation=(1, 1)):

.. code-block:: bash
    :linenos:

	功能：将卷积层的连接矩阵转换为描述连接关系的三元组
	输入：
	'weight_matrix':四维权重矩阵，形状为(输出通道数，输入通道数，卷积核高，卷积核宽)
	'input_size':输入尺寸
	'input_neuron_ids':连接的输入层神经元ID列表
	'output_neuron_ids':连接的输出层ID列表
	'stride':卷积步长，默认为(1,1)
	'padding':填充值，默认为(0,0)
	'dilation':膨胀率，默认为(1,1)
	输出：
	'connections' :描述连接关系的三元组

(17) find_gnc_cycles(connections) :

.. code-block:: bash
    :linenos:

	功能：判断映射完GNC之间是否成环
	输入：
	'connections' :连接关系的三元组
	输出：
	'cycles'：形成的GNC环路，若无环则为空
	'gnc_graph'：成环的GNC图 
	'cycle_details':环路详细信息

(18) get_all_cycle_gnc_info(cycle_details):

.. code-block:: bash
    :linenos:

	功能：从环路详情中提取所有环路的GNC信息
	输入：
	'cycle_details':环路详细信息
	输出：
	'gnc_info':环路中GNC信息
	'gnccycle_key':各个环路的键值


(19) get_net_info(net) ：

.. code-block:: bash
    :linenos:

	功能：获取网络相关参数
	输入：
	‘net’ :脉冲神经网络模型
	输出：
	'input_info' :网络输入信息
	'output_info' :网络输出信息
	'layer_sizes' :脉冲神经网络层数
	'total_params' :脉冲神经网络所有参数

(20) get_model_input_size(net,sample_input) ：

.. code-block:: bash
    :linenos:

	功能： 自动推断输入模型的输入尺寸
	输入：
	'net' :神经网络模型
	'sample_input' :输入数据的样本张量
	输出：
	'sample_input_size' :网络模型的输入尺寸

(21) input_data_load(inputdata,data_list,inputspike_address) :

.. code-block:: bash
    :linenos:

	功能： 将输入数据添加到数据列表并为其分配相应地址空间
	输入：
	'inputdata' :输入数据
	'data_list' :整体数据列表
	'inputspike_address' :输入数据地址空间
	输出：
	'data_list' :加入了输入数据的整体数据列表

(22) linear_connections_trans(weight_matrix, input_size, input_neuron_ids, output_neuron_ids):

.. code-block:: bash
    :linenos:

	功能：将全连接层的连接矩阵转换为描述连接关系的三元组
	输入：
	'weight_matrix' :二维权重矩阵，形状为（输出，输入）
	'input_size' :连接的输入spike层神经元数量
	'input_neuron_ids' :连接的输入spike层神经元ID列表
	'output_neuron_ids' :连接的输出spike层神经元ID列表
	输出：
	'connections' :描述连接关系的三元组

(23) output_transfer(neuron_id_map,nfu,connections) :

.. code-block:: bash
    :linenos:

	功能：将每层神经元的映射结果保存在输出文件1，将连接关系转换为 GNC映射形式保存在输出文件2
	输入：
	'neuron_id_map' :神经元ID映射列表
	'nfu' :nfu实例
	'connections' :连接关系
	输出：
	保存映射结果和连接关系的文件


(24) outputnel_data_load(output_layer_value,data_list) ：

.. code-block:: bash
    :linenos:

	功能：将输出神经元的映射信息放入数据列表
	输入：
	'output_layer_value' :输出神经元列表
	'data_list' :整体数据列表
	输出：
	'data_list' :加入了输出神经元数据的整体数据列表

(25) sibilis_count(list) :

.. code-block:: bash
    :linenos:

	功能：计算各个神经元的前向连接数， 即兄弟节点数并将数据添加到连接关系数据列表中
	输入：
	'list' :描述连接关系的三元组
	输出：
	'NET_CONNECTION' :包含兄弟节点个数的连接关系四元组







	



	



工具函数
--------

TODO: 添加工具函数文档

配置选项
--------

TODO: 添加配置选项文档
