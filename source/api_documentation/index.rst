API 文档
============

本部分提供详细的 API 参考文档。

.. toctree::
   :maxdepth: 2

网络映射相关API
--------------------

(1) **assign_neuron_ids_1d** 一维网络神经元分配ID

.. code-block:: bash
    :linenos:

    assign_neuron_ids_1d(net,input_size) :

    功能：为一维神经网络中的每个神经元分配唯一的全局ID
    输入：
    'net' :脉冲神经网络模型
    'input_size' :代表输入纬度的元组（通道数， 长度）
    输出：
    'neuron_id_map' :神经元ID，各层神经元映射
    'total_neurons' :神经网络中所有神经元数量

(2) **assign_neuron_ids_2d** 二维网络神经元分配ID

.. code-block:: bash
    :linenos:

    assign_neuron_ids_2d(net,input_size) :

    功能：为二维神经网络中的每个神经元分配唯一的全局ID
    输入：
    'net' :脉冲神经网络模型
    'input_size' :代表输入纬度的元组（通道数， 高度， 宽度）
    输出：
    'neuron_id_map' :神经元ID，各层神经元映射
    'total_neurons' :神经网络中所有神经元数量


(3) **assign_neuron_ids_3d** 三维网络神经元分配ID

.. code-block:: bash
    :linenos:

    assign_neuron_ids_3d(net,input_size) :
    
    功能：为三维神经网络中的每个神经元分配唯一的全局ID
    输入：
    'net' :脉冲神经网络模型
    'input_size' :代表输入纬度的元组（通道数， 深度， 高度， 宽度）
    输出：
    'neuron_id_map' :神经元ID，各层神经元映射
    'total_neurons' :神经网络中所有神经元数量

(4) **assign_select** 按尺寸选择ID分配函数

.. code-block:: bash
    :linenos:

    assign_select(net,input_size) :

    功能：依据输入尺寸选择合适的assign_neuron_ids函数
    输入：
    'net' :脉冲神经网络模型
    'input_size' :代表输入纬度的元组
    输出：调用对应的assign_neuron_ids函数，输出与对应函数输出—致

(5) **build_connections** 构建神经元的连接关系

.. code-block:: bash
    :linenos:

    build_connections(net,neuron_id_map,input_size) :

    功能：构建神经网络内各个神经元之间的连接关系(测试可用，权重值随机)
    输入：
    'net' :脉冲神经网络模型
    'neuron_id_map' :神经元映射ID列表
    'input_size' :代表输入纬度的元组
    输出：
    'connections' :描述连接关系的三元组

(6) **check_save_eq** 校验加载同名文件一致性

.. code-block:: bash
    :linenos:

    check_save_eq(save_file_name, net, sample_input_size) :

    功能：检查是否存在与输入文件名称相同的文件，如果存在文件则尝试加载。 加载成功则对比文件内神经网络、输入尺寸与输入是否一致， 一致则返回所有加载内容，不一致或其他异常则返回None
    输入：
    'save_file_name' :保存的文件名
    'net' :脉冲神经网络
    'sample_input_size' :网络输入尺寸
    输出：
    'save_data' :如果文件存在， 返回保存的数据
    'None' ：没有文件存在， 返回None

(7) **class GNC** GNC类

.. code-block:: bash
    :linenos:

    class GNC(gnc_id,neuron_id) :

    功能 ：GNC类，用于创建GNC和放置神经元
    参数：
    'gnc_id' :gnc ID号
    'neuron_id' :神经元ID号

(8) **cluster** 基于连接关系聚类映射神经元-方式一

.. code-block:: bash
    :linenos:

    cluster(neuron_id_map, connections, relation=1) 

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

(9) **cluster_input2center** 基于连接关系聚类映射神经元-方式二

.. code-block:: bash
    :linenos:

    cluster_input2center(neuron_id_map, connections, relation=1) :

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

(10) **cluster2complex_single** 基于连接关系聚类映射神经元-方式三

.. code-block:: bash
    :linenos:

    cluster2complex_single(neuron_id_map, connections) :

    功能 ：基于连接关系将神经元聚类映射到NFU的GNC中（模拟退火法）
    输入：
    'neuron_id_map' :神经元ID映射列表
    'connections' :连接关系
    输出：
    'input_mapping' :输入层神经元映射结果
    'output_mapping' :输出层神经元映射结果
    'nfu' :NFU使用率
    'longest_time_expression' ：网络中spike传输的最长时间

(11) **ComplexSNN** 构建脉冲神经网络

.. code-block:: bash
    :linenos:

    ComplexSNN(input_neu,hidden_neu,output_neu) :

    功能：依据各层神经元的数量组成脉冲神经网络，用于后续构建连接关系以及神经元映射
    输入：
    'input_neu' :输入层神经元数量
    'hidden_neu' :隐藏层神经元数量
    'output_neu' :输出层神经元数量
    输出：
    'net' :根据输入要求形成的脉冲神经网络

(12) **conv_connections_trans** 卷积层连接矩阵转换三元组

.. code-block:: bash
    :linenos:

    conv_connections_trans(weight_matrix, input_size, input_neuron_ids, output_neuron_ids,stride=(1, 1), padding=(0, 0), dilation=(1, 1)) :

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

(13) **get_model_input_size** 获取模型输入尺寸

.. code-block:: bash
    :linenos:

    get_model_input_size(net,sample_input) :

    功能： 自动推断输入模型的输入尺寸
    输入：
    'net' :神经网络模型
    'sample_input' :输入数据的样本张量
    输出：
    'sample_input_size' :网络模型的输入尺寸

(14) **linear_connections_trans** 全连接层连接矩阵转换三元组

.. code-block:: bash
    :linenos:

    linear_connections_trans(weight_matrix, input_size, input_neuron_ids, output_neuron_ids):

    功能：将全连接层的连接矩阵转换为描述连接关系的三元组
    输入：
    'weight_matrix' :二维权重矩阵，形状为（输出，输入）
    'input_size' :连接的输入spike层神经元数量
    'input_neuron_ids' :连接的输入spike层神经元ID列表
    'output_neuron_ids' :连接的输出spike层神经元ID列表
    输出：
    'connections' :描述连接关系的三元组

(15) **get_net_info** 获取网络参数

.. code-block:: bash
    :linenos:

    get_net_info(net) :

    功能：获取网络相关参数
    输入：
    'net' :脉冲神经网络模型
    输出：
    'input_info' :网络输入信息
    'output_info' :网络输出信息
    'layer_sizes' :脉冲神经网络层数
    'total_params' :脉冲神经网络所有参数

(16) **output_transfer** 输出数据转换保存

.. code-block:: bash
    :linenos:

    output_transfer(neuron_id_map,nfu,connections) :

    功能：将每层神经元的映射结果保存在输出文件1，将连接关系转换为 GNC映射形式保存在输出文件2
    输入：
    'neuron_id_map' :神经元ID映射列表
    'nfu' :nfu实例
    'connections' :连接关系
    输出：
    保存映射结果和连接关系的文件

空间分配API
--------------------

(1) **address** 预分配数据存储空间 :

.. code-block:: bash
    :linenos:

    address(length,output_layer_length,input_layer_length,BATCH_SIZE,BATCH) :

    功能：为连接关系数据、输出数据、输入数据、输出层神经元数据、输入层神经元数据分配空间用于后续保存相关数据
    输入：
    'length' :连接关系数据规模
    'output_layer_length' :输出层数据规模
    'input_layer_length' :输入层数据规模
    'BATCH_SIZE' :每批次输入数据长度
    'BATCH' :输入数据批次
    输出：
    各段数据空间的起始与结束地址（基地址为0）

(2) **address_list_init** 数据列表初始化

.. code-block:: bash
    :linenos:

    address_list_init(address_data_list) :

    功能:将数据列表依据地址空间分配列表进行初始化
    输入:
    'address_data_list' :地址空间分配列表
    输出:
    'mem_file_content' :初始化后的数据空间

(3) **check_64bit_align** 检查64位对齐

.. code-block:: bash
    :linenos:

    check_64bit_align(mem_file_content,address_data_list) :

    功能：检查分配的地址空间是否为64位对齐，若是则不变，若否，则将地址转换为64位对齐数据后存入地址分配空间
    输入：
    'mem_file_content' :数据空间列表
    'address_data_list' :地址分配列表
    输出：
    'addresses' :修改后的地址空间分配列表

(4) **connection_data_load** 连接关系数据加载

.. code-block:: bash
    :linenos:

    connection_data_load(connection_list,data_list) :

    功能：将连接关系列表中的数据按照一定的规则放入数据列表
    输入：
    'connection_list' :连接关系列表
    'data_list' :整体数据列表
    输出：
    'data_list' :加入了连接关系数据的整体数据列表

(5) **input_data_load** 输入数据加载

.. code-block:: bash
    :linenos:

    input_data_load(inputdata,data_list,inputspike_address) :

    功能： 将输入数据添加到数据列表并为其分配相应地址空间
    输入：
    'inputdata' :输入数据
    'data_list' :整体数据列表
    'inputspike_address' :输入数据地址空间
    输出：
    'data_list' :加入了输入数据的整体数据列表

(6) **sibilis_count** 兄弟节点数计算

.. code-block:: bash
    :linenos:

    sibilis_count(list) :

    功能：计算各个神经元的前向连接数，即兄弟节点数并将数据添加到连接关系数据列表中
    输入：
    'list' :描述连接关系的三元组
    输出：
    'NET_CONNECTION' :包含兄弟节点个数的连接关系四元组

(7) **outputnel_data_load** 输出神经元映射信息加载

.. code-block:: bash
    :linenos:

    outputnel_data_load(output_layer_value,data_list) :

    功能：将输出神经元的映射信息放入数据列表
    输入：
    'output_layer_value' :输出神经元列表
    'data_list' :整体数据列表
    输出：
    'data_list' :加入了输出神经元数据的整体数据列表

HDF5接口及驱动API
-------------------
(1) **net_process** 网络处理

.. code-block:: bash
    :linenos:

    net_process(net,connection_path,inputdata_path,output_file_path) :

    功能：该函数处理SNN网络的数据，并生成子网执行体参数与NFU硬件参数，供后续NFU驱动使用。此外，该函数会调用paras_process类完成HDF5的参数保存工作。
    输入：
    'net': 实例化对象，为已完成实例化的 SNN 网络对象；
    'connection_path': 字符串，文件输入路径，为存储 SNN 网络结构信息（含.weight 类权重数据）的文件，支持 pkl、pth 格式；
    'inputdata_path': 字符串，文件输入路径，为SNN网络的输入脉冲文件，仅支持txt格式，每行代表一个脉冲事件；
    'output_file_path': 字符串，文件输出路径，为存储 NFU所需数据的文件（包含子网参数、结构信息等），供后续 C 库解析，默认采用 HDF5 格式存储。支持hdf5/HDF5/H5/h5多种HDF5后缀名。
    输出：
    'paras':  字典，包含存入HDF5文件里的所有参数，参数存入HDF5后，可不用再次读取HDF5直接使用该参数继续后续流程。

(2) **convert_paras** 将参数存入HDF5

.. code-block:: bash
    :linenos:

    convert_paras(self, netparas: dict, path: str, save_file:bool = False ):
    属于paras_process类

    功能：接收字典型的网络参数，并按要求保存为HDF5文件。
    输入：
    'paras': 字典，包含需要存入HDF5文件里的所有参数
    'output_file_path': 字符串，文件输出路径，为存储 NFU所需数据的文件（包含子网参数、结构信息等），供后续 C 库解析，默认采用 HDF5 格式存储。支持hdf5/HDF5/H5/h5多种HDF5后缀名。
    输出：
    HDF5文件：包含所有参数的HDF5文件，供后续 C 库解析使用。

(3) **load_file** 从HDF5加载参数

.. code-block:: bash
    :linenos:

    @classmethod
    load_file(cls, path: str, net_num:int = -1, paras_name:str = "all" ):
    属于paras_process类

    功能：按要求读取出HDF5文件数据。
    输入：
    'path': 字符串，文件输入路径，为符合格式的HDF5文件，路径支持hdf5/HDF5/H5/h5多种HDF5后缀名。
    'net_num': 整形，子网编号，指定从HDF5文件中读出几号子网，-1表示一次性读出全部子网。
    'paras_name': 字符串，子网参数名称，指定从子网中读出某个参数，“all”“connection_data”、“inputneuronlist_data”、“neuronbase_data”“register_paras”，默认“all”读出全部参数。
    输出：
    'subnetlist': 字典，包含按要求从HDF5文件里的读出的参数，读出的格式为“{'NFUnet子网编号': {'参数名': 参数内容...}...}”，如{'NFUnet1': {'register_paras': [0, 1048576, 1125889]}, 'NFUnet2': {'register_paras': [1, 123441, 131875]}}

(4) **parse_collect_to_struct** 提取参数并构建结构体

.. code-block:: bash
    :linenos:

    parse_collect_to_struct(self,
                           spikes_in_path: str,
                           neurondata_in_path: str,
                           subnetsandparas_in_path: str,
                           subnet_num: int = 1,
                           subnet_paras_name: str = "all"):
    属于paras_process类

    功能：从指定的多个路径读取数据文件（输入脉冲txt文件、神经元模型文件和HDF5文件），并对这些数据进行处理，最终转换为ctypes结构体SNNData。该方法会调用load_file方法读取HDF5文件。
    输入：
    'spikes_in_path': 字符串，文件输入路径，为SNN网络的输入脉冲文件，仅支持txt格式，每行代表一个脉冲事件。
    'neurondata_in_path': 字符串，文件输入路径，为神经元模型的参数文件，定义神经元模型；
    'subnetsandparas_in_path': 字符串，文件输入路径，为符合格式的HDF5文件，路径支持hdf5/HDF5/H5/h5多种HDF5后缀名。
    'subnet_num': 整形，子网编号，指定从HDF5文件中读出几号子网，-1表示一次性读出全部子网。
    'subnet_paras_name': 字符串，子网参数名称，指定从子网中读出某个参数，“all”“connection_data”、“inputneuronlist_data”、“neuronbase_data”“register_paras”，默认“all”读出全部参数。
    输出：
    'SNNData': 实例化对象，为ctypes结构体，该ctypes结构体实现 Python 与 C 语言的内存数据互通，确保 Python 侧构造的SNNData实例，其内存布局及字段类型与 C 端 NFU 驱动中定义的SNNData结构体完全对齐，支持 Python 侧参数直接传递至 C 语言函数调用。

(5) **execute_computing** 使用输入的数据执行计算

.. code-block:: bash
    :linenos:

    execute_computing(self,
                    spikes_in_path: str,
                    neurondata_in_path: str,
                    subnetsandparas_in_path: str,
                    subnet_num: int = 1):
    属于paras_process类

    功能：从指定的多个文件里读取数据，并使用这些数据执行计算，最终返回结果。该方法会调用parse_collect_to_struct方法构建结构体，也会调用SNNDriver类的execute方法执行计算。
    输入：
    'spikes_in_path': 字符串，文件输入路径，为SNN网络的输入脉冲文件，仅支持txt格式，每行代表一个脉冲事件。
    'neurondata_in_path': 字符串，文件输入路径，为神经元模型的参数文件，定义神经元模型。
    'subnetsandparas_in_path': 字符串，文件输入路径，为符合格式的HDF5文件，路径支持hdf5/HDF5/H5/h5多种HDF5后缀名。
    'subnet_num': 整形，子网编号，指定从HDF5文件中读出几号子网，-1表示一次性读出全部子网。
    输出：
    'output_results': 列表，为NFU计算完的原始结果，需要处理，注意，输出脉冲列表中的首个“1”表示存在输出，此为标志位，并非实际的输出脉冲，实际输出脉冲应从列表第二个元素开始计算。

(6) **get_last_error** 获取上次的错误信息

.. code-block:: bash
    :linenos:

    get_last_error(self):
    属于SNNDriver类

    功能：调用C库的snn_get_last_error函数，获取其以C字符串指针返回的最后一次错误信息，并将该C字符串解码为Python字符串，若为空则类方法返回 "未知错误"。
    输出：
    处理后的错误信息字符串。

(7) **execute** 使用输入数据执行计算

.. code-block:: bash
    :linenos:

    execute(self, data):
    属于SNNDriver类

    功能：该类方法调用C库的snn_execute函数执行SNN计算，计算后检查返回值，若不为 0（表示失败），则获取错误信息并抛出RuntimeError异常。NFU计算返回的原始结果会保存在snndata.outputdata数组里。
    输入：
    'data': 实例化对象，具体为实例化的SNNdata类，包含计算所需的所有数据；
    输出：
    正常返回0，出错则抛出错误并终止。

(8) **free_output** 释放内存

.. code-block:: bash
    :linenos:

    free_output(self, data):
    属于SNNDriver类

    功能：直接调用C库的snn_free_output函数，释放C库分配的输出内存。注意，确保每次调用后都释放内存，避免内存泄漏。
    输入：
    'data': 实例化对象，具体为实例化的SNNdata类，其output_data字段指向需要释放的内存；

工具API
-------------------

(1) **conv_connections_trans** 卷积层连接矩阵转换三元组

.. code-block:: bash
    :linenos:

    conv_connections_trans(weight_matrix, input_size, input_neuron_ids, output_neuron_ids,stride=(1, 1), padding=(0, 0), dilation=(1, 1)):

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

(2) **linear_connections_trans** 全连接层连接矩阵转换三元组

.. code-block:: bash
    :linenos:

    linear_connections_trans(weight_matrix, input_size, input_neuron_ids, output_neuron_ids):

    功能：将全连接层的连接矩阵转换为描述连接关系的三元组
    输入：
    'weight_matrix' :二维权重矩阵，形状为（输出，输入）
    'input_size' :连接的输入spike层神经元数量
    'input_neuron_ids' :连接的输入spike层神经元ID列表
    'output_neuron_ids' :连接的输出spike层神经元ID列表
    输出：
    'connections' :描述连接关系的三元组

(3) **check_and_create_path** 路径处理

.. code-block:: bash
    :linenos:

    check_and_create_path(path: Optional[str], mode: str = 'check') :
    属于paras_process类

    功能：接收路径参数，并使用校验/创建模式处理路径（只支持HDF5/JSON的文件后缀）。
    输入：
    'path': 字符串，输入路径；
    'mode': 字符串，模式，分别为“check”和“create”两种模式，“check”模式为校验模式，“create”为创建模式，校验模式下，仅对包含文件名的目标路径执行合法性校验，校验不通过时抛出异常；创建模式下，优先尝试创建用户指定的自定义路径（含文件名），若创建操作失败，则降级创建默认路径。
    输出：
    'ext_name|new_file_path':  字符串，校验模式返回文件扩展名，创建模式返回使用的路径。

(4) **_nested_depth** 计算嵌套层数

.. code-block:: bash
    :linenos:

    _nested_depth(obj) :
    属于paras_process类

    功能：计算传进来的字典的层数，为后续其他函数根据层数判断子网是否切割提供帮助。
    输入：
    'obj': 实例化对象，需要判断层数的字典。
    输出：
    整形，返回深度层数。

(5) **read_decorhex_file** 读取十进制或十六进制文件

.. code-block:: bash
    :linenos:

    read_decorhex_file(self,filename) :
    属于paras_process类

    功能：是读取十进制或十六进制文件，其根据行的格式解析值，若是以 "0x" 开头的字符串，按十六进制解析；其他情况按十进制解析。
    输入：
    'filename': 字符串，输入路径；
    输出：
    列表，返回解析后的整数列表。

(6) **read_binary_file** 读取二进制文件

.. code-block:: bash
    :linenos:

    read_binary_file(self,filename) :
    属于paras_process类

    功能：读取二进制文件，其根据行的格式解析值，若是32位长度的字符串或是以 "0b" 开头的字符串，按二进制解析；其他情况则按十进制解析。
    输入：
    'filename': 字符串，输入路径；
    输出：
    列表，返回解析后的整数列表。 

