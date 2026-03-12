数据结构说明
================

本部分提供详细的数据结构说明。

HDF5文件接口相关数据结构
-----------------------------

.. _label_paras_process:

**paras_process** 类
    HDF5 接口核心功能类

    该类作为 HDF5 接口的核心功能载体，承担 HDF5 参数的写入与读取操作；同时完成 HDF5 参数、输入脉冲数据及神经元模型参数的整合与预处理，并将其转换为ctypes结构体 ``SNNData``，向下传递至 C 语言实现的 NFU 驱动模块。

    类内部函数说明（点击跳转）：

    - :ref:`主要函数<label_parasprocess_Mfuncs>`
    - :ref:`内部功能函数<label_path_process>`

    .. code-block:: bash
        :class: wrap-code

        class para_process:
            # 保存参数为HDF5格式
            def convert_paras(self, netparas: dict, path: str, save_file:bool = False)
            # 从HDF5中读取参数
            def load_file(cls, path: str, net_num:int = -1, paras_name:str = "all" )
            # 对输入的 HDF5 参数、输入脉冲数据及神经元模型参数进行预处理，并将其转换为ctypes结构体
            def parse_collect_to_struct(self,
                                    spikes_in_path: str,
                                    neurondata_in_path: str,
                                    subnetsandparas_in_path: str,
                                    subnet_num: int = 1,
                                    subnet_paras_name: str = "all")
            # 内部功能，检查文件路径
            def check_and_create_path(path: Optional[str], mode: str = 'check')
            # 内部功能，计算子网字典的数据深度
            def _nested_depth(obj)
            # 内部功能，读取二进制文件
            def read_binary_file(self,filename)
            # 内部功能，读取十进制或十六进制文件
            def read_decorhex_file(self,filename)


NFU驱动（C语言）相关数据结构
------------------------------------

**SNNData** 类
    SNN计算数据结构

    该类是一个基于 ``ctypes.Structure`` 构建的类，用于定义SNN计算所需的数据结构。这个结构体与C库中的对应定义保持一致，作为Python与C库之间的数据传输桥梁。

    .. code-block:: bash
        :class: wrap-code

        class SNNData(Structure):
            """SNN计算所需的数据结构，与C库中的定义对应"""
            _fields_ = [
                ("params", c_uint64 * 27),           # 27个配置参数
                ("connection_data", POINTER(c_uint64)),      # 连接关系数据
                ("connection_len", c_size_t),                # 连接关系数据长度
                ("inputneuronlist_data", POINTER(c_uint32)), # 输入神经元列表
                ("inputneuronlist_len", c_size_t),           # 输入神经元列表长度
                ("inputspike_data", POINTER(c_uint32)),      # 输入脉冲数据
                ("inputspike_len", c_size_t),                # 输入脉冲数据长度
                ("neuronbase_data", POINTER(c_uint32)),      # 神经元状态空间数据
                ("neuronbase_len", c_size_t),                # 神经元状态空间长度
                ("neuron_data", POINTER(c_uint32)),          # 神经元配置数据
                ("neuron_data_len", c_size_t),               # 神经元配置数据长度
                ("output_data", POINTER(c_uint32)),          # 输出结果数据
                ("output_len", c_size_t)                     # 输出结果长度
            ]

    **字段说明**：

    - ``params``: 27个NFU的寄存器参数，为硬件参数，定义硬件工作模式和内存布局
    - ``connection_data``: SNN网络的连接权重数据，存储了神经网络中所有神经元之间的连接关系和权重信息
    - ``inputneuronlist_data``: 输入神经元映射列表数据，用于定义输入数据如何映射到NFU的输入神经元
    - ``inputspike_data``: 为输入脉冲数据，从SNN网络的输入脉冲文件（.txt格式）中读取
    - ``neuronbase_data``: 神经元状态空间数据，存储SNN网络中每个神经元的状态
    - ``neuron_data``: 神经元模型的配置参数数据，从模型配置参数文件（.data格式）中读取
    - ``output_data``: 用于保存NFU的计算结果，由驱动在计算完成后填充
    - ``_len`` 字段: 对应数据数组的元素数量

**SNNDriver** 类
    SNN驱动封装类

    该类是一个封装类，用于调用C共享库执行SNN计算。

    类内部函数说明（点击跳转）：

    - :ref:`主要函数<label_SNNDriver_Mfuncs>`

    .. code-block:: bash
        :class: wrap-code

        class SNNDriver:
            """SNN驱动封装类，用于调用C库函数"""
            def __init__(self, lib_path='./libsnndriver.so'):
                """初始化驱动，加载C共享库"""
                if not os.path.exists(lib_path):
                    raise FileNotFoundError(f"SNN库文件不存在: {lib_path}")
                # 加载C共享库
                self.lib = ctypes.CDLL(lib_path)
                # 设置C函数原型
                self.lib.snn_execute.restype = c_int
                self.lib.snn_execute.argtypes = [POINTER(SNNData)]
                self.lib.snn_get_last_error.restype = c_char_p
                self.lib.snn_get_last_error.argtypes = []
                self.lib.snn_free_output.restype = None
                self.lib.snn_free_output.argtypes = [POINTER(SNNData)]

            # 获取最后一次错误信息
            def get_last_error(self)
            # 执行SNN计算
            def execute(self, data)
            # 释放输出内存
            def free_output(self, data)
