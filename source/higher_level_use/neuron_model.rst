神经元模型
==========

7.1 汇编代码
------------
本节指出神经元模型在底层由 BCE 扩展指令集实现。详细指令请参见《BCE 体系结构之指令集》。

7.2 事件驱动高级语言（EDL）
----------------------------

事件驱动高级语言（Event-Driven Language, EDL）是专为描述神经元模型的事件驱动处理逻辑而设计的高级编程语言。本节详细介绍EDL的语法规范和使用方法。

7.2.1 语言总体结构
^^^^^^^^^^^^^^^^^^

``start`` 用于标记程序开始。

``end`` 用于标记程序结束。

基本结构：

.. code-block:: text

    start

    定义神经元
    声明事件类型
    初始化变量

    定义事件处理
    定义事件调度逻辑
    
    end

7.2.2 数据类型
^^^^^^^^^^^^^^

支持如下类型：

- ``int``    16-bit 整数
- ``float``  16-bit 浮点
- ``bool``   1-bit 布尔
- ``neuron`` 神经元复合类型

变量命名规则：

- 以字母或下划线开头
- 由字母、数字、下划线组成

**基本类型示例：**

.. code-block:: text

    int ID
    int num
    float weight
    float threshold
    bool fired
    bool is_reset

类型运算规则：

- int 与 float 运算结果默认 float
- 若赋值给 int，则自动截断为 int
- bool 不参与数值类型转换
- 不处理溢出

7.2.3 neuron 复合类型
^^^^^^^^^^^^^^^^^^^^^

支持分层存储映射：

- ``@fast``   高频访问，小容量
- ``@share``  共享参数，中容量
- ``@slow``   低频访问，大容量

定义格式：

.. code-block:: text

    neuron lif:
        @fast
            float v = 0.0

        @share
            float threshold = 1.0

        @slow
            float reset = 0.0

成员访问：

.. code-block:: text

    lif.v = 0.1

7.2.4 事件机制
^^^^^^^^^^^^^^

**事件声明：**

格式：

.. code-block:: text

    set 事件名 事件ID

ID 编码规则：

- 00 : spike
- 01 : timestep
- 10 : custom
- 11 : timestep_global

示例：

.. code-block:: text

    set event_spike 00
    set event_timestep 01

**事件跳转：**

.. code-block:: text

    goto event_spike

**emit_spike：**

用于产生脉冲事件：

.. code-block:: text

    if lif.v > lif.threshold:
        emit_spike
        lif.v = lif.reset

**wait_event：**

- 若无事件 → halt
- 若有事件 → 跳转至主循环判断

**事件定义：**

事件定义必须在声明之后。使用 ``end_event`` 结束事件块。

.. code-block:: text

    event_timestep:
        lif.v = potential + weight
        if lif.v > lif.threshold:
            lif.v = lif.reset
    end_event

7.2.5 控制流
^^^^^^^^^^^^

支持条件语句：

.. code-block:: text

    if 条件:
    elif 条件:
    else:

7.2.6 字面量
^^^^^^^^^^^^

支持：

- 二进制：``0b1010``
- 十进制：``42``、``3.14``、``-1``
- 布尔：``true``、``false``

示例：

.. code-block:: text

    float num = -0.2
    int binaryNum = 0b1011
    bool fired = false

7.2.7 内置变量
^^^^^^^^^^^^^^

只读变量：

- ``potential``    当前膜电位
- ``weight``       当前连接权重
- ``event_type``   当前事件类型

示例：

.. code-block:: text

    float v = potential + weight

    if event_type == event_spike:
        goto event_spike

7.2.8 运算符
^^^^^^^^^^^^

算术运算符：

.. code-block:: text

    + - * / += -= *= /= & ^ ++ -- ! % >> <<

比较运算符：

.. code-block:: text

    > < >= <= == !=

7.2.9 注释
^^^^^^^^^^

.. code-block:: text

    # 单行注释
    ## 多行注释 ##

7.2.10 完整示例
^^^^^^^^^^^^^^^

下列示例给出一个典型的 LIF（leaky integrate-and-fire）神经元模型及其事件处理流程，并在注释中说明了关键语义。

**基础示例：**

.. code-block:: text

    start

    neuron lif:
        @fast
            float v = 0.0

        @share
            float threshold = 1.0
            float reset = 0.0

        @slow
            bool spike = false

    set event_spike 00
    set event_timestep 01

    wait_event

    main:
        if event_type == event_spike:
            goto event_spike

    event_spike:
        lif.v = potential + weight

    end_event

    end

**详细示例：**

.. code-block:: text

    start
    # 全局变量声明
    int global_count = 0

    # 自定义神经元模型
    neuron lif:
        float v = 0.0
        float threshold = 1.0
        float reset = 0.0
        bool spike = false

    # 事件声明（使用二进制/十六进制或自定义编码）
    set event_spike                       00
    set event_timestep                    01
    set event_timestep_with_spike         10
    set event_decay                       11

    # 等待事件到来：编译器将 wait_event 映射为分支指令以跳转到 main
    wait_event
    # 示例映射（说明）：wait_event -> bre a1 a1  （a1 存放 main 的地址）

    # 主事件循环：根据事件类型跳转到对应处理块
    main:
        if event_type == event_spike:
            goto event_spike
        if event_type == event_timestep:
            goto event_timestep
        if event_type == event_timestep_with_spike:
            goto event_timestep_with_spike
        if event_type == event_decay:
            goto event_decay
        # 忽略其它类型的事件（假设调用方设置正确的事件类型）

    # 处理 event_spike：只做积分，不立即发放脉冲；结束时更新神经元状态
    event_spike:
        # 将输入权重累加到膜电位（示例表述）
        lif.v = potential + weight
    end_event
    # end_event 语义：在硬件上触发状态寄存器更新，然后返回 main 循环等待下一个事件

    # 处理 event_timestep：判断是否产生脉冲（spike）；若产生则发放并重置状态
    event_timestep:
        lif.v = potential
        if (lif.v >= lif.threshold):
            lif.spike = true
            # 发放脉冲事件（由运行时/硬件处理）
            emit_spike
            # 重置神经元状态
            lif.v = lif.reset
            lif.threshold += 0.1 * lif.threshold
        else:
            lif.spike = false
    end_event

    # 处理膜电位衰减事件（decay）：仅做衰减，最后更新状态
    event_decay:
        lif.v = 0.9 * potential
    end_event

    # 处理同时有输入和判断发放的事件
    event_timestep_with_spike:
        lif.v = potential + weight
        if (lif.v >= lif.threshold):
            lif.spike = true
            emit_spike
            lif.v = lif.reset
            lif.threshold += 0.1 * lif.threshold
        else:
            lif.spike = false
    end_event

    end

说明：

- 各事件处理块以 end_event 结束，编译器负责将其映射为必要的寄存器/指令序列以更新硬件状态并返回主循环。
- emit_spike 语义由运行时或硬件实现，表示向外部发放脉冲。
- 示例中的变量（potential、weight 等）为外部输入或上文定义的上下文变量，应根据实际模型补充声明。
