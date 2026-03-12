.. _neuron_model_definition:

神经元模型
==========

汇编代码
-----------------
本节指出神经元模型在底层由 BCE 扩展指令集实现。详细指令请参见《BCE 体系结构之指令集》。

事件驱动高级语言（EDL）
------------------------------

事件驱动高级语言（Event-Driven Language, EDL）是专为描述神经元模型的事件驱动处理逻辑而设计的高级编程语言。本节详细介绍EDL的语法规范和使用方法。

.. contents::
   :local:
   :depth: 3

语言使用逻辑
^^^^^^^^^^^^^^^^^^

``start`` 用于标记程序的开始。
``end`` 用于标记程序的结束。

编程逻辑架构：

.. code-block:: text

    start
    定义神经元
    声明事件类型
    初始化相关变量
    
    定义事件处理过程和事件跳转逻辑
    end

数据类型
^^^^^^^^^^^^^^

支持如下数据类型：

- ``int``       整数类型（16bit）
- ``float``     浮点类型（16bit）
- ``bool``      bool类型（1bit）
- ``neuron``    神经元类型

*变量命名规则：要以字母或者下划线开头，变量以数字、字母、下划线组成。*

**int 类型定义格式**

.. code-block:: text

    int 变量名

多个变量，通过换行符实现：

.. code-block:: text

    int ID
    int num

**float 类型定义格式**

.. code-block:: text

    float 浮点变量名

多个变量，通过换行实现：

.. code-block:: text

    float weight
    float threshold

**bool 类型定义格式**

.. code-block:: text

    bool 布尔变量名

多个变量，通过换行实现：

.. code-block:: text

    bool fired
    bool is_reset

**类型运算规则**

注意：int 和 float 类型做运算，比如：相加，相减，默认结果为 float，但是如果声明了一个变量为 int 类型，来存储运算的结果，就将float 类型转换为 int 后进行计算。bool 类型不能和其他两种类型进行转换。不用处理溢出的问题。

神经元复合类型
^^^^^^^^^^^^^^^^^^^^

使用neuron关键字来自定义神经元类型，并通过 ``@fast``、 ``@share``、 ``@slow`` 将不同参数分配到不同的存储区域，加快数据的读写。

存储区域说明：

- ``@fast``   频繁读写的神经元参数变量，容量极小
- ``@share``  可共享的、经常读取的神经元变量，容量稍大  
- ``@slow``   稀疏访问的神经元参数，容量较大

定义格式：

.. code-block:: text

    neuron 变量名:
        @fast
            float 成员变量 = 初始值
        @share
            int   成员变量 = 初始值
        @slow
            bool  成员变量 = 初始值

neuron变量使用示例：

.. code-block:: text

    neuron lif:
        @fast
            float membrane_potential = 0.0
        @share
            float reset_value = -0.1
        @slow
            float threshold = 1.0

**成员访问：**

- 变量名.成员名
- 变量名.成员名 = 字面量

示例：

.. code-block:: text

    lif.membrane_potential = 0.1

事件驱动机制
^^^^^^^^^^^^^^^^^^

为了充分利用类脑计算"事件触发计算"的优势，本语言在设计上引入了事件驱动编程机制，以事件作为程序执行的核心触发条件。通过显式定义事件类型、事件等待机制以及事件处理逻辑，程序仅在相关事件到达时才触发对应计算，从而避免对所有神经元进行无效遍历，提高计算效率。

**事件声明：set**

set关键字用于声明不同的事件类型，声明事件后，系统会关联对应的ID。

ID 编码规则：

- 00 : spike 事件
- 01 : timestep 事件
- 10 : 自定义事件
- 11 : timestep_global 事件

事件声明格式： ``set 事件名 事件id``

示例：

.. code-block:: text

    set event_spike 00
    set event_timestep 01
    set event_custom 10
    set event_timestep_global 11

**事件跳转：goto**

goto关键字用于事件地址的跳转。

事件跳转格式： ``goto 事件名``

示例：

.. code-block:: text

    goto event_spike    // 跳转至 spike 事件的处理地址

**产生脉冲：emit_spike**

用于产生脉冲事件：

.. code-block:: text

    if lif.v > lif.threshold:
        emit_spike
        lif.v = lif.reset

**wait_event**

为了实现高效的事件驱动，设计了 wait_event 关键字来表示等待事件到来，有事件到来就跳转到判断事件类型的地址，没有就halt。

**事件定义**

事件定义需要在事件声明之后进行，事件定义用于定义事件处理相关代码和处理逻辑。事件定义模块定义的最后，需要使用 ``end_event`` 关键字，将标志事件处理结束，编译器会自动进行状态更新，并在神经元状态更新后自动进入时间等待状态。

使用示例：

.. code-block:: text

    event_timestep_local:
        lif.membrane_potential = potential + weight
        if lif.membrane_potential > lif.threshold:
            spike = 1
            lif.membrane_potential = lif.reset_value
    end_event

控制流
^^^^^^^^^^^^

为了更好地支持复杂事件的处理，事件驱动语言增加了控制流关键字，来实现不同条件下的数据处理。

支持条件语句：

.. code-block:: text

    if 条件判断表达式:
    elif 条件判断表达式：
    else:

使用示例：

.. code-block:: text

    if lif.v >= lif.threshold:
        lif.v = lif.reset

字面量
^^^^^^^^^^^^

**常规字面量支持**

- 二进制：``0b1010``
- 十进制：``42``、``3.14``、``-1``、``-0.2``
- 布尔：``true``、``false``

示例：

.. code-block:: text

    float num = -0.2
    int binaryNum = 0b1011
    bool is_fired = false

**内置变量**

内置变量是一个只读变量，只能放在赋值表达式的右边，用于读取一些特殊变量（事件到达时，硬件自动加载的变量）的值，或者进行判断语句和表达式。

每个变量存储的值如下：

- ``potential``    当前膜电位
- ``weight``       当前神经元与发送事件的神经元的连接权重
- ``event_type``   事件类型

示例：

.. code-block:: text

    float membranePotential = potential + weight
    
    if event_type == event_spike:
        goto event_spike

运算符
^^^^^^^^^^^^

算术运算符：

.. code-block:: text

    + - * / += -= *= /= & ^ ++ -- ! % >> <<

比较运算符：

.. code-block:: text

    > < >= <= == !=

语法规则
^^^^^^^^^^^^^^

**注释**

- 单行注释：``# 注释内容``
- 多行注释：``## 注释内容 ##``

**语句结束**

- 使用换行符表示语句结束
- 缩进表示代码块层级

**分隔符**

- **小括号 ()** ：支持运算优先级、未来扩展
- **冒号 :**
  - if、else 代码块
  - 事件处理块

完整示例
^^^^^^^^^^^^^^

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
        @fast
            float v = 0.0
        @share
            float threshold = 1.0
            float reset = 0.0
        @slow
            bool spike = false

    # 事件声明
    set event_spike                       00
    set event_timestep                    01
    set event_timestep_with_spike         10
    set event_decay                       11

    # 使用wait_event，等待事件到来，如果有事件到来，就进入 main 循环中，进行循环的判断。
    wait_event

    # 主事件循环
    main:
        if event_type == event_spike:
            goto event_spike
        if event_type == event_timestep:
            goto event_timestep
        if event_type == event_timestep_with_spike:
            goto event_timestep_with_spike
        if event_type == event_decay:
            goto event_decay
        # 这里不做不属于这 4 种类型的其他事件的处理，用户需正确设置事件。

    # 处理事件 event_spike
    event_spike:
        # spike事件只做积分操作，不发放脉冲，但是在最后更新神经元的状态
        lif.v = potential + weight
    end_event
    # end_event 首先更新神经元的状态，然后进入wait_event状态

    # 处理事件 event_timestep
    event_timestep:
        # timestep事件只做是否产生spike，不更新神经元的状态
        lif.v = potential
        if(lif.v >= lif.threshold):
            lif.spike = true
            # 发放脉冲事件
            emit_spike
            # 重置神经元的状态
            lif.v = lif.reset
            lif.threshold += 0.1 * lif.threshold
        else:
            lif.spike = false
    end_event

    event_decay:
        # decay事件只做膜电位衰减操作，不发放脉冲，但是在最后更新神经元的状态
        lif.v = 0.9 * potential
    end_event

    event_timestep_with_spike:
        # timestep_with_spike事件做积分操作，同时判断是否产生spike，最后要更新神经元状态
        lif.v = potential + weight
        if(lif.v >= lif.threshold):
            lif.spike = true
            # 发放脉冲事件
            emit_spike
            # 重置神经元的状态
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

与编程框架集成
^^^^^^^^^^^^^^^^^^^^^^^^^

针对使用事件驱动高级语言开发的程序，需要使用对应的编译器进行编译，生成NFU汇编指令，然后重新安装框架，即可在框架中调用自定义的神经元模型。

编译命令：

.. code-block:: bash

    evc myprogram.edl -o myprogram.s

注意：需要将生成的汇编代码移动到框架源代码目录下。

使用自定义的神经元模型：详见系统中调用LIF等内置的神经元模型调用方式。
