安装指南
========

系统要求
-----------------

TruSynapse 在主流 x86_64 Linux 发行版上测试。建议使用 Python 3.9 及以上（推荐 3.9–3.11）。
下面为经测试的依赖及推荐版本：

- Python：3.9（建议使用 virtualenv 或 conda 管理环境）
- PyTorch：2.8.0
- snntorch：0.9.4
- numpy：2.0.2
- scipy：1.13.1
- tqdm：4.67.1
- TensorFlow：2.20.0
- matplotlib：3.9.4（用于可视化，非必需）


安装步骤
-----------------

下面给出一种常见、稳健的安装流程，支持使用 virtualenv 或 conda 管理隔离环境。根据是否需要 GPU，请按注释选择 PyTorch 的安装方式。

1. 创建并激活虚拟环境

- 使用 venv：

.. code-block:: bash
    :linenos:

    python3 -m venv .venv
    source .venv/bin/activate

- 使用 conda：

.. code-block:: bash
    :linenos: 

    conda create -n trusynapse python=3.10 -y
    conda activate trusynapse

2. 安装 PyTorch（根据是否需要 GPU 选择）

- CPU 版本： 

.. code-block:: bash
    :linenos:
    
    pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

- GPU 版本（以 CUDA 11.8 为例）： 

.. code-block:: bash
    :linenos:
    
    pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

3. 安装其他依赖

.. code-block:: bash
    :linenos:

    pip install snntorch==0.9.4 numpy==2.0.2 scipy==1.13.1 tqdm==4.67.1 tensorflow==2.20.0 matplotlib==3.9.4

4. 安装 TruSynapse

下面说明如何将 TruSynapse 的 snntorch 框架文件替换到已安装的 snntorch 库位置（建议先备份原始文件）。

**步骤一：查找并备份原库位置**

.. code-block:: bash
    :linenos:

    python -c "import snntorch; print(snntorch.__file__)"
    # 记下输出路径（通常指向 .../snntorch/__init__.py 或其上级目录）
    cp -r /path/to/snntorch /path/to/snntorch.bak

**步骤二：删除原库内容并复制 TruSynapse 框架文件**

.. code-block:: bash
    :linenos:

    rm -rf /path/to/snntorch/*
    cp -r /our/framework/snntorch/* /path/to/snntorch/

    # 若遇到权限问题，可在命令前加 sudo（谨慎使用）
    # sudo rm -rf /path/to/snntorch/*
    # sudo cp -r /our/framework/snntorch/* /path/to/snntorch/

**步骤三：设置 PYTHONPATH（使替换后的库可被 Python 识别）**

.. code-block:: bash
    :linenos:

    export PYTHONPATH="/path/to/snntorch:$PYTHONPATH"

    # 若需永久生效，可追加到 shell 启动脚本:
    # echo 'export PYTHONPATH="/path/to/snntorch:$PYTHONPATH"' >> ~/.bashrc

大页内存配置
--------------------
为了支持 NFU，这里需要配置大页内存预先分配大小为 1GB 或更大的大页内存以供 NFU 使用。下面给出推荐的步骤与示例命令（根据需要调整大页大小与数量）。


1. 创建挂载点

.. code-block:: bash
    :linenos:

    sudo mkdir -p /mnt/huge

2. 临时设置大页数量（立即生效，但重启后失效）

.. code-block:: bash
    :linenos:

    sudo sysctl -w vm.nr_hugepages=512

3. 使配置永久生效（写入 /etc/sysctl.conf 并立即加载）

.. code-block:: bash
    :linenos:

    echo 'vm.nr_hugepages=512' | sudo tee -a /etc/sysctl.conf
    sudo sysctl -p

4. 挂载 hugetlbfs（根据所需的大页大小选择 pagesize，常用 1G）

.. code-block:: bash
    :linenos:

    # 使用 1G 大页
    sudo mount -t hugetlbfs none /mnt/huge -o pagesize=1G,mode=0777

5. 检查配置是否生效

.. code-block:: bash
    :linenos:

    grep Huge /proc/meminfo

    # 应能看到 HugePages_Total、Hugepagesize 等信息，
    # 若重启后仍存在相应信息，则说明配置已生效。

6. 在挂载目录下创建或预分配大页文件用于映射

.. code-block:: bash
    :linenos:

    cd /mnt/huge
    # 使用 truncate 预分配 1G 文件（根据实际需要调整大小）
    sudo truncate -s 1G myhugepage
    sudo chmod 660 myhugepage

提示

- ``pagesize`` 与 ``vm.nr_hugepages`` 的组合决定总的大页内存：总内存 ``≈ pagesize × nr_hugepages``。
- 若需在启动时自动挂载 ``hugetlbfs``，可在 ``/etc/fstab`` 添加相应条目。
- 根据系统策略与安全要求，调整文件与目录权限（示例使用宽松权限以便测试）。

执行体驱动
------------------

以 root 权限执行脚本 setup_hugepages_once.sh，系统启动后仅需执行一次。操作示例：

.. code-block:: bash
    :linenos:

    chmod +x setup_hugepages_once.sh
    sudo ./setup_hugepages_once.sh

脚本内容如下：

.. code-block:: bash
    :linenos:

    #!/bin/bash
    # setup_hugepages_once.sh - 由root执行一次

    echo "=== One-time HugePage Setup ==="

    # 1. 确保大页已分配（如果系统启动时已分配，这步可选）
    echo 1 > /proc/sys/vm/nr_hugepages

    # 2. 允许所有用户访问大页
    echo 0 > /proc/sys/vm/hugetlb_shm_group

    # 3. 确保hugetlbfs挂载
    mkdir -p /dev/hugepages
    mount -t hugetlbfs -o pagesize=1G, mode=0777 none /dev/hugepages 2>/dev/null || true

    # 4. 创建大页文件并设置权限
    truncate -s 1G /dev/hugepages/shared_mem
    chmod 666 /dev/hugepages/shared_mem

    # 5. 安装驱动并设置设备权限
    insmod ./accelerator.ko
    chmod 666 /dev/accelerator

    echo "One-time setup completed!"