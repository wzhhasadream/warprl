# 安装配置文档

## 系统要求

- **操作系统**：推荐使用 Ubuntu 22.04
- **显卡**：Nvidia 显卡  
- **驱动版本**：建议使用 550 或更高版本  

---

## 1. 创建虚拟环境

建议在虚拟环境中运行训练或部署程序，推荐使用 Conda 创建虚拟环境。如果您的系统中已经安装了 Conda，可以跳过步骤 1.1。

### 1.1 下载并安装 MiniConda

MiniConda 是 Conda 的轻量级发行版，适用于创建和管理虚拟环境。使用以下命令下载并安装：

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

安装完成后，初始化 Conda：

```bash
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

### 1.2 创建新环境

使用以下命令创建虚拟环境：

```bash
conda create -n unitree_rl_mjlab python=3.11
```

### 1.3 激活虚拟环境

```bash
conda activate unitree_rl_mjlab
```

---

## 2. 安装

### 2.1 下载

通过 Git 克隆仓库：

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
```

### 2.2 安装依赖

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

我们将其余所需依赖放入 setup.py 文件中，
进入 unitree_rl_mjlab 项目根目录并安装：

```bash
cd unitree_rl_mjlab
pip install -e .
```

## 总结

按照上述步骤完成后，您已经准备好在虚拟环境中运行相关程序。若遇到问题，请参考各组件的官方文档或检查依赖安装是否正确。

