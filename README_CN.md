# 🚀 CLGSDN: 对比学习图结构去噪网络

本仓库是 IEEE Internet of Things Journal (JIOT) 论文 "**CLGSDN: Contrastive-Learning-Based Graph Structure Denoising Network for Traffic Prediction**" 的官方 PyTorch 实现。

<p align="center">
  <b>中文说明</b> | <a href="./README.md">English</a>
</p>

---

## 📖 目录
1. [模型架构](#-1-模型架构)
2. [动机与可视化展示](#-2-动机与可视化展示)
3. [数据准备](#-3-数据准备)
4. [相关项目](#-4-相关项目)
5. [环境配置](#-5-环境配置)
6. [运行实验](#-6-运行实验)
7. [实验结果分析与解读](#-7-实验结果分析与解读)
8. [引用](#-8-引用)

---

## 🏗️ 1. 模型架构

CLGSDN 旨在解决交通预测中由于人工定义或自动生成的图结构包含噪声，从而导致时空模型性能下降的问题。该网络主要包含三个模块：PNO 模块生成观测值，PLM 模块学习并记忆最优图的概率，DF 模块去除噪声边。

<p align="center">
  <img src="./figures/structure.png" alt="CLGSDN 总体架构图" width="80%">
  <br>
  <b>图 1: CLGSDN 总体架构。</b> 包含三个主要模块：首先，PNO 模块从输入数据中生成一组观测值；其次，PL 和记忆（PLM）模块学习并记忆最优图的概率；最后，DF 模块去除噪声边。
</p>

---

## 📊 2. 动机与可视化展示

### 2.1 现有图结构的缺陷 (Motivation)
传统的静态图过于稀疏，而基于注意力机制的动态图虽然捕捉了更多关系，但引入了大量不必要的噪声（弱连接）。

<p align="center">
  <img src="./figures/figure-2-a-static.png" alt="静态图热力图" width="45%">
  <img src="./figures/figure-2-b-attn.png" alt="注意力图热力图" width="45%">
  <br>
  <b>图 2: 静态图与注意力图的可视化热力图对比。</b> 黑色表示无连接，白色表示强连接。(a) 静态图仅包含少量连接。(b) 注意力图几乎包含了所有可能的连接，但大部分是无用的（深红色表示的弱连接）。
</p>

### 2.2 CLGSDN 的去噪效果 (Our Solution)
CLGSDN 能够有效学习图概率并进行去噪，下图标明了去噪前后的对比。

<p align="center">
  <img src="./figures/figure-6-heatmap-a.png" alt="CLGSDN 概率图" width="45%">
  <img src="./figures/figure-6-heatmap-b.png" alt="CLGSDN 去噪概率图" width="45%">
  <br>
  <b>图 3: CLGSDN 生成的图热力图对比 (数据集: Metr-la, 基座模型: STGCN)。</b> (a) 初始概率图。(b) 去噪后的概率图。可以看到，去噪后的整体图像变暗，表明由深红色代表的弱连接（噪声）已被移除。
</p>

---

## 📂 3. 数据准备

请确保你的工作目录按照以下结构组织：

```text
├─datasets
│ └─raw_dataset
│   ├─metr_la
│   ├─pems_bay
│   ├─pems04
│   └─pems08
├─configs
├─model
├─utils
├─engine.py
└─exp.py
````

  * **数据下载**: [Google Drive](https://drive.google.com/file/d/1gt4f9-NlcH6IKBzsDsTSUaaPrUONIgqV/view?usp=sharing) 下载后解压至 `datasets/raw_dataset` 目录下。

-----

## 🔗 4. 相关项目

为了进一步提高本工作的可用性和可复现性，我们开源了以下配套项目：

- **[Scientific-Data-Pipeline](https://github.com/P-PPPPP/Scientific-Data-Pipeline)**  
  一个全流程数据集仓库。它整合了**本论文使用的所有数据集**（METR-LA, PEMS-BAY, PEMS04, PEMS08 等）的一键下载与预处理方法，提供了更优、更稳健的数据处理流程。

- **[Scientific-Neural-Lab](https://github.com/P-PPPPP/Scientific-Neural-Lab)**  
  *(即将开放)*  
  该仓库将提供一个更简洁、先进的多任务训练框架，解决当前代码库中逻辑混乱的问题。项目正在积极构建中，即将开源。



-----

## 🛠️ 5. 环境配置

### 5.1 创建并激活环境

```bash
# 创建环境 (建议 Python 3.11)
conda create -n CLGSDN_envs python=3.11 -y
# 激活环境
conda activate CLGSDN_envs
```

### 5.2 安装 PyTorch (核心步骤)

由于硬件（CPU/GPU）及 CUDA 版本不同，**请务必根据自己的设备环境安装适配的 PyTorch**。

1.  访问 [PyTorch 官网本地安装页面](https://pytorch.org/get-started/locally/)。
2.  根据你的 OS、Package (Conda/Pip)、CUDA 版本生成安装命令并执行。

### 5.3 安装其他依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 注意：PyTables 有时通过 pip 安装会报错，可以尝试使用 conda 安装
# conda install pytables
```

-----

## 🏃 6. 运行实验

### 6.1 基础命令

```bash
python exp.py --model <model> --dataset <data> --GSL <generator> [其他参数]
```

### 6.2 参数说明

| 参数名 | 类型 | 核心可选值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--model` | Str | `agcrn`, `astgcn`, `tgcn`, `gw`, `dstagnn`, `dcrnn` 等 | 指定基座预测模型。 |
| `--dataset` | Str | `metr_la`, `pems_bay`, `pems04`, `pems08` | 指定数据集。 |
| `--GSL` | Str | `CLGSDN`, `None` | `CLGSDN`: 使用我们的图生成器;<br>`None`: 使用数据集原始图或单位矩阵。 |
| `--select_channels`| List[Int] | `0`, `1`, `2`, `-1` | **核心参数，请根据数据选对：**<br>• Metr-la/Pems-bay: 默认为 `0` (仅单通道)<br>• Pems04/08: 流量选 `0`，速度选 `2`<br>• TaxiBJ: Flow\_in 选 `0`, Flow\_out 选 `1` |
| `--n_prob` | Int | `3` (默认) | 概率图的相关参数。 |
| `--device` | Str | `cuda:0`, `cpu` | 指定运行设备。 |

### 6.3 快速实验示例 (Graph WaveNet)

以下命令展示了 Graph WaveNet (gw) 在不同数据集上结合 CLGSDN 与不结合的对比实验。

```bash
# === Metr-la ===
# 使用 CLGSDN (Ours)
nohup python -u exp.py --model gw --dataset metr_la --GSL CLGSDN --n_prob 3 --select_channels 0 --device cuda:0 > ./logs/CLGSDNxGW_on_metr-la.log &
# 不使用 CLGSDN (Baseline)
nohup python -u exp.py --model gw --dataset metr_la --GSL None --select_channels 0 --device cuda:0 > ./logs/GW_on_metr-la.log &

# === Pems-bay ===
# 使用 CLGSDN (Ours)
nohup python -u exp.py --model gw --dataset pems_bay --GSL CLGSDN --n_prob 3 --select_channels 0 --device cuda:0 > ./logs/CLGSDNxGW_on_pems-bay.log &
# 不使用 CLGSDN (Baseline)
nohup python -u exp.py --model gw --dataset pems_bay --GSL None --select_channels 0 --device cuda:0 > ./logs/GW_on_pems-bay.log &

# === Pems04 (选择流量通道 0) ===
# 使用 CLGSDN (Ours)
nohup python -u exp.py --model gw --dataset pems04 --GSL CLGSDN --n_prob 3 --select_channels 0 --device cuda:0 > ./logs/CLGSDNxGW_on_pems04.log &
# 不使用 CLGSDN (Baseline)
nohup python -u exp.py --model gw --dataset pems04 --GSL None --select_channels 0 --device cuda:0 > ./logs/GW_on_pems04.log &

# === Pems08 (选择速度通道 2) ===
# 使用 CLGSDN (Ours)
nohup python -u exp.py --model gw --dataset pems08 --GSL CLGSDN --n_prob 3 --select_channels 2 --device cuda:0 > ./logs/CLGSDNxGW_on_pems08.log &
# 不使用 CLGSDN (Baseline)
nohup python -u exp.py --model gw --dataset pems08 --GSL None --select_channels 2 --device cuda:0 > ./logs/GW_on_pems08.log &
```

-----

## 📊 7. 实验结果分析与解读

### 📝 结果解读
* **Experimental logs**: 实验进行时或结束后，你可以在 `./datasets/results/logs/` 找到相应的实验日志。
* **Final Report**: 程序结束后会在日志中输出验证集 Loss 最小那一轮的测试结果。

### 🛠️ 实验结果分析工具

* 我们提供了实验结果分析工具 `./tools/analyse_tools.py`，该脚本可自动提取所有实验的结果，并生成一个 Excel 文件，方便您直观对比不同实验配置的性能。
* 工具中包含一个 **Notes** 列，该列的内容来源于实验启动时通过 `--notes <your note>` 参数添加的文本备注。
* 该 Excel 文件包含 12 个时间步（Step 1-12）的详细 MAE。例如：Step 3 (15 min), Step 6 (30 min), Step 12 (1 hour)。

-----

## 📜 8. 引用

如果您觉得这项工作对您的研究有帮助，请考虑引用：

```bibtex
@ARTICLE{10757324,
  author={Peng, Peng and Chen, Xuewen and Zhang, Xudong and Tang, Haina and Shen, Hanji and Li, Jun},
  journal={IEEE Internet of Things Journal}, 
  title={CLGSDN: Contrastive-Learning-Based Graph Structure Denoising Network for Traffic Prediction}, 
  year={2025},
  volume={12},
  number={7},
  pages={8638-8652},
  doi={10.1109/JIOT.2024.3502517}
}
```