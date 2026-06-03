# MVCLEEA

Multi-View Contrastive Learning for Enzyme Function Annotation (EC Number Prediction).

## 环境配置

```bash
git clone <repo-url>
cd MVCLEEA
conda env create -f environment.yml
conda activate mvcleea
```

如果你没有CUDA兼容的GPU，在创建环境前从`environment.yml`中移除`pytorch-cuda=12.1`，或者安装CPU-only版本的PyTorch：

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## 项目结构

```
MVCLEEA/
├── models/                  # 模型模块
│   ├── semantic_view.py     # 语义视图（基于ESM2嵌入的MLP）
│   ├── evolutionary_view.py # 进化视图（基于PSSM的CNN）
│   ├── syntactic_view.py   # 句法视图（基于One-hot的CNN）
│   ├── cross_attention.py  # 交叉注意力池化 → 共享视图
│   ├── adaptive_fusion.py  # 自适应特征融合
│   ├── classifier.py       # 分层分类器
│   └── mvcleea.py          # 主模型MVCLEEA
├── features/               # 从原始序列中提取特征
│   ├── esm2_extractor.py   # ESM2嵌入提取（HuggingFace）
│   ├── pssm_extractor.py   # PSSM生成（PSI-BLAST）
│   └── onehot_extractor.py # One-hot编码
├── losses/                 # 损失函数
│   ├── contrastive_loss.py # InfoNCE跨视图对比损失
│   └── joint_loss.py       # 联合损失（BCE + 对比损失）
├── data/                   # 原始标签/序列映射
│   ├── id_ec*.pkl          # 蛋白质ID → EC编号映射
│   └── id_sequence*.pkl    # 蛋白质ID → 序列映射
├── configs/default.yaml    # 默认超参数
├── prepare_data.py         # 特征提取与数据预处理
├── train.py                # 训练脚本
├── test_model.py           # 前向传播验证
└── main.py                 # 入口点
```

## 数据准备

### 1. 原始数据

`data/`包含蛋白质ID映射（`.pkl`文件）。提供三个数据集划分：

| 文件                                              | 蛋白质数量 | 描述           |
| ------------------------------------------------- | --------- | -------------- |
| `id_ec.pkl` / `id_sequence.pkl`                   | 227,177   | 主要训练集     |
| `id_ec_new.pkl` / `id_sequence_new.pkl`           | 392       | 新评估集       |
| `id_ec_price.pkl` / `id_sequence_price.pkl`       | 149       | Price评估集    |

### 2. 提取特征

特征提取从原始氨基酸序列运行。每种特征类型可以独立开关：

```bash
# 完整流程：ESM2 + PSSM + One-hot
python prepare_data.py --data-dir data --output-dir data/processed --device cuda

# 仅ESM2（没有BLAST数据库时跳过PSSM）
python prepare_data.py --data-dir data --output-dir data/processed --skip-pssm --device cuda

# 仅One-hot（最快，无外部依赖）
python prepare_data.py --data-dir data --output-dir data/processed --skip-esm2 --skip-pssm

# 处理较小的评估集
python prepare_data.py --data-dir data --output-dir data/processed --dataset new --device cuda
```

**PSSM需要PSI-BLAST**和参考数据库（如UniRef90）：

```bash
python prepare_data.py --data-dir data --output-dir data/processed \
    --blast-db /path/to/uniref90 --blast-dir /path/to/blast+/bin
```

### 3. 输出

`data/processed/`将包含：

- `esm2_main.npy` — (N, 1280) ESM2嵌入
- `pssm_main.npy` — (N, 1024, 20) PSSM矩阵（Sigmoid归一化）
- `onehot_main.npy` — (N, 1024, 21) One-hot编码序列
- `labels_main.npy` — (N, C) 多标签EC注释
- `ec_list.pkl` — 所有EC编号的排序列表

## 使用方法

### 验证架构

```bash
python test_model.py
```

### 训练

```bash
python train.py
python train.py --config configs/your_config.yaml
```

训练脚本自动从`data/processed/`加载数据。如果没有找到数据，它会回退到虚拟数据进行测试。

### 快速开始

```bash
python main.py test    # 架构验证
python main.py train   # 使用默认配置训练
```

## 主要超参数

| 参数            | 默认值   | 描述                                      |
| -------------- | ------- | ---------------------------------------- |
| `lambda_val`   | 0.15    | BCE损失与对比损失之间的平衡系数（0.1–0.2） |
| `temperature`  | 0.07    | InfoNCE损失的温度参数                     |
| `num_classes`  | 588     | EC编号类别总数                            |
| `feature_dim`   | 128     | 每个视图网络的输出维度                    |
| `batch_size`   | 32      | 训练批量大小                              |
| `learning_rate`| 1e-4    | Adam学习率                               |
