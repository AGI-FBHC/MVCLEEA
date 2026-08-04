# MVCLEEA

MVCLEEA（Multi-View Contrastive Learning for Enzyme Function Prediction）是一个面向酶 EC 编号预测的多视图对比学习框架。当前版本联合建模三类互补的序列衍生特征：

- **ESM-2 语义视图**：描述蛋白质序列的上下文语义；
- **PSSM 进化视图**：描述残基位置的进化保守性；
- **One-Hot 残基视图**：保留原始氨基酸身份与局部排列信息。

模型使用带 padding mask 的残基编码器、跨视图注意力、特征融合和多标签层级监督对比学习生成 128 维表示，最后通过余弦 KNN 预测完整 EC 编号。

## 目录结构

```text
MVCLEEA/
├── src/
│   ├── train.py             # 训练、评估和消融入口
│   ├── components.py        # 注意力、投影头等模型组件
│   └── extract_esm2.py      # ESM-2 特征提取
├── scripts/                 # 主实验、消融和结果汇总脚本
├── data/
│   ├── splits/              # split10、split50、split100、New-392
│   ├── features/            # 本地预计算特征（不纳入 Git）
│   └── MANIFEST.json        # 特征形状与文件规模清单
├── models/                  # 三个主模型 checkpoint
├── results/                 # 主实验和消融结果
├── logs/                    # 可复现实验日志
└── docs/                    # 方法与消融说明
```

## 实验环境

服务器上的已验证环境如下：

| 组件 | 版本 |
|---|---|
| 操作系统 | Linux |
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| CUDA Runtime | 12.1 |
| NumPy | 2.2.6 |
| scikit-learn | 1.7.2 |
| pandas | 2.3.3 |
| GPU | NVIDIA GeForce RTX 3090 24 GB |

建议使用 Conda 创建独立环境：

```bash
conda create -n mvcleea python=3.10.20 -y
conda activate mvcleea

# CUDA 12.1 对应的 PyTorch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

安装完成后可检查环境：

```bash
python - <<'PY'
import torch, numpy, sklearn
print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
PY
```

## 数据与特征

仓库包含四个 TSV 格式的数据划分：

| 文件 | 蛋白质数量 | 用途 |
|---|---:|---|
| `split10.csv` | 7,757 | 低同源训练集 |
| `split50.csv` | 29,430 | 中等规模训练集 |
| `split100.csv` | 227,362 | 全量训练集 |
| `new.csv` | 392 | New-392 独立测试集 |

预计算特征约 44 GB，不适合直接存入 GitHub。训练前需按以下结构放置：

```text
data/features/
├── esm2/
│   └── <UniProt_ID>.pt
├── pssm/
│   ├── split10_pssm.npy
│   ├── split50_pssm.npy
│   ├── split100_pssm.npy
│   └── new_pssm.npy
└── onehot/
    ├── split10_onehot.npy
    ├── split50_onehot.npy
    ├── split100_onehot.npy
    └── new_onehot.npy
```

主要张量规格：

- ESM-2：每个蛋白质一个 `1280` 维 `.pt` 文件；
- PSSM：`N × 1024 × 20`，`float32`；
- One-Hot：`N × 1024 × 21`，`float32`；
- 长度不足 1024 的序列在末尾补零，模型根据 One-Hot 全零位置构造 padding mask。

ESM-2 特征可由仓库脚本提取：

```bash
python src/extract_esm2.py \
  --data_dir data/splits \
  --output_dir data/features/esm2 \
  --datasets split10 split50 split100 new \
  --batch_size 8 \
  --gpu 2
```

PSSM 与 One-Hot 数组需预先生成或从项目数据发布页面获取。完整文件大小和形状见 `data/MANIFEST.json`。

## 训练与评估

以下命令默认使用 GPU 2，可按机器情况修改 `--gpu`。

### 训练主模型

```bash
python src/train.py \
  --train_data split10 \
  --test_data new \
  --epochs 100 \
  --batch_size 128 \
  --gpu 2 \
  --seed 42 \
  --retrain
```

也可通过日志化脚本运行：

```bash
GPU=2 EPOCHS=100 BATCH_SIZE=128 \
  bash scripts/train_experiment.sh split10
```

将 `split10` 替换为 `split50` 或 `split100` 即可运行对应规模实验。

### 评估已有模型

不添加 `--retrain` 时，程序加载 `models/mvcleea_v2_<split>.pth`：

```bash
python src/train.py \
  --train_data split100 \
  --test_data new \
  --gpu 2
```

### 消融实验

运行指定训练规模的完整消融套件：

```bash
GPU=2 SEED=42 EPOCHS=100 BATCH_SIZE=128 \
  bash scripts/run_ablation_suite.sh split10
```

运行单个消融配置：

```bash
python src/train.py \
  --train_data split10 \
  --test_data new \
  --gpu 2 \
  --seed 42 \
  --ablation no_cross_attention \
  --experiment_name ablation_split10_no_cross_attention_seed42 \
  --retrain
```

可用配置：`full`、`no_mask`、`single_label`、`l4_only`、`no_dynamic_fusion`、`no_cross_attention`、`no_alignment`、`no_pssm`、`no_onehot`。

汇总已有消融结果：

```bash
python scripts/summarize_ablation.py --split split100 --seed 42
```

## 结果

固定随机种子 `42` 的完整模型结果如下，指标为 weighted F1：

| 训练集 | L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|
| split10 | 0.8498 | 0.8307 | 0.6815 | 0.3043 |
| split50 | 0.9744 | 0.9430 | 0.9077 | 0.5481 |
| split100 | 0.9640 | 0.9473 | 0.9264 | **0.6720** |

完整 Precision、Recall、F1 和消融结果位于 `results/`，对应训练过程位于 `logs/ablations/`。

## 复现说明

- 论文级结果建议使用多个随机种子并报告均值与标准差；当前消融实验统一使用 `seed=42`。
- New-392 应仅作为独立测试集，不应被用于选择融合权重、KNN 阈值或最佳随机种子。
- 使用 PSSM 进行时间外评估时，应记录搜索数据库及其发布日期，并确保不引入测试期之后的序列信息。
- 预训练特征、主模型和数据文件的体积信息可通过 `python scripts/build_manifest.py`重新生成。

## 许可证与引用

正式许可证和论文引用信息将在论文录用或代码发布策略确定后补充。使用本项目数据时，请同时遵守 UniProt、ESM 和相关数据源的许可要求。
