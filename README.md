# MVCLEEA

MVCLEEA（Multi-View Contrastive Learning for Enzyme Function Prediction）是一个面向酶 EC 编号预测的多视图对比学习框架。

本仓库只保存方法实现，不包含数据集、预计算特征、训练日志、模型 checkpoint 或实验结果。训练数据与特征应由使用者按本文说明在本地准备。

## 方法组成

- **ESM-2 语义视图**：对真实残基位置进行平均池化，排除 BOS、EOS 和 padding；
- **PSSM 进化视图**：使用带残差连接的序列一维卷积编码；
- **One-Hot 残基视图**：保留氨基酸身份和局部顺序信息；
- **跨视图注意力**：通过全局 token 聚合三类视图；
- **动态特征融合**：为各视图生成逐蛋白权重，并保留 ESM-2 残差；
- **层级对比学习**：按 EC L1-L4 前缀构造多标签正样本；
- **教师对齐**：将 PSSM 与 One-Hot 表示弱对齐到停止梯度的 ESM-2 表示；
- **KNN 预测**：使用最终 128 维表示和余弦 KNN 预测完整 EC 编号。

详细方法说明见 [`docs/METHOD.md`](docs/METHOD.md)。

## 目录结构

```text
MVCLEEA/
├── README.md
├── requirements.txt
├── docs/
│   └── METHOD.md
└── src/
    ├── train.py             # 模型、损失、训练、评估和消融入口
    ├── components.py        # 编码器、跨视图注意力和投影头
    └── extract_esm2.py      # ESM-2 特征提取
```

## 环境

已验证环境为 Python 3.10、PyTorch 2.5.1 + CUDA 12.1、NumPy 2.2.6、scikit-learn 1.7.2 和 NVIDIA RTX 3090。

```bash
conda create -n mvcleea python=3.10 -y
conda activate mvcleea

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 外部数据布局

训练代码默认从仓库根目录读取 `data/`、`models/` 和 `results/`。如果这些目录位于其他位置，可通过 `--project_root` 指定。

```text
workspace/
├── data/
│   ├── splits/
│   │   ├── split10.csv
│   │   ├── split50.csv
│   │   ├── split100.csv
│   │   └── new.csv
│   └── features/
│       ├── esm2/
│       │   └── <UniProt_ID>.pt
│       ├── esm2_split100_cache.npy
│       ├── esm2_split100_cache_ids.pkl
│       ├── pssm/
│       │   ├── <split>_pssm.npy
│       │   └── <split>_ids.pkl
│       └── onehot/
│           ├── <split>_onehot.npy
│           └── <split>_ids.pkl
├── models/
└── results/
```

张量规格：

- ESM-2：每个蛋白质一个 1280 维 `.pt` 文件；
- PSSM：`N × 1024 × 20`，`float32`；
- One-Hot：`N × 1024 × 21`，`float32`；
- 序列长度不足 1024 时在末尾补零，padding mask 由 One-Hot 全零位置构造。

## 提取 ESM-2 特征

数据划分 CSV 使用 TAB 分隔，ESM-2 提取脚本读取第一列的蛋白质 ID 和第三列的序列。

```bash
python src/extract_esm2.py \
  --data_dir workspace/data/splits \
  --output_dir workspace/data/features/esm2 \
  --datasets split10 split50 split100 new \
  --batch_size 8 \
  --gpu 0
```

PSSM 与 One-Hot 数组需在训练前单独生成。

## 训练

```bash
python src/train.py \
  --train_data split10 \
  --test_data new \
  --epochs 100 \
  --batch_size 128 \
  --gpu 0 \
  --seed 42 \
  --retrain \
  --project_root workspace
```

将 `split10` 替换为 `split50` 或 `split100` 可运行其他训练规模。模型保存到 `workspace/models/`，评估结果保存到 `workspace/results/`。

## 评估已有模型

```bash
python src/train.py \
  --train_data split100 \
  --test_data new \
  --gpu 0 \
  --project_root workspace
```

不添加 `--retrain` 时，程序会加载对应的已有模型 checkpoint 并进行评估。

## 消融配置

`src/train.py` 支持以下 `--ablation` 取值：

```text
full
no_mask
single_label
l4_only
no_dynamic_fusion
no_cross_attention
no_alignment
no_pssm
no_onehot
pssm_only
pssm_onehot
pssm_esm2
esm_only
onehot_only
esm_onehot
struct_basic
struct_mask
struct_align
struct_attn
```

例如：

```bash
python src/train.py \
  --train_data split10 \
  --test_data new \
  --gpu 0 \
  --seed 42 \
  --ablation no_cross_attention \
  --experiment_name ablation_split10_no_cross_attention_seed42 \
  --retrain \
  --project_root workspace
```

训练中断后可通过 `--resume_path` 指定断点，并使用 `--resume_epoch` 指定已完成的 epoch 数。
