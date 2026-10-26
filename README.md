# MVCLEEA

MVCLEEA（Multi-View Contrastive Learning for Enzyme Function Prediction）是一个面向酶 EC 编号预测的多视图对比学习框架。

本仓库保存方法实现、ESM-2/PSSM/One-Hot 特征提取代码，以及论文使用的三个独立测试集（New-392、Price-149、New-815）。训练 split、预计算特征、训练日志、模型 checkpoint 和实验结果不纳入 Git。

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
├── data/
│   ├── README.md
│   └── datasets/
│       ├── new.csv          # New-392
│       ├── price.csv        # Price-149
│       └── new-815.csv      # New-815
├── docs/
│   └── METHOD.md
└── src/
    ├── train.py             # 模型、损失、训练、评估和消融入口
    ├── components.py        # 编码器、跨视图注意力和投影头
    ├── extract_esm2.py      # ESM-2 特征提取
    ├── extract_pssm.py      # PSI-BLAST PSSM 特征提取
    └── extract_onehot.py    # One-Hot 残基特征提取
```

## 环境

已验证环境为 Python 3.10、PyTorch 2.5.1 + CUDA 12.1、NumPy 2.2.6、scikit-learn 1.7.2 和 NVIDIA RTX 3090。

```bash
conda create -n mvcleea python=3.10 -y
conda activate mvcleea

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 数据布局

论文测试集位于 `data/datasets/`。训练代码仍从 `data/splits/` 读取训练划分，但训练 split 不随仓库发布，需要在本地准备。如果训练数据位于其他位置，可通过 `--project_root` 指定工作目录。

```text
workspace/
├── data/
│   ├── splits/                 # 用户本地提供，不纳入 Git
│   │   ├── split10.csv
│   │   ├── split50.csv
│   │   ├── split100.csv
│   │   ├── new.csv
│   │   ├── price.csv
│   │   └── new-815.csv
│   └── features/               # 生成结果，不纳入 Git
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

训练与评估代码当前统一从 `data/splits/` 读取 CSV。使用本仓库附带的测试集时，可将 `data/datasets/*.csv` 复制或软链接到工作目录的 `data/splits/` 下。

张量规格：

- ESM-2：每个蛋白质一个 1280 维 `.pt` 文件；
- PSSM：`N × 1024 × 20`，`float32`；
- One-Hot：`N × 1024 × 21`，`float32`；
- 序列长度不足 1024 时在末尾补零，padding mask 由 One-Hot 全零位置构造。

## 特征提取

三个脚本都读取 TAB 分隔的 CSV，使用第一列蛋白质 ID、第三列氨基酸序列，并按输入 CSV 的顺序输出 ID 映射。

### ESM-2

```bash
python src/extract_esm2.py \
  --data_dir data/datasets \
  --output_dir data/features/esm2 \
  --datasets new price new-815 \
  --batch_size 8 \
  --gpu 0
```

脚本使用 `esm2_t33_650M_UR50D` 的第 33 层表示，并只对真实残基位置做平均池化。

### PSSM

PSSM 由 NCBI PSI-BLAST 生成，默认执行 3 轮，E-value 阈值为 `1e-3`，截断或补齐到 1024 个残基，并对 20 列 PSSM 分数执行 sigmoid 归一化。BLAST 程序和数据库不随仓库发布。PSSM 数值依赖 BLAST 版本和数据库快照，严格复现实验时必须同时记录并固定所用数据库。

```bash
python src/extract_pssm.py \
  --data_dir data/datasets \
  --output_dir data/features/pssm \
  --datasets new price new-815 \
  --psiblast /path/to/psiblast \
  --blast_db /path/to/blast_db \
  --num_iterations 3 \
  --evalue 1e-3 \
  --workers 8
```

输出为 `<dataset>_pssm.npy` 和 `<dataset>_ids.pkl`。

### One-Hot

One-Hot 使用 `ACDEFGHIKLMNPQRSTVWY` 作为前 20 个通道，第 21 个通道表示非标准残基；超过 1024 个残基的序列截断，较短序列在末尾补零。

```bash
python src/extract_onehot.py \
  --data_dir data/datasets \
  --output_dir data/features/onehot \
  --datasets new price new-815
```

输出为 `<dataset>_onehot.npy` 和 `<dataset>_ids.pkl`。

## 训练

训练前需自行准备 `split10.csv`、`split50.csv` 和 `split100.csv`，放到 `workspace/data/splits/`。

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
