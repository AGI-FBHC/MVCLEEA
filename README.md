# MVCLEEA

### Multi-View Contrastive Learning for Enzyme Function Prediction

MVCLEEA predicts Enzyme Commission (EC) numbers from protein sequences by combining **ESM-2 semantics**, **PSSM evolutionary profiles**, and **one-hot residue identities**. Hierarchical multilabel contrastive learning organizes the protein embedding space, and cosine KNN voting assigns complete EC annotations.

This repository provides the method implementation, three feature-extraction utilities, and the New-392, Price-149, and New-815 evaluation datasets used in the paper.

[Overview](#overview) | [Results](#results) | [Installation](#installation) | [Datasets](#datasets) | [Feature extraction](#feature-extraction) | [Training and evaluation](#training-and-evaluation)

## Overview

![MVCLEEA framework: multi-view feature extraction, collaborative representation learning, and KNN-based EC prediction](docs/assets/mvcleea_framework.png)

*Figure 1 from the manuscript. The model uses amino-acid sequences and sequence-derived features as inputs.*

The framework comprises three modules:

1. **Multi-View Feature Extraction Module (MVFEM).** Frozen ESM-2 supplies a protein-level semantic vector. PSI-BLAST generates evolutionary profiles, and one-hot encoding records residue identities. Dedicated encoders map these views to a common 256-dimensional space.
2. **Multi-View Collaborative Representation Learning Module (MVCRL).** A learnable global token attends to the three views to obtain a consensus representation. A protein-specific gate combines the individual views and the consensus, followed by projection to a 128-dimensional embedding. Hierarchical multilabel supervision and ESM-guided alignment train the embedding network.
3. **KNN-based Hierarchical Prediction Module (KNN-HPM).** Cosine similarity retrieves ten reference proteins. Similarity-weighted voting returns complete EC labels with accumulated votes of at least 0.3, with the highest-vote label as a fallback. Evaluation at L1-L3 uses prefixes of the complete EC predictions.

Two proteins form a positive pair at an EC level when their label sets share at least one prefix at that level. The contrastive objective weights L1-L4 by `(0.1, 0.2, 0.3, 0.4)`. At the default alignment weight, the total loss is:

```text
loss = 0.9 * hierarchical_multilabel_contrastive_loss
     + 0.1 * ESM_guided_alignment_loss
```

See [METHOD.md](docs/METHOD.md) for the method outline.

## Results

Complete EC number prediction at **Level 4**, as reported in Table 2 of the manuscript:

| Dataset | Precision | Recall | F1 |
|---|---:|---:|---:|
| New-392 | 0.652 ± 0.001 | 0.777 ± 0.001 | **0.672 ± 0.001** |
| Price-149 | 0.594 ± 0.002 | 0.592 ± 0.002 | **0.559 ± 0.002** |
| New-815 | 0.729 ± 0.002 | 0.797 ± 0.002 | **0.743 ± 0.002** |

The manuscript reports means and standard deviations over ten runs. Each metric is calculated per EC category and averaged using true-label frequencies as weights. F1 is therefore calculated separately from the aggregate Precision and Recall.

## Installation

The reference software environment uses Python 3.10, PyTorch 2.5.1 with CUDA 12.1, NumPy 2.2.6, scikit-learn 1.7.2, and fair-esm 2.0.0. The paper experiments use an NVIDIA RTX 3090. The shell commands below assume Bash and run from the repository root.

```bash
git clone https://github.com/AGI-FBHC/MVCLEEA.git
cd MVCLEEA

conda create -n mvcleea python=3.10 -y
conda activate mvcleea

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

PSSM extraction also requires an NCBI BLAST+ installation containing `psiblast` and a protein BLAST database. Supply both paths when running the extractor. The ESM-2 extraction script requires a CUDA GPU and downloads the pretrained model on first use.

## Datasets

| Dataset | Proteins | Complete EC categories | Included file |
|---|---:|---:|---|
| New-392 | 392 | 177 | [new.csv](data/datasets/new.csv) |
| Price-149 | 149 | 56 | [price.csv](data/datasets/price.csv) |
| New-815 | 815 | 380 | [new-815.csv](data/datasets/new-815.csv) |

The `.csv` files are **TAB-separated**, with a header and three columns:

```text
Entry    EC number    Sequence
```

Multiple EC labels for a protein are separated by semicolons. Dataset provenance and SHA-256 checksums are listed in [data/README.md](data/README.md).

The repository includes the three external test sets. Prepare training and validation tables separately. The manuscript describes a source collection of 227,362 proteins covering 5,236 complete EC categories, with 5% held out for validation.

### Prepare a local workspace

The training entry point reads both reference and test tables from `data/splits/`. Copy the included test sets there and add your training table as `train.csv`:

```bash
mkdir -p data/splits data/features/esm2 data/features/pssm data/features/onehot
mkdir -p models results
cp data/datasets/*.csv data/splits/
```

Use the same dataset stem in the table and feature filenames. For example, `train.csv` corresponds to `train_pssm.npy`, `train_onehot.npy`, and their ID mappings.

```text
data/
  datasets/                  # Included external test tables
  splits/                    # Local tables consumed by train.py
    train.csv
    new.csv
    price.csv
    new-815.csv
  features/                  # Generated locally
    esm2/<protein_id>.pt
    pssm/<dataset>_pssm.npy
    pssm/<dataset>_ids.pkl
    onehot/<dataset>_onehot.npy
    onehot/<dataset>_ids.pkl
```

Training tables, generated features, checkpoints, and run outputs remain outside Git. Use `--project_root /path/to/workspace` when these runtime directories are stored elsewhere.

## Feature extraction

| View | Feature per protein | Extraction |
|---|---|---|
| Semantic | `1280` | Frozen `esm2_t33_650M_UR50D`, layer 33; mean over real residues |
| Evolutionary | `1024 x 20` | PSI-BLAST substitution scores followed by sigmoid normalization |
| Residue identity | `1024 x 21` | 20 standard amino acids plus one channel for non-standard residues |

PSSM and one-hot arrays use `float32` and include a leading protein dimension: `(N, 1024, 20)` and `(N, 1024, 21)`, respectively. Their ID files record the row order. ESM-2 stores one tensor per protein ID.

The following commands extract features for the included test sets. After preparing `data/splits/train.csv`, run the same commands with `--data_dir data/splits --datasets train` to generate the training features.

### ESM-2

```bash
python src/extract_esm2.py \
  --data_dir data/datasets \
  --output_dir data/features/esm2 \
  --datasets new price new-815 \
  --batch_size 2 \
  --gpu 0
```

Mean pooling excludes BOS, EOS, and batch-padding tokens. The script reuses existing `<protein_id>.pt` files; use a separate output directory when changing the model or pooling procedure.

### PSSM

```bash
python src/extract_pssm.py \
  --data_dir data/datasets \
  --output_dir data/features/pssm \
  --datasets new price new-815 \
  --psiblast /path/to/psiblast \
  --blast_db /path/to/protein_database \
  --num_iterations 3 \
  --evalue 1e-3 \
  --max_length 1024 \
  --workers 4
```

Replace the executable and database paths with your installation. The script writes `<dataset>_pssm.npy` and `<dataset>_ids.pkl`, and raises an error if PSI-BLAST fails. Record the BLAST+ version, source database release, and database-building command when preparing an experiment.

The parser applies sigmoid to a zero-initialized score matrix, so unused rows can contain `0.5`. The model derives valid positions from the one-hot view and masks PSSM padding before encoding. Generate both views from the same sequences and keep their ID mappings aligned.

### One-hot

```bash
python src/extract_onehot.py \
  --data_dir data/datasets \
  --output_dir data/features/onehot \
  --datasets new price new-815 \
  --max_length 1024
```

Channels 0-19 follow `ACDEFGHIKLMNPQRSTVWY`; channel 20 represents non-standard residues. The script truncates sequences to 1024 positions, pads shorter sequences with zeros, and writes `<dataset>_onehot.npy` and `<dataset>_ids.pkl`.

## Training and evaluation

Prepare all three feature types for the training reference set and the selected test set before running the model. Reference proteins and their EC annotations are required for KNN evaluation as well as for training.

### Train a model

```bash
python src/train.py \
  --project_root . \
  --train_data train \
  --test_data new \
  --epochs 100 \
  --batch_size 128 \
  --lr 0.0005 \
  --alpha 0.1 \
  --seed 42 \
  --gpu 0 \
  --experiment_name mvcleea_train_seed42 \
  --retrain
```

The script evaluates the trained model at EC levels L1-L4 and saves:

- Model weights: `models/mvcleea_train_seed42.pth`
- Metrics and run settings: `results/mvcleea_train_seed42.json`

The training loop selects weights using the lowest training loss. It does not create a validation partition or tune hyperparameters automatically. To follow the paper protocol, prepare the training/validation separation first, select settings using validation data, and keep the external test sets for final evaluation.

### Evaluate saved weights

```bash
python src/train.py \
  --project_root . \
  --train_data train \
  --test_data new \
  --experiment_name mvcleea_train_seed42 \
  --gpu 0
```

This loads `models/mvcleea_train_seed42.pth` when the file exists. If the checkpoint is absent, the current entry point starts training automatically. Change `--test_data` to `price` or `new-815` to evaluate the other datasets. Each call writes to the same experiment-named JSON file; preserve it under a dataset-specific name before evaluating the next test set.

KNN uses `K = 10` and a vote threshold of `0.3`, fixed inside `evaluate_model()`. `--seed` controls one training run; repeat training with distinct seeds and experiment names to collect repeated-run statistics.

### Ablations

Use `--ablation` together with a distinct `--experiment_name` to select a variant:

| Group | Available values |
|---|---|
| Full model | `full` |
| Individual views | `esm_only`, `pssm_only`, `onehot_only` |
| View pairs | `pssm_esm2`, `esm_onehot`, `pssm_onehot` |
| Component removal | `no_mask`, `no_alignment`, `no_cross_attention`, `no_dynamic_fusion`, `no_pssm`, `no_onehot` |
| Supervision | `single_label`, `l4_only` |
| Stepwise structure | `struct_basic`, `struct_mask`, `struct_align`, `struct_attn` |

For example, append `--ablation no_cross_attention` and set `--experiment_name no_cross_attention_seed42` in the training command. Keep the data, seed, and remaining settings consistent across variants.

## Repository structure

```text
MVCLEEA/
  README.md
  requirements.txt
  data/
    README.md
    datasets/{new,price,new-815}.csv
  docs/
    METHOD.md
    assets/mvcleea_framework.png
  src/
    components.py            # View encoders, attention, projection head
    train.py                 # Model, loss, training, and KNN evaluation
    extract_esm2.py
    extract_pssm.py
    extract_onehot.py
```

## Paper and acknowledgements

**Multi-View Contrastive Learning for Enzyme Function Prediction**

Jie Ge, Haijun Shi, Zhihao Liu, Guanjin Wang, Zhaohong Deng, and Hailong Yang.

We use ESM-2 for pretrained sequence representations. The evaluation datasets follow [CLEAN](https://github.com/tttianhao/CLEAN) and [MAPred](https://github.com/Rongdingyi/MAPred); see the [dataset provenance](data/README.md#provenance) for the file-to-source mapping.
