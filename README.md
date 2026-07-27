# MVCLEEA v2

Standalone implementation of MVCLEEA v2 for EC-number prediction using:

- length-aware ESM-2 embeddings;
- mask-aware PSSM and OneHot residue encoders;
- cross-attention pooling;
- per-protein dynamic feature fusion;
- multi-label hierarchical supervised contrastive learning;
- cosine KNN prediction.

## Project layout

```text
MVCLEEA/
├── src/                 # extraction, model components and training
├── scripts/             # reproducible experiment entry points
├── data/
│   ├── splits/          # split10/split50/split100/new TSV files
│   ├── features/
│   │   ├── esm2/        # per-protein 1280-d ESM-2 embeddings
│   │   ├── pssm/        # split-level PSSM arrays
│   │   └── onehot/      # split-level OneHot arrays
│   └── MANIFEST.json    # local artifact inventory
├── models/              # trained split10/split50/split100 checkpoints
├── logs/                # complete extraction/training/evaluation logs
├── results/             # machine-readable metrics and comparisons
└── docs/                # method and reproducibility notes
```

Large files under `data/` are intentionally excluded from Git. They are
present in this server checkout and documented by `data/MANIFEST.json`.
Use Git LFS, a GitHub Release, or a scientific data repository when
publishing them.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The server experiments used Python 3.10, PyTorch, CUDA and GPU index 2.

## Evaluate an existing model

```bash
python src/train.py \
  --train_data split10 \
  --test_data new \
  --gpu 2
```

## Train

```bash
python src/train.py \
  --train_data split10 \
  --test_data new \
  --epochs 100 \
  --batch_size 128 \
  --gpu 2 \
  --retrain
```

## Main results

| Train split | L1 F1 | L2 F1 | L3 F1 | L4 F1 |
|---|---:|---:|---:|---:|
| split10 | 0.8783 | 0.8513 | 0.7468 | 0.3363 |
| split50 | 0.9510 | 0.9301 | 0.8431 | 0.4882 |
| split100 | 0.9745 | 0.9567 | 0.9370 | 0.6424 |

See `results/metrics.json` and `logs/` for full precision, recall and AUC.
