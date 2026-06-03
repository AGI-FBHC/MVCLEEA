# MVCLEEA

Multi-View Contrastive Learning for Enzyme Function Annotation (EC Number Prediction).

## Environment Setup

```bash
git clone <repo-url>
cd MVCLEEA
conda env create -f environment.yml
conda activate mvcleea
```

If you don't have a CUDA-compatible GPU, remove `pytorch-cuda=12.1` from `environment.yml` before creating the environment, or install the CPU-only PyTorch:

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## Project Structure

```
MVCLEEA/
├── models/                  # Model modules
│   ├── semantic_view.py     # Semantic View (MLP on ESM2 embeddings)
│   ├── evolutionary_view.py # Evolutionary View (CNN on PSSM)
│   ├── syntactic_view.py    # Syntactic View (CNN on One-hot)
│   ├── cross_attention.py   # Cross-Attention Pooling → Shared View
│   ├── adaptive_fusion.py   # Adaptive Feature Fusion
│   ├── classifier.py        # Hierarchical Classifier
│   └── mvcleea.py           # Main MVCLEEA model
├── features/                # Feature extraction from raw sequences
│   ├── esm2_extractor.py    # ESM2 embedding extraction (HuggingFace)
│   ├── pssm_extractor.py    # PSSM generation (PSI-BLAST)
│   └── onehot_extractor.py  # One-hot encoding
├── losses/                  # Loss functions
│   ├── contrastive_loss.py  # InfoNCE cross-view contrastive loss
│   └── joint_loss.py        # Joint loss (BCE + contrastive)
├── data/                    # Raw label/sequence mappings
│   ├── id_ec*.pkl           # Protein ID → EC number mappings
│   └── id_sequence*.pkl     # Protein ID → sequence mappings
├── configs/default.yaml     # Default hyperparameters
├── prepare_data.py          # Feature extraction & data preprocessing
├── train.py                 # Training script
├── test_model.py            # Forward pass verification
└── main.py                  # Entry point
```

## Data Preparation

### 1. Raw data

`data/` contains protein ID mappings (`.pkl` files). Three dataset splits are available:

| File                                        | Proteins | Description          |
| ------------------------------------------- | -------- | -------------------- |
| `id_ec.pkl` / `id_sequence.pkl`             | 227,177  | Main training set    |
| `id_ec_new.pkl` / `id_sequence_new.pkl`     | 392      | New evaluation set   |
| `id_ec_price.pkl` / `id_sequence_price.pkl` | 149      | Price evaluation set |

### 2. Extract features

Feature extraction runs from raw amino acid sequences. Each feature type can be toggled independently:

```bash
# Full pipeline: ESM2 + PSSM + One-hot
python prepare_data.py --data-dir data --output-dir data/processed --device cuda

# ESM2 only (skip PSSM if no BLAST database available)
python prepare_data.py --data-dir data --output-dir data/processed --skip-pssm --device cuda

# One-hot only (fastest, no external dependencies)
python prepare_data.py --data-dir data --output-dir data/processed --skip-esm2 --skip-pssm

# Process a smaller evaluation set
python prepare_data.py --data-dir data --output-dir data/processed --dataset new --device cuda
```

**PSSM requires PSI-BLAST** with a reference database (e.g., UniRef90):

```bash
python prepare_data.py --data-dir data --output-dir data/processed \
    --blast-db /path/to/uniref90 --blast-dir /path/to/blast+/bin
```

### 3. Output

`data/processed/` will contain:

- `esm2_main.npy` — (N, 1280) ESM2 embeddings
- `pssm_main.npy` — (N, 1024, 20) PSSM matrices (Sigmoid-normalized)
- `onehot_main.npy` — (N, 1024, 21) One-hot encoded sequences
- `labels_main.npy` — (N, C) Multi-hot EC annotations
- `ec_list.pkl` — Sorted list of all EC numbers

## Usage

### Verify the architecture

```bash
python test_model.py
```

### Train

```bash
python train.py
python train.py --config configs/your_config.yaml
```

The training script automatically loads from `data/processed/`. If no data is found, it falls back to dummy data for testing.

### Quick start

```bash
python main.py test    # Architecture verification
python main.py train   # Train with default config
```

## Key Hyperparameters

| Parameter       | Default | Description                                        |
| --------------- | ------- | -------------------------------------------------- |
| `lambda_val`    | 0.15    | Balance between BCE and contrastive loss (0.1–0.2) |
| `temperature`   | 0.07    | Temperature for InfoNCE loss                       |
| `num_classes`   | 588     | Total EC number classes                            |
| `feature_dim`   | 128     | Output dimension per view network                  |
| `batch_size`    | 32      | Training batch size                                |
| `learning_rate` | 1e-4    | Adam learning rate                                 |

