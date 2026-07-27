#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/gej/.conda/envs/mvcleea/bin/python}"
SPLIT="${1:-split10}"
GPU="${GPU:-2}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"

ABLATIONS=(
    full
    no_mask
    single_label
    l4_only
    no_dynamic_fusion
    no_cross_attention
    no_alignment
    no_pssm
    no_onehot
)

cd "${PROJECT_ROOT}"
mkdir -p logs/ablations results models

for ablation in "${ABLATIONS[@]}"; do
    name="ablation_${SPLIT}_${ablation}_seed${SEED}"
    log="logs/ablations/${name}.log"
    start_epoch="$(date +%s)"
    {
        echo "============================================================"
        echo "MVCLEEA v2 ablation experiment"
        echo "============================================================"
        echo "experiment=${name}"
        echo "ablation=${ablation}"
        echo "train_split=${SPLIT}"
        echo "test_split=new"
        echo "seed=${SEED}"
        echo "epochs=${EPOCHS}"
        echo "batch_size=${BATCH_SIZE}"
        echo "gpu=${GPU}"
        echo "start_time=$(date --iso-8601=seconds)"
        echo "hostname=$(hostname)"
        echo "git_commit=$(git rev-parse HEAD)"
        echo "python=$(${PYTHON} --version 2>&1)"
        "${PYTHON}" - <<'PY'
import numpy
import sklearn
import torch
print(f'torch={torch.__version__}')
print(f'cuda={torch.version.cuda}')
print(f'numpy={numpy.__version__}')
print(f'scikit_learn={sklearn.__version__}')
PY
        nvidia-smi --query-gpu=index,name,memory.total,memory.used \
            --format=csv,noheader
        echo "command=${PYTHON} src/train.py --train_data ${SPLIT} --test_data new --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --gpu ${GPU} --seed ${SEED} --ablation ${ablation} --experiment_name ${name} --retrain"
        "${PYTHON}" src/train.py \
            --train_data "${SPLIT}" \
            --test_data new \
            --epochs "${EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --gpu "${GPU}" \
            --seed "${SEED}" \
            --log_interval 1 \
            --ablation "${ablation}" \
            --experiment_name "${name}" \
            --retrain
        end_epoch="$(date +%s)"
        echo "end_time=$(date --iso-8601=seconds)"
        echo "elapsed_seconds=$((end_epoch - start_epoch))"
    } 2>&1 | tee "${log}"
done

"${PYTHON}" scripts/summarize_ablation.py --split "${SPLIT}" --seed "${SEED}"
