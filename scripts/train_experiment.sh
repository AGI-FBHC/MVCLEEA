#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/gej/.conda/envs/mvcleea/bin/python}"
SPLIT="${1:-split10}"
GPU="${GPU:-2}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LOG="${PROJECT_ROOT}/logs/train_${SPLIT}_$(date +%Y%m%d_%H%M%S).log"

cd "${PROJECT_ROOT}"
mkdir -p logs

{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "python=$(${PYTHON} --version 2>&1)"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used \
        --format=csv,noheader
    echo "split=${SPLIT}"
    echo "epochs=${EPOCHS}"
    echo "batch_size=${BATCH_SIZE}"
    echo "gpu=${GPU}"
    "${PYTHON}" src/train.py \
        --train_data "${SPLIT}" \
        --test_data new \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --gpu "${GPU}" \
        --retrain
} 2>&1 | tee "${LOG}"
