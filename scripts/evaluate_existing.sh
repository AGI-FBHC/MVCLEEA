#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/gej/.conda/envs/mvcleea/bin/python}"
GPU="${GPU:-2}"

cd "${PROJECT_ROOT}"
mkdir -p logs

for split in split10 split50; do
    log="logs/recheck_${split}.log"
    {
        echo "timestamp=$(date --iso-8601=seconds)"
        echo "command=${PYTHON} src/train.py --train_data ${split} --test_data new --gpu ${GPU}"
        echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo uncommitted)"
        "${PYTHON}" src/train.py \
            --train_data "${split}" \
            --test_data new \
            --gpu "${GPU}"
    } 2>&1 | tee "${log}"
done
