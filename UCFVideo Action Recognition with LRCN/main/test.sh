#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

PYTHONPATH=src python src/run.py \
    --ckpt "${PROJECT_ROOT}/models/best_model_wts.pt" \
    --frame_dir "${PROJECT_ROOT}/UCF50_frames" \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --use_wandb \
    --wandb_project ucf50-lrcn \
    --mode eval
