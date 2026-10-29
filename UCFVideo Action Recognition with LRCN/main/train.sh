#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

PYTHONPATH=src python src/run.py \
    --frame_dir "${PROJECT_ROOT}/UCF50_frames" \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --cnn_backbone resnet50 \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --dropout 0.5 \
    --rnn_hidden_size 256 \
    --rnn_n_layers 2 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --label_smoothing 0.1 \
    --n_epochs 80 \
    --use_wandb \
    --wandb_project ucf50-lrcn \
    --mode train
