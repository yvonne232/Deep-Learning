# UCF50 Video Action Recognition with LRCN

This repository contains a cleaned and improved LRCN pipeline for action recognition on `UCF50`. This version fixes those issues, adds experiment tracking, introduces model improvements, and includes tests/linting support.

## What Was Broken and How It Was Fixed

### 1) Data leakage in dataset split (fixed)

- **Problem:** split logic originally stratified at the clip level only. In UCF50, clips from the same
  group share actor/background/viewpoint, so train/test leakage occurred.
- **Fix:** split is now **group-aware** using `_gXX_` identifiers from clip names, so each group is
  assigned to exactly one of train/val/test.

### 2) Sequence bug in LRCN (fixed)

- **Problem:** LSTM consumed frame features with incorrect sequence semantics.
- **Fix:** model now uses `batch_first=True`, packs padded sequences with true lengths, and masks padded
  timesteps during pooling.


## Model Improvements Implemented

1. **Bidirectional LSTM** (`--bidirectional`) for richer temporal context.
2. **Temporal attention pooling + masked mean pooling** for better sequence aggregation.
3. Additional training improvements:
   - stronger augmentation (`RandomResizedCrop`, `ColorJitter`)
   - label smoothing
   - weight decay regularization

## Project Structure

- `src/models.py`: LRCN architecture.
- `src/video_datasets.py`: dataset loader, group-aware splitting, collate functions.
- `src/train.py`: training loop + optional W&B logging.
- `src/test.py`: evaluation helpers.
- `src/run.py`: main train/eval CLI.
- `main/train.sh`: training script for repo root execution.
- `main/test.sh`: evaluation script for repo root execution.
- `tests/`: pytest unit tests.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
bash main/train.sh
```

Or run directly:

```bash
python src/run.py \
  --frame_dir ./UCF50 \
  --mode train \
  --batch_size 8 \
  --n_classes 50 \
  --use_wandb \
  --wandb_project ucf50-lrcn
```

## Evaluate

```bash
bash main/test.sh
```

## Weights & Biases Logging

Enable with `--use_wandb`.

## Notes on 93% Target

I have trained multiple times (6-7 times) and tried changing hyperparameters; my best result is 89.28% test accuracy. I am stuck getting it to 93% after trying the approaches I could think of, and I would be interested in how to get it higher by changing the model structure.
