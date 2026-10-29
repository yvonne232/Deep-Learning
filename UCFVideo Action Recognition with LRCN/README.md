# UCF50 Video Action Recognition with LRCN

This repository contains a cleaned and improved LRCN pipeline for action recognition on `UCF50`.
The codebase was inherited with multiple correctness issues. This version fixes those issues,
adds experiment tracking, introduces model improvements, and includes tests/linting support.

## What Was Broken and How It Was Fixed

### 1) Data leakage in dataset split (fixed)

- **Problem:** split logic originally stratified at the clip level only. In UCF50, clips from the same
  group share actor/background/viewpoint, so train/test leakage occurred.
- **Fix:** split is now **group-aware** using `_gXX_` identifiers from clip names, so each group is
  assigned to exactly one of train/val/test.

### 2) Major sequence bug in LRCN (fixed)

- **Problem:** LSTM consumed frame features with incorrect sequence semantics.
- **Fix:** model now uses `batch_first=True`, packs padded sequences with true lengths, and masks padded
  timesteps during pooling.

### 3) Script/import/layout issues (fixed)

- Added missing `src/train.py`.
- Fixed runnable root-level scripts in `main/train.sh` and `main/test.sh`.
- Updated defaults from HMDB51 to UCF50 (`n_classes=50`).

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

Expected frame directory layout:

```text
UCF50/
  ApplyEyeMakeup/
    v_ApplyEyeMakeup_g01_c01/
      frame0.jpg
      frame1.jpg
      ...
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

Enable with `--use_wandb`. Logged metrics include:

- `train/loss`, `train/accuracy`
- `val/loss`, `val/accuracy`
- `train/lr`
- `test/accuracy` (eval mode)

To include submission evidence, attach screenshots of your W&B run dashboard showing train/val/test curves.

## Lint and Tests

Run pylint:

```bash
pylint src tests
```

Run pytest:

```bash
pytest -q
```

## Reproducibility

- Set seed via `--seed`.
- Splits are saved to `splits.npy`.
- Best checkpoint saved to `models/best_model_wts.pt`.

## Notes on 93% Target

Training quality depends on:

- frame extraction quality,
- GPU availability,
- chosen hyperparameters,
- and proper UCF50 preprocessing.

This repository now removes known correctness blockers so reported accuracy is meaningful and comparable.
