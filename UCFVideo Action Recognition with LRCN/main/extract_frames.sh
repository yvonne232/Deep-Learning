#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

PYTHONPATH=src python src/extract_frames.py \
    --video_dir "${PROJECT_ROOT}/UCF50" \
    --frame_dir "${PROJECT_ROOT}/UCF50_frames_32" \
    --n_frames 32
