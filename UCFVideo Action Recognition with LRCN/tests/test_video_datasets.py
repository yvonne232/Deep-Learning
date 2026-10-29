"""Tests for dataset loading/splitting and collate behavior."""

import os
import re

import pytest

torch = pytest.importorskip("torch")
from PIL import Image

from src.video_datasets import _select_frame_paths, collate_fn_rnn, dataset_split, load_dataset


def _make_img(path):
    Image.new("RGB", (8, 8), color=(120, 30, 200)).save(path)


def _group_keys(split):
    """(class_label, gXX) pairs — the unit the split is supposed to keep together."""
    keys = set()
    for path, label in split:
        match = re.search(r"_g(\d+)_", os.path.basename(path))
        keys.add((int(label), int(match.group(1))))
    return keys


def test_group_split_prevents_leakage(tmp_path):
    # Two classes x four groups each x two clips each.
    for class_name in ("class_a", "class_b"):
        class_dir = tmp_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for group_idx in range(1, 5):
            for clip_idx in range(1, 3):
                vid_dir = class_dir / f"v_{class_name}_g{group_idx:02d}_c{clip_idx:02d}"
                vid_dir.mkdir(parents=True, exist_ok=True)
                _make_img(vid_dir / "frame0.jpg")

    vid_dataset, _ = load_dataset(str(tmp_path))
    tr_split, val_split, ts_split = dataset_split(vid_dataset, tr_ratio=0.5, ts_ratio=0.25, seed=7)

    train_g = _group_keys(tr_split)
    val_g = _group_keys(val_split)
    test_g = _group_keys(ts_split)

    assert train_g.isdisjoint(val_g)
    assert train_g.isdisjoint(test_g)
    assert val_g.isdisjoint(test_g)


def test_collate_rnn_pads_time_axis():
    batch = [
        (torch.randn(4, 3, 8, 8), 1),
        (torch.randn(2, 3, 8, 8), 0),
    ]
    padded_imgs, labels = collate_fn_rnn(batch)

    assert padded_imgs.shape == (2, 4, 3, 8, 8)
    assert labels.tolist() == [1, 0]


def test_select_frame_paths_uniform_not_prefix():
    paths = [f"frame{i}.jpg" for i in range(32)]
    chosen = _select_frame_paths(paths, 8)
    assert chosen[0] == "frame0.jpg"
    assert chosen[-1] == "frame31.jpg"
    assert len(chosen) == 8
    assert chosen != paths[:8]


def test_collate_rnn_handles_empty_batch():
    empty = torch.empty((0, 3, 8, 8))
    padded_imgs, labels = collate_fn_rnn([(empty, 0)])
    assert padded_imgs is None
    assert labels is None
