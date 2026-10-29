"""Tests for the LRCN forward path (BiLSTM + temporal attention)."""

import pytest

torch = pytest.importorskip("torch")

from src.models import LRCN


def test_lrcn_forward_shape():
    model = LRCN(
        hidden_size=16,
        n_layers=1,
        dropout_rate=0.1,
        n_classes=5,
        pretrained=False,
        cnn_model="resnet18",
    )
    x_batch = torch.randn(2, 3, 3, 64, 64)
    out = model(x_batch)
    assert out.shape == (2, 5)


def test_lrcn_forward_with_padding():
    model = LRCN(
        hidden_size=8,
        n_layers=1,
        dropout_rate=0.1,
        n_classes=4,
        pretrained=False,
        cnn_model="resnet18",
    )
    x_batch = torch.zeros(2, 4, 3, 64, 64)
    x_batch[0, :4] = torch.randn(4, 3, 64, 64)
    x_batch[1, :2] = torch.randn(2, 3, 64, 64)
    out = model(x_batch)
    assert out.shape == (2, 4)
