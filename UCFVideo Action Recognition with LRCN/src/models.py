"""
Module: models.py

This module defines the LRCN (Long-term Recurrent Convolutional Network) model for video
classification. The LRCN model combines a 2D CNN backbone (e.g., ResNet) for spatial feature
extraction from individual frames with an LSTM to capture temporal dynamics across frames.
An additional fully-connected layer is used to output the final class predictions.

Classes:
    Identity: A helper module that returns the input unchanged. It is used to replace the fully-connected
              layer of the ResNet backbone.
    LRCN: The main model that integrates the CNN backbone, an LSTM, dropout regularization, and a final
          fully-connected layer to produce class logits.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models

class Identity(nn.Module):
    """
    A placeholder identity operator that is argument-insensitive.

    This module is used to replace the fully-connected (fc) layer in the ResNet backbone,
    effectively making the backbone output the raw features before classification.

    Example:
        >>> identity = Identity()
        >>> output = identity(input_tensor)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass that returns the input as is.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: The same tensor x.
        """
        return x

class LRCN(nn.Module):
    """
    LRCN (Long-term Recurrent Convolutional Network) for video classification.

    This model uses a ResNet backbone as a 2D CNN to extract spatial features from each video frame.
    An LSTM network is then used to model the temporal dynamics across the sequence of frame features.
    Dropout is applied before the final fully-connected layer that produces class logits.

    Args:
        hidden_size (int): Number of features in the hidden state of the LSTM.
        n_layers (int): Number of recurrent layers in the LSTM.
        dropout_rate (float): Dropout rate applied before the final classification layer.
        n_classes (int): Number of output classes.
        pretrained (bool, optional): If True, uses a ResNet model pretrained on ImageNet. Default is True.
        cnn_model (str, optional): Specifies the ResNet variant to use as the backbone.
                                   Options: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.
                                   Default is 'resnet34'.

    Raises:
        ValueError: If the specified cnn_model is not supported.
    """
    def __init__(self, hidden_size, n_layers, dropout_rate, n_classes, pretrained=True, cnn_model='resnet34'):
        super().__init__()

        # Set up the ResNet backbone as a 2D CNN feature extractor.
        if cnn_model == 'resnet18':
            base_cnn = models.resnet18(pretrained=pretrained)
        elif cnn_model == 'resnet34':
            base_cnn = models.resnet34(pretrained=pretrained)
        elif cnn_model == 'resnet50':
            base_cnn = models.resnet50(pretrained=pretrained)
        elif cnn_model == 'resnet101':
            base_cnn = models.resnet101(pretrained=pretrained)
        elif cnn_model == 'resnet152':
            # Note: This example uses resnet34 for resnet152 option as a placeholder.
            base_cnn = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError('The input CNN backbone is not supported, please choose a valid ResNet variant.')

        # Retrieve the number of features output by the CNN's original fully-connected layer.
        num_features = base_cnn.fc.in_features

        # Replace the original fc layer with an identity mapping so that raw features are returned.
        base_cnn.fc = Identity()
        self.base_model = base_cnn

        # batch_first=True so input is (batch, time, features), matching the frame tensor.
        # Bidirectional: each frame sees past and future context; hidden state is 2 * hidden_size.
        self.rnn = nn.LSTM(
            num_features,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if n_layers > 1 else 0.0,
        )

        rnn_dim = hidden_size * 2
        # One score per timestep; softmax over time becomes the attention weights.
        self.attn = nn.Linear(rnn_dim, 1)
        self.feat_dropout = nn.Dropout(dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_dim, n_classes)

    def forward(self, x):
        """
        Forward pass for the LRCN model.

        Input x has shape (batch, time, channels, height, width).

        All frames are encoded by the CNN, then stacked as (batch, time, features)
        and fed to a bidirectional LSTM. Temporal attention pools all valid
        timesteps (padding is masked) into one video vector for classification.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time_steps, channels, height, width).

        Returns:
            Tensor: Output logits for each sample in the batch with shape (batch_size, n_classes).
        """
        batch_size, time_steps, channels, height, width = x.shape

        # CNN expects (N, C, H, W); encode every frame, then restore the time axis.
        frame_features = self.base_model(x.view(batch_size * time_steps, channels, height, width))
        frame_features = self.feat_dropout(frame_features.view(batch_size, time_steps, -1))

        # pad_sequence pads with zeros; those padded frames are not real timesteps.
        lengths = (x.view(batch_size, time_steps, -1).abs().sum(dim=-1) > 0).sum(dim=1)
        lengths = lengths.clamp(min=1)

        packed = pack_padded_sequence(
            frame_features,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=time_steps)

        # Mask padded steps so they get zero attention weight.
        time_idx = torch.arange(time_steps, device=x.device).unsqueeze(0)
        mask = time_idx < lengths.to(x.device).unsqueeze(1)

        attn_scores = self.attn(rnn_out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        out = torch.sum(rnn_out * attn_weights, dim=1)

        out = self.dropout(out)
        return self.fc(out)
