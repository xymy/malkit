from typing import Optional

import torch
import torch.nn as nn


class MalConv(nn.Module):
    """The MalConv model.

    References:
        - Edward Raff et al. 2018. Malware Detection by Eating a Whole EXE.
          https://arxiv.org/abs/1710.09435
    """

    def __init__(
        self,
        num_classes: int = 2,
        *,
        num_embeddings: int = 257,
        embedding_dim: int = 8,
        channels: int = 128,
        kernel_size: int = 512,
        stride: int = 512,
        padding_idx: Optional[int] = 256,
    ) -> None:
        super().__init__()

        # By default, num_embeddings (257) = byte (0-255) + padding (256).
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

        self.conv1 = nn.Conv1d(embedding_dim, channels, kernel_size=kernel_size, stride=stride, bias=True)
        self.conv2 = nn.Conv1d(embedding_dim, channels, kernel_size=kernel_size, stride=stride, bias=True)

        self.maxpool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)

        # Treat embedding dimension as channel.
        x = x.permute(0, 2, 1)

        # Perform gated convolution.
        x = self.conv1(x) * torch.sigmoid(self.conv2(x))

        x = self.maxpool(x)

        x = self.fc(x)
        return x
