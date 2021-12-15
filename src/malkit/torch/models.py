from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


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
        padding_idx: Optional[int] = 256,
        window_size: int = 500,
    ) -> None:
        super().__init__()

        # By default, num_embeddings (257) = byte (0-255) + padding (256).
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=window_size, stride=window_size)
        self.conv2 = nn.Conv1d(embedding_dim, 128, kernel_size=window_size, stride=window_size)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.forward_embedding(x)
        return x

    def forward_embedding(self, x: Tensor) -> Tensor:
        # Treat embedding dimension as channel.
        x = x.permute(0, 2, 1)

        # Perform gated convolution.
        x = self.conv1(x) * torch.sigmoid(self.conv2(x))
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.fc(torch.flatten(x, 1))

        return x
