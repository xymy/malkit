import torch
import torch.nn as nn


class MalConv(nn.Module):
    """The MalConv model.

    References:
        - Edward Raff et al. 2018. Malware Detection by Eating a Whole EXE.
          https://www.aaai.org/ocs/index.php/WS/AAAIW18/paper/view/16422
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()

        # 257 = byte (0-255) + padding (256)
        self.embedding = nn.Embedding(257, 8)

        self.conv1 = nn.Sequential(
            nn.Conv1d(8, 128, kernel_size=500, stride=500, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(8, 128, kernel_size=500, stride=500, bias=True),
            nn.Sigmoid(),
        )

        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)

        # Treat embedding dimension as channel.
        x = x.permute(0, 2, 1)

        # Perform gated convolution.
        x = self.conv1(x) * self.conv2(x)

        x = self.maxpool(x)
        x = self.fc(x)
        return x
