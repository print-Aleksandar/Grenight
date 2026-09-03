import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual
        x = self.act(x)

        return x


class Network(nn.Module):

    def __init__(self, num_planes: int, rows: int, columns: int, num_actions: int,
                 num_blocks: int=4, channels: int=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(num_planes, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True)
        )

        flat_size = channels * rows * columns

        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 1024),
            nn.SiLU(inplace=True)
        )

        self.value = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SiLU(inplace=True),
            nn.Linear(512, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SiLU(inplace=True),
            nn.Linear(512, num_actions)
        )

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        x = self.res_blocks(x)

        x = self.shared(x)

        value = self.value(x)
        advantage = self.advantage(x)

        legal_mask = legal_mask.bool()

        masked_advantage = advantage.masked_fill(~legal_mask, 0.0)

        legal_count = legal_mask.sum(dim=1, keepdim=True).clamp_min(1)

        advantage_mean = masked_advantage.sum(dim=1, keepdim=True) / legal_count

        q = value + advantage - advantage_mean

        q = q.masked_fill(~legal_mask, -float("inf"))

        return q
