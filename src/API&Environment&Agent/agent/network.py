import torch
import torch.nn as nn
from domain.configs import NUM_CHANNELS, NUM_BLOCKS


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=8, num_channels=out_channels),
        nn.LeakyReLU(0.1)
    )


class ResidualBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.block1 = conv_block(channels, channels)
        self.block2 = conv_block(channels, channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.block1(x)
        x = self.block2(x)

        x = x + residual
        x = self.act(x)

        return x


class Network(nn.Module):

    def __init__(self, is_dueling_net: bool,
                 is_residual_net: bool,
                 num_planes: int,
                 rows: int,
                 columns: int,
                 num_actions: int) -> None:

        super().__init__()

        self.is_dueling_net = is_dueling_net
        self.is_residual_net = is_residual_net

        if self.is_residual_net:
            self.conv = conv_block(num_planes, NUM_CHANNELS)

            self.res_blocks = nn.Sequential(
                *[ResidualBlock(NUM_CHANNELS) for _ in range(NUM_BLOCKS)]
            )

        else:
            self.res_blocks = None

            self.conv = nn.Sequential(
                conv_block(num_planes, NUM_CHANNELS),
                conv_block(NUM_CHANNELS, NUM_CHANNELS),
                conv_block(NUM_CHANNELS, NUM_CHANNELS)
            )

        flat_size = NUM_CHANNELS * rows * columns

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 1024),
            nn.LeakyReLU(0.1)
        )

        if self.is_dueling_net:
            self.value = nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.1),
                nn.Linear(512, 1)
            )

            self.advantage = nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.1),
                nn.Linear(512, num_actions)
            )

            self.head = None

        else:
            self.head = nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.1),
                nn.Linear(512, num_actions)
            )

            self.value = None

            self.advantage = None

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.conv(x)

        if self.is_residual_net:
            x = self.res_blocks(x)

        x = self.shared(x)

        if self.is_dueling_net:
            if legal_mask is None:
                raise ValueError('Mask cannot be None.')

            value = self.value(x)
            advantage = self.advantage(x)

            legal_mask = legal_mask.bool()

            masked_advantage = advantage.masked_fill(~legal_mask, 0.0)
            legal_count = legal_mask.sum(dim=1, keepdim=True).clamp_min(1)
            advantage_mean = masked_advantage.sum(dim=1, keepdim=True) / legal_count

            return value + advantage - advantage_mean
        else:
            return self.head(x)
