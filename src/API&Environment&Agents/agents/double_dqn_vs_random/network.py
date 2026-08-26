import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self, num_planes: int, rows: int, columns: int,
                 num_actions: int, num_groups: int=8) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(num_planes, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 64),
            nn.ReLU()
        )

        flat_size = 64 * rows * columns

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.head(x)
