import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self, num_planes: int, rows: int, columns: int, num_actions: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(num_planes, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        flat_size = 128 * rows * columns

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.head(x)
