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
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        flat_size = 64 * rows * columns

        self.value_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)

        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        return values + (advantages - advantages.mean(dim=1, keepdim=True))
