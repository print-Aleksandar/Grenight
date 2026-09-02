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

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 1024),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
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
