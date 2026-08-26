import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self, num_planes: int, rows: int, columns: int,
                 num_actions: int, num_groups: int=8) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(num_planes, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 128),
            nn.ReLU(),
        )

        flat_size = 128 * rows * columns

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        last_layer = self.head[-1]
        nn.init.uniform_(last_layer.weight, -3e-3, 3e-3)
        nn.init.zeros_(last_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.head(x)
