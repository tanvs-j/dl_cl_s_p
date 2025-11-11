from __future__ import annotations
import torch
import torch.nn as nn

class EEGNet1D(nn.Module):
    def __init__(self, in_channels: int = 19, num_classes: int = 2, base: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, base, kernel_size=7, padding=3),
            nn.BatchNorm1d(base), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(base, base*2, kernel_size=7, padding=3),
            nn.BatchNorm1d(base*2), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(base*2, base*4, kernel_size=5, padding=2),
            nn.BatchNorm1d(base*4), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(base*4, base*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base*4), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base*4, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T)
        z = self.features(x)
        return self.head(z)
