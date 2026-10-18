"""Arquitectura base, compatible con distintas resoluciones de entrada."""

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """Tres bloques convolucionales y un clasificador global de nueve clases."""

    def __init__(self, num_classes: int = 9, batch_norm: bool = True) -> None:
        super().__init__()
        self.batch_norm = batch_norm
        self.features = nn.Sequential(
            self._block(3, 32, batch_norm),
            self._block(32, 64, batch_norm),
            self._block(64, 128, batch_norm),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    @staticmethod
    def _block(in_channels: int, out_channels: int, batch_norm: bool) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(self.pool(features).flatten(1))
