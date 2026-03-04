"""
Baseline CNN for AgroVision crop disease classification.
Lightweight custom architecture: 4 conv blocks → AdaptiveAvgPool2d → Dropout → FC.
Uses Kaiming initialization; suitable for 224×224 ImageNet-normalized input.
"""
import torch
import torch.nn as nn
from typing import Optional


def _conv_block(in_ch: int, out_ch: int, kernel_size: int = 3) -> nn.Sequential:
    """Single block: Conv2d → BatchNorm → ReLU → MaxPool2d."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BaselineCNN(nn.Module):
    """
    Custom CNN: 4 conv blocks (32→64→128→256 channels), then global pool, dropout, classifier.
    Input: (N, 3, 224, 224). Output: (N, num_classes).
    """

    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.num_classes = num_classes

        # 224 → 112 → 56 → 28 → 14 (after 4× MaxPool2d(2))
        self.features = nn.Sequential(
            _conv_block(3, 32),    # 224→112
            _conv_block(32, 64),   # 112→56
            _conv_block(64, 128),  # 56→28
            _conv_block(128, 256), # 28→14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(256, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming initialization for conv and linear layers; BatchNorm uses default."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)
