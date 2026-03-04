"""
AgroVision models: baseline CNN and ResNet18 transfer learning.
"""
from .baseline_model import BaselineCNN
from .transfer_model import ResNet18Transfer
import torch.nn as nn


def get_model(
    model_name: str,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    """
    Factory: returns a model by name.

    Args:
        model_name: "baseline" or "resnet18"
        num_classes: Number of output classes (e.g. disease categories).
        **kwargs: Passed to the model constructor (e.g. freeze_backbone for resnet18).

    Returns:
        Model instance (GPU-compatible; move with .to(device) when training).

    Raises:
        ValueError: If model_name is not supported.
    """
    name = model_name.strip().lower()
    if name == "baseline":
        return BaselineCNN(num_classes=num_classes, **kwargs)
    if name == "resnet18":
        return ResNet18Transfer(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model: {model_name}. Use 'baseline' or 'resnet18'.")


__all__ = ["BaselineCNN", "ResNet18Transfer", "get_model"]
