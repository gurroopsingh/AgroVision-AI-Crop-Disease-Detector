"""
Transfer learning model for AgroVision: ResNet18 backbone + custom classifier.
Uses ImageNet-pretrained weights; backbone can be frozen for fast training.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Transfer(nn.Module):
    """
    ResNet18 with ImageNet pretrained backbone; final FC replaced for num_classes.
    Optionally freeze backbone (all params except fc). Can unfreeze layer4 for fine-tuning.
    """

    def __init__(self, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self._freeze_backbone = freeze_backbone

        # Pretrained backbone (ImageNet1K_V1)
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)

        self.backbone = backbone

        if freeze_backbone:
            self._freeze_backbone_params()

    def _freeze_backbone_params(self) -> None:
        """Freeze all parameters except the final FC layer."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_layer4(self) -> None:
        """
        Unfreeze only the last residual block (layer4) for gradual fine-tuning.

        This method keeps earlier backbone layers frozen and leaves the final
        classifier trainable.
        """
        for name, param in self.backbone.named_parameters():
            param.requires_grad = ("layer4" in name) or ("fc" in name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
