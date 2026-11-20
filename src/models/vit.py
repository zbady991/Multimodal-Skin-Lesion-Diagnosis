
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTClassifier(nn.Module):
    """
    Vision Transformer B/16 for HAM10000 skin lesion classification.

    Input:
        Images: [B, 3, 224, 224]

    Output:
        Logits: [B, num_classes]
    """

    def __init__(
        self,
        num_classes=7,
        pretrained=True,
        dropout=0.1
    ):
        super().__init__()

        # Load pretrained ViT-B/16 or initialize from scratch
        weights = (
            ViT_B_16_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        self.model = vit_b_16(weights=weights)

        # Replace the original ImageNet classification head
        in_features = self.model.heads.head.in_features

        self.model.heads.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

        self.num_classes = num_classes
        self.pretrained = pretrained

    def forward(self, x):
        return self.model(x)


def build_vit(
    num_classes=7,
    pretrained=True,
    dropout=0.1
):
    """
    Factory function for building the ViT classifier.
    """

    return ViTClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
