
"""Ablation models used for train/validation comparisons.

These classes are separate from the original multimodal and ResNet50
implementations so existing checkpoints are never overwritten.
"""

import torch
import torch.nn as nn
from torchvision import models


def load_resnet_backbone(resnet_checkpoint=None, pretrained=False):

    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    image_branch = models.resnet50(weights=weights)

    if resnet_checkpoint is None:
        image_features = image_branch.fc.in_features
        image_branch.fc = nn.Identity()
        return image_branch, image_features

    checkpoint = torch.load(
        resnet_checkpoint,
        map_location="cpu",
        weights_only=True
    )

    state_dict = checkpoint["model_state_dict"]

    image_state_dict = {
        key.replace("model.", "", 1): value
        for key, value in state_dict.items()
        if key.startswith("model.") and not key.startswith("model.fc.")
    }

    load_result = image_branch.load_state_dict(
        image_state_dict,
        strict=False
    )

    expected_missing = {"fc.weight", "fc.bias"}

    if set(load_result.missing_keys) != expected_missing:
        raise RuntimeError(
            "Unexpected missing keys while loading ResNet50 backbone: "
            f"{load_result.missing_keys}"
        )

    if load_result.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys while loading ResNet50 backbone: "
            f"{load_result.unexpected_keys}"
        )

    image_features = image_branch.fc.in_features
    image_branch.fc = nn.Identity()
    return image_branch, image_features


def make_classifier(in_features, num_classes):

    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )


def make_tabular_branch(tabular_features):

    return nn.Sequential(
        nn.Linear(tabular_features, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2)
    )


class ImageOnlyClassifier(nn.Module):

    def __init__(
        self,
        num_classes=7,
        pretrained=False,
        resnet_checkpoint=None
    ):
        super().__init__()

        self.image_branch, image_features = load_resnet_backbone(
            resnet_checkpoint=resnet_checkpoint,
            pretrained=pretrained
        )
        self.classifier = make_classifier(image_features, num_classes)

    def forward(self, images, tabular=None):

        image_features = self.image_branch(images)
        return self.classifier(image_features)


class TabularOnlyClassifier(nn.Module):

    def __init__(
        self,
        num_classes=7,
        tabular_features=19
    ):
        super().__init__()

        self.tabular_branch = make_tabular_branch(tabular_features)
        self.classifier = make_classifier(64, num_classes)

    def forward(self, tabular):

        tabular_features = self.tabular_branch(tabular)
        return self.classifier(tabular_features)
