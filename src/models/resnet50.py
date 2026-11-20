
import torch
import torch.nn as nn
from torchvision import models


class ResNet50Classifier(nn.Module):

    def __init__(self, num_classes=7, pretrained=True):

        super().__init__()

        # Load ResNet50
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = None

        self.model = models.resnet50(weights=weights)

        # Get the number of input features
        num_features = self.model.fc.in_features

        # Replace the final classification layer
        self.model.fc = nn.Linear(
            num_features,
            num_classes
        )

    def forward(self, x):

        return self.model(x)


if __name__ == "__main__":

    model = ResNet50Classifier(
        num_classes=7,
        pretrained=True
    )

    dummy_input = torch.randn(2, 3, 224, 224)

    output = model(dummy_input)

    print("Output Shape:", output.shape)
    print("Number of Parameters:",
          sum(p.numel() for p in model.parameters()))

    print("ResNet50 Model Created Successfully!")
