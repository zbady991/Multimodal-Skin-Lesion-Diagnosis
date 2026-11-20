
import torch
import torch.nn as nn
from torchvision import models


class MultimodalClassifier(nn.Module):

    def __init__(
        self,
        num_classes=7,
        tabular_features=19,
        pretrained=False,
        resnet_checkpoint=None,
        freeze_image=False
    ):
        super().__init__()

        # ==========================================
        # 1. IMAGE BRANCH: ResNet50
        # ==========================================

        weights = (
            models.ResNet50_Weights.DEFAULT
            if pretrained else None
        )

        self.image_branch = models.resnet50(weights=weights)

        # ==========================================
        # 2. LOAD TRAINED RESNET50 CHECKPOINT
        # ==========================================

        if resnet_checkpoint is not None:

            checkpoint = torch.load(
                resnet_checkpoint,
                map_location="cpu",
                weights_only=True
            )

            state_dict = checkpoint["model_state_dict"]

            # Remove "model." prefix
            # Exclude the original 7-class classification head
            image_state_dict = {
                key.replace("model.", "", 1): value
                for key, value in state_dict.items()
                if key.startswith("model.")
                and not key.startswith("model.fc.")
            }

            # Load ResNet50 feature extractor weights
            load_result = self.image_branch.load_state_dict(
                image_state_dict,
                strict=False
            )

            print("\nCheckpoint Loading Results:")
            print("-----------------------------------")
            print("Missing Keys:", load_result.missing_keys)
            print("Unexpected Keys:", load_result.unexpected_keys)

            # Verify that only the original FC layer is missing
            expected_missing = {
                "fc.weight",
                "fc.bias"
            }

            if set(load_result.missing_keys) != expected_missing:
                raise RuntimeError(
                    "Unexpected missing keys in ResNet50 checkpoint!"
                )

            if load_result.unexpected_keys:
                raise RuntimeError(
                    "Unexpected keys found in ResNet50 checkpoint!"
                )

            print("ResNet50 Checkpoint Loaded Successfully!")

        # ==========================================
        # 3. FREEZE IMAGE BRANCH (OPTIONAL)
        # ==========================================

        if freeze_image:

            for param in self.image_branch.parameters():
                param.requires_grad = False

        # ==========================================
        # 4. REMOVE ORIGINAL CLASSIFICATION HEAD
        # ==========================================

        image_features = self.image_branch.fc.in_features

        self.image_branch.fc = nn.Identity()

        # ==========================================
        # 5. TABULAR BRANCH: MLP
        # ==========================================

        self.tabular_branch = nn.Sequential(

            nn.Linear(tabular_features, 128),

            nn.ReLU(),

            nn.BatchNorm1d(128),

            nn.Dropout(0.3),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.2)

        )

        # ==========================================
        # 6. FEATURE FUSION
        # ==========================================

        fusion_features = image_features + 64

        # ==========================================
        # 7. CLASSIFICATION HEAD
        # ==========================================

        self.classifier = nn.Sequential(

            nn.Linear(fusion_features, 256),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(256, num_classes)

        )

    # ==========================================
    # 8. FORWARD PASS
    # ==========================================

    def forward(self, images, tabular):

        # Extract image features
        image_features = self.image_branch(images)

        # Extract tabular features
        tabular_features = self.tabular_branch(tabular)

        # Concatenate image and tabular features
        combined_features = torch.cat(
            (image_features, tabular_features),
            dim=1
        )

        # Final classification
        output = self.classifier(combined_features)

        return output
