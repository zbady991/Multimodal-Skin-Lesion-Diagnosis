
import torch
from src.models.multimodal import MultimodalClassifier


def main():

    print("Starting Multimodal Model Test...")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using Device:", device)

    checkpoint_path = "outputs/resnet50/best_resnet50.pth"

    model = MultimodalClassifier(
        num_classes=7,
        tabular_features=19,
        pretrained=False,
        resnet_checkpoint=checkpoint_path,
        freeze_image=False
    )

    model = model.to(device)

    images = torch.randn(2, 3, 224, 224).to(device)
    tabular = torch.randn(2, 19).to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(images, tabular)

    print("\nModel Test Results")
    print("------------------")
    print("Output Shape:", outputs.shape)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Total Parameters:", total_params)
    print("Trainable Parameters:", trainable_params)

    print("\nMultimodal Model Test Successful!")


if __name__ == "__main__":
    main()
