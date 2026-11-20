
import torch

checkpoint_path = "outputs/resnet50/best_resnet50.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)

print("Checkpoint Type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nCheckpoint Keys:")
    for key in checkpoint.keys():
        print("-", key)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    print("\nFirst 10 Model Parameters:")
    for key in list(state_dict.keys())[:10]:
        print(key)
