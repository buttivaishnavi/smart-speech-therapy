import torch

checkpoint = torch.load("model.pct", map_location="cpu")

model = checkpoint.get("model") or checkpoint

# Save in HF-compatible way
torch.save(model.state_dict(), "pytorch_model.bin")

print("Model converted")
