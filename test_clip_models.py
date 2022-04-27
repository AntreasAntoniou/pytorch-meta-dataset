import torch
from src.models.standard.clip import CLIPModel, default_cfgs
from rich import print

model = CLIPModel(**default_cfgs["ViT-B/16-pretrained"])
x_image = torch.randn(2, 3, 288, 176)
x_text = torch.randint(high=100, size=(2, 77))

out_image = model.forward(x_image)

print(f"Test with {x_image.shape} and {x_text.shape} -> {out_image.shape}")
