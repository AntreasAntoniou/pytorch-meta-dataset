import torch
from rich import print

from src.models.standard.tali import TALIMP, default_cfgs

model = TALIMP(**default_cfgs["modus_prime_tali_viat_pretrained"])
x_image = torch.randn(2, 3, 288, 176)
x_text = torch.randint(high=100, size=(2, 77))

out_image = model.forward(x_image)

print(f"Test with {x_image.shape} and {x_text.shape} -> {out_image.shape}")
