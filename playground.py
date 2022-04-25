import torch

from src.models.standard.tali import TALIMP, default_cfgs

model = TALIMP(**default_cfgs["modus_prime_tali_viat_pretrained"])
x_image = torch.randn(2, 3, 288, 176)
x_text = torch.randn(2, 77)

out = model.forward(x_image, x_text)
