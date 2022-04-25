from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from dotted_dict import DottedDict

# TODO add short tutorial on how to add a new model in this repo
from gate.models.tali import TALIModusPrime


def _cfg(model_name_to_download, model_root_dir, pretrained, **kwargs):
    return {
        "model_name_to_download": model_name_to_download,
        "model_root_dir": model_root_dir,
        "input_shape_dict": (3, 224, 224),
        "pretrained": pretrained,
        **kwargs,
    }


default_cfgs = {
    "modus_prime_tali_viat_pretrained": _cfg(
        input_shape_dict=DottedDict(
            {
                "image": DottedDict(
                    {"shape": DottedDict(channels=3, width=288, length=176)}
                ),
                "text": DottedDict({"shape": DottedDict(sequence_length=77)}),
            }
        ),
        model_name_to_download=None,
        model_root_dir="tali/",
        pretrained=True,
    ),
    "modus_prime_tali_viat_scratch": _cfg(
        input_shape_dict=DottedDict(
            {
                "image": DottedDict(
                    {"shape": DottedDict(channels=3, width=288, length=176)}
                ),
                "text": DottedDict({"shape": DottedDict(sequence_length=77)}),
            }
        ),
        model_name_to_download=None,
        model_root_dir="tali/",
        pretrained=False,
    ),
}


class TALIMP(nn.Module):
    def __init__(
        self,
        input_shape_dict: DottedDict,
        model_root_dir: str = None,
        model_name_to_download: str = "resnet18",
        pretrained: bool = True,
    ):
        super().__init__()
        self.model = TALIModusPrime(
            input_shape_dict, model_root_dir, model_name_to_download, pretrained
        )
        self.model.build()

    def forward(self, x_image, x_text=None):
        out_image = self.model.forward_image(x_image)

        if x_text is None:
            return out_image

        out_text = self.model.forward_text(x_text)
        return out_image, out_text


model = TALIMP(**default_cfgs["modus_prime_tali_viat_pretrained"])
x_image = torch.randn(2, 3, 288, 176)
x_text = torch.randn(2, 77)

out = model.forward(x_image, x_text)
