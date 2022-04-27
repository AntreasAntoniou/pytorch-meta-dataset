import torch
import torch.nn as nn
from dotted_dict import DottedDict

# TODO add short tutorial on how to add a new model in this repo
from gate.models.clip import CLIP
from gate.models.tali import TALIModusPrime


def _cfg(model_name_to_download, model_root_dir, pretrained, **kwargs):
    return {
        "model_name_to_download": model_name_to_download,
        "model_root_dir": model_root_dir,
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "input_shape_dict": DottedDict(
            {
                "image": DottedDict(
                    {"shape": DottedDict(channels=3, width=288, length=176)}
                ),
                "text": DottedDict({"shape": DottedDict(sequence_length=77)}),
            }
        ),
        "pretrained": pretrained,
        "num_classes": 1000,
        **kwargs,
    }


default_cfgs = {
    "clip_vit_b_16_pretrained": _cfg(
        input_shape_dict=DottedDict(
            image=DottedDict(shape=DottedDict(channels=3, width=224, length=224)),
        ),
        model_name_to_download="ViT-B/16",
        model_root_dir="clip_models/",
        pretrained=True,
    ),
    "clip_vit_b_16_scratch": _cfg(
        input_shape_dict=DottedDict(
            image=DottedDict(shape=DottedDict(channels=3, width=224, length=224)),
        ),
        model_name_to_download="ViT-B/16",
        model_root_dir="clip_models/",
        pretrained=False,
    ),
}


class CLIPModel(nn.Module):
    def __init__(
        self,
        input_shape_dict: DottedDict,
        model_root_dir: str = "clip_models",
        model_name_to_download: str = "ViT-B/16",
        pretrained: bool = True,
        num_classes: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.linear_layer_dict = None
        self.model = CLIP(
            input_shape_dict=input_shape_dict,
            model_root_dir=model_root_dir,
            model_name_to_download=model_name_to_download,
            pretrained=pretrained,
        )

        self.num_classes = num_classes
        self.build()

    def build(self):
        self.model.build(
            batch_dict=DottedDict(
                image=torch.randn(
                    2,
                    3,
                    self.model.input_shape_dict.image.shape.width,
                    self.model.input_shape_dict.image.shape.length,
                )
            )
        )

        self.linear_layer_dict = nn.ModuleDict(
            dict(
                image=nn.Linear(768, self.num_classes, bias=True),
            )
        )
        self.model.model.visual.proj = None
        self.model.model.token_embedding = nn.Identity()
        self.model.model.transformer = nn.Identity()
        self.ln_final = nn.Identity()

    def forward_features(self, x_image):
        return self.model.forward_image(x_image)

    def forward(self, x_image, feature=False):

        if feature:
            return self.forward_features(x_image)

        out_image_features = self.model.forward_image(x_image)

        return self.linear_layer_dict["image"](out_image_features)


def clip_vit_b_16_pretrained(
    num_classes: int = 1000,
    **kwargs,
):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    config = default_cfgs["ViT-B/16-pretrained"]
    config["num_classes"] = num_classes
    return CLIPModel(**config)


def clip_vit_b_16_scratch(
    num_classes: int = 1000,
    **kwargs,
):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    config = default_cfgs["ViT-B/16-scratch"]
    config["num_classes"] = num_classes
    return CLIPModel(**config)
