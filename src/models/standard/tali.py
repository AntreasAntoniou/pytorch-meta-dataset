import torch
import torch.nn as nn
from dotted_dict import DottedDict

# TODO add short tutorial on how to add a new model in this repo
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
        num_classes: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.linear_layer_dict = None
        self.model = TALIModusPrime(
            input_shape_dict, model_root_dir, model_name_to_download, pretrained
        )
        self.model.build()
        self.num_classes = num_classes
        self.build()

    def build(self):
        self.linear_layer_dict = nn.ModuleDict(
            dict(
                image=nn.Linear(768, self.num_classes, bias=True),
            )
        )
        self.model.system.modality_embeddings["text"] = nn.Identity()
        self.model.system.modality_embeddings["video"] = nn.Identity()
        self.model.system.modality_embeddings["audio"] = nn.Identity()

    def forward_features(self, x_image):
        out_image_features = self.model.forward_image(x_image)
        return out_image_features

    def forward(self, x_image, feature=False):

        if feature:
            return self.forward_features(x_image)

        out_image_features = self.model.forward_image(x_image)

        out_image = self.linear_layer_dict["image"](out_image_features)

        return out_image


def modus_prime_tali_viat_pretrained(
    input_shape_dict: DottedDict = DottedDict(
        {
            "image": DottedDict(
                {"shape": DottedDict(channels=3, width=288, length=176)}
            ),
            "text": DottedDict({"shape": DottedDict(sequence_length=77)}),
        }
    ),
    model_root_dir: str = "tali_models/",
    model_name_to_download: str = None,
    pretrained: bool = True,
    num_classes: int = 1000,
):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """

    return TALIMP(
        input_shape_dict=input_shape_dict,
        model_root_dir=model_root_dir,
        model_name_to_download=model_name_to_download,
        pretrained=True,
        num_classes=num_classes,
    )


def modus_prime_tali_viat_scratch(
    input_shape_dict: DottedDict = DottedDict(
        {
            "image": DottedDict(
                {"shape": DottedDict(channels=3, width=288, length=176)}
            ),
            "text": DottedDict({"shape": DottedDict(sequence_length=77)}),
        }
    ),
    model_root_dir: str = "tali_models/",
    model_name_to_download: str = None,
    pretrained: bool = True,
    num_classes: int = 1000,
):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    return TALIMP(
        input_shape_dict=input_shape_dict,
        model_root_dir=model_root_dir,
        model_name_to_download=model_name_to_download,
        pretrained=False,
        num_classes=num_classes,
    )
