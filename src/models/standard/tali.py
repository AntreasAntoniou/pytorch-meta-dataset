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
            image=DottedDict(shape=DottedDict(channels=3, width=288, length=176)),
        ),
        model_name_to_download="model-deep-salad-17",
        project_name="machinelearningbrewery/godzilla-gcp-experiments",
        model_version="v190",
        model_root_dir="tali_models/",
        pretrained=True,
    ),
    "modus_prime_tali_viat_pretrained_v187": _cfg(
        input_shape_dict=DottedDict(
            image=DottedDict(shape=DottedDict(channels=3, width=288, length=176)),
        ),
        model_name_to_download="model-deep-salad-17",
        project_name="machinelearningbrewery/godzilla-gcp-experiments",
        model_version="v187",
        model_root_dir="tali_models/",
        pretrained=True,
    ),
    "modus_prime_tali_viat_pretrained_v188": _cfg(
        input_shape_dict=DottedDict(
            image=DottedDict(shape=DottedDict(channels=3, width=288, length=176)),
        ),
        model_name_to_download="model-deep-salad-17",
        project_name="machinelearningbrewery/godzilla-gcp-experiments",
        model_version="v188",
        model_root_dir="tali_models/",
        pretrained=True,
    ),
    "modus_prime_tali_viat_pretrained_v189": _cfg(
        input_shape_dict=DottedDict(
            image=DottedDict(shape=DottedDict(channels=3, width=288, length=176)),
        ),
        model_name_to_download="model-deep-salad-17",
        project_name="machinelearningbrewery/godzilla-gcp-experiments",
        model_version="v189",
        model_root_dir="tali_models/",
        pretrained=True,
    ),
    "modus_prime_tali_viat_scratch": _cfg(
        input_shape_dict=DottedDict(
            image=DottedDict(shape=DottedDict(channels=3, width=288, length=176)),
        ),
        model_name_to_download="model-deep-salad-17",
        project_name="machinelearningbrewery/godzilla-gcp-experiments",
        model_version="v190",
        model_root_dir="tali_models/",
        pretrained=False,
    ),
}


class TALIMP(nn.Module):
    def __init__(
        self,
        input_shape_dict: DottedDict,
        model_root_dir: str = "tali_models",
        model_name_to_download: str = "model-deep-salad-17",
        project_name: str = "machinelearningbrewery/godzilla-gcp-experiments",
        model_version: str = "v187",
        pretrained: bool = True,
        num_classes: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.linear_layer_dict = None
        self.model = TALIModusPrime(
            input_shape_dict=input_shape_dict,
            model_root_dir=model_root_dir,
            model_name_to_download=model_name_to_download,
            model_version=model_version,
            project_name=project_name,
            pretrained=pretrained,
        )
        self.model.build()
        self.num_classes = num_classes
        self.build()

    def build(self):
        self.linear_layer_dict = nn.ModuleDict(
            dict(image=nn.Linear(768, self.num_classes, bias=True),)
        )
        self.model.model.system.modality_embeddings["text"] = nn.Identity()
        self.model.model.system.modality_embeddings["video"] = nn.Identity()
        self.model.model.system.modality_embeddings["audio"] = nn.Identity()

    def forward_features(self, x_image):
        return self.model.forward_image(x_image)

    def forward(self, x_image, feature=False):

        if feature:
            return self.forward_features(x_image)

        out_image_features = self.model.forward_image(x_image)

        return self.linear_layer_dict["image"](out_image_features)


def modus_prime_tali_viat_pretrained(
    num_classes: int = 1000, **kwargs,
):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    config = default_cfgs["modus_prime_tali_viat_pretrained"]
    config["num_classes"] = num_classes
    return TALIMP(**config)


def modus_prime_tali_viat_pretrained_v187(
    num_classes: int = 1000, **kwargs,
):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    config = default_cfgs["modus_prime_tali_viat_pretrained_v187"]
    config["num_classes"] = num_classes
    return TALIMP(**config)


def modus_prime_tali_viat_pretrained_v188(
    num_classes: int = 1000, **kwargs,
):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    config = default_cfgs["modus_prime_tali_viat_pretrained_v188"]
    config["num_classes"] = num_classes
    return TALIMP(**config)


def modus_prime_tali_viat_pretrained_v189(
    num_classes: int = 1000, **kwargs,
):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    config = default_cfgs["modus_prime_tali_viat_pretrained_v189"]
    config["num_classes"] = num_classes
    return TALIMP(**config)


def modus_prime_tali_viat_scratch(
    num_classes: int = 1000, **kwargs,
):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    config = default_cfgs["modus_prime_tali_viat_scratch"]
    config["num_classes"] = num_classes
    return TALIMP(**config)
