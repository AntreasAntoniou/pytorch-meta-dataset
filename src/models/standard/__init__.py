from .clip import *
from .clip import default_cfgs as clip_config
from .resnet import *
from .resnet import default_cfgs as resnet_config
from .resnet_v2 import *
from .resnet_v2 import default_cfgs as resnet_v2_config
from .tali import *
from .tali import default_cfgs as tali_config
from .vit import *
from .vit import default_cfgs as vit_config

model_configs = resnet_v2_config
model_configs.update(resnet_config)
model_configs.update(vit_config)
model_configs.update(tali_config)
model_configs.update(clip_config)
