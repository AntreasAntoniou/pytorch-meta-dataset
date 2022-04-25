from .resnet import *
from .resnet_v2 import *
from .vit import *
from .tali import *
from .clip import *

from .resnet_v2 import default_cfgs as resnet_v2_config
from .resnet import default_cfgs as resnet_config
from .vit import default_cfgs as vit_config
from .tali import default_cfgs as tali_config
from .clip import default_cfgs as clip_config

model_configs = resnet_v2_config
model_configs.update(resnet_config)
model_configs.update(vit_config)
model_configs.update(tali_config)
model_configs.update(clip_config)
