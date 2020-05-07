dependencies = [
    'torch', 'torchvision', 'pretrainedmodels',
    'efficientnet-pytorch', 'timm'
]


from .unet import Unet
from .linknet import Linknet
from .fpn import FPN
from .pspnet import PSPNet
from .deeplabv3 import DeepLabV3, DeepLabV3Plus
from .pan import PAN
