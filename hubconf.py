dependencies = [
    'torch', 'torchvision', 'pretrainedmodels',
    'efficientnet-pytorch', 'timm'
]


from segmentation_models_pytorch.unet import Unet
from segmentation_models_pytorch.linknet import Linknet
from segmentation_models_pytorch.fpn import FPN
from segmentation_models_pytorch.pspnet import PSPNet
from segmentation_models_pytorch.deeplabv3 import DeepLabV3, DeepLabV3Plus
from segmentation_models_pytorch.pan import PAN
