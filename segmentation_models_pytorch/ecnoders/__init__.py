import torch.utils.model_zoo as model_zoo

from .resnet import resnet_encoders
from .dpn import dpn_encoders
from .vgg import vgg_encoders
from .senet import senet_encoders
from .densenet import densenet_encoders
from .inceptionresnetv2 import inception_encoders

encoders = {}
encoders.update(resnet_encoders)
encoders.update(dpn_encoders)
encoders.update(vgg_encoders)
encoders.update(senet_encoders)
encoders.update(densenet_encoders)
encoders.update(inception_encoders)


def get_encoder(name, pretrained=True):

    Encoder = encoders[name]['encoder']
    encoder = Encoder(**encoders[name]['params'])
    encoder.out_shapes = encoders[name]['out_shapes']

    if pretrained:
        encoder.load_state_dict(model_zoo.load_url(encoders[name]['url']))
    return encoder
