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


def get_encoder(name, encoder_weights=None):

    Encoder = encoders[name]['encoder']
    encoder = Encoder(**encoders[name]['params'])
    encoder.out_shapes = encoders[name]['out_shapes']

    if encoder_weights is not None:
        settings = encoders[name]['pretrained_settings'][encoder_weights]

        encoder.load_state_dict(model_zoo.load_url(settings['url']))
        encoder.input_space = settings['input_space']
        encoder.input_size = settings['input_size']
        encoder.input_range = settings['input_range']
        encoder.mean = settings['mean']
        encoder.std = settings['std']

    return encoder

def get_encoder_names():
    return list(encoders.keys())
