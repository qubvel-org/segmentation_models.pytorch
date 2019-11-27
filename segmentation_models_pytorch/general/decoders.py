from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.pspnet.decoder import PSPDecoder
from segmentation_models_pytorch.linknet.decoder import LinknetDecoder

decoder_name_2_object = {'unet': UnetDecoder, 'fpn': FPNDecoder, 'pspnet': PSPDecoder, 'linknet': LinknetDecoder}


def get_decoder(arch, **kwargs):
    decoder = decoder_name_2_object[arch](**kwargs)
    return decoder
