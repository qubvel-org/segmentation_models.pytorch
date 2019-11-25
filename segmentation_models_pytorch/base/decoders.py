from ..unet.decoder import UnetDecoder
from ..fpn.decoder import FPNDecoder
from ..pspnet.decoder import PSPDecoder
from ..linknet.decoder import LinknetDecoder

decoder_name_2_object = {'unet': UnetDecoder, 'fpn': FPNDecoder, 'pspnet': PSPDecoder, 'linknet': LinknetDecoder}


def get_decoder(arch, decoder_params):
    decoder = decoder_name_2_object[arch](**decoder_params)
    return decoder
