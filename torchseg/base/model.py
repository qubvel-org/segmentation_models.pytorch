import abc

import torch.nn as nn

from . import initialization as init


class SegmentationModel(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            err = f"""
                Wrong input shape height={h}, width={w}. Expected image height
                and width divisible by {output_stride}. Consider pad your images
                to shape ({new_h}, {new_w}).
            """
            raise RuntimeError(err)

    def forward(self, x):
        """Sequentially pass `x` through model`s encoder, decoder and heads"""
        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks
