from typing import Optional, Union, List
from ..base import SegmentationModel, SegmentationHead, ClassificationHead
from ..encoders import get_encoder
from .decoders import get_decoder


# class EncoderDecoder(SegmentationModel):
#     """Unet_ is a fully convolution neural network for image semantic segmentation
#
#     Args:
#         encoder_name: name of classification model (without last dense layers) used as feature
#             extractor to build segmentation model.
#         encoder_depth (int): number of stages used in decoder, larger depth - more features are generated.
#             e.g. for depth=3 encoder will generate list of features with following spatial shapes
#             [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature tensor will have
#             spatial resolution (H/(2^depth), W/(2^depth)]
#         encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
#         decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
#         decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
#             is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
#             One of [True, False, 'inplace']
#         decoder_attention_type: attention module used in decoder of the model
#             One of [``None``, ``scse``]
#         in_channels: number of input channels for model, default is 3.
#         num_classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
#         activation: activation function to apply after final convolution;
#             One of [``sigmoid``, ``softmax``, ``logsoftmax``, ``identity``, callable, None]
#         aux_params: if specified model will have additional classification auxiliary output
#             build on top of encoder, supported params:
#                 - classes (int): number of classes
#                 - pooling (str): one of 'max', 'avg'. Default is 'avg'.
#                 - dropout (float): dropout factor in [0, 1)
#                 - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)
#
#     Returns:
#         ``torch.nn.Module``: **Unet**
#
#     .. _Unet:
#         https://arxiv.org/pdf/1505.04597
#
#     """
#
#     def __init__(
#         self,
#         arch: str = 'unet',
#         encoder_name: str = "resnet34",
#         encoder_depth: int = 5,
#         encoder_weights: str = "imagenet",
#         decoder_channels: List[int] = (256, 128, 64, 32, 16),
#         in_channels: int = 3,
#         num_classes: int = 1,
#         aux_params: Optional[dict] = None,
#         segmentation_head_kernel_size: int = 3,
#     ):
#         super().__init__(num_classes=num_classes)
#
#         self.encoder = get_encoder(
#             encoder_name,
#             in_channels=in_channels,
#             depth=encoder_depth,
#             weights=encoder_weights,
#         )
#
#         self.decoder = get_decoder(
#             arch=arch,
#             encoder_channels=self.encoder.out_channels,
#             **kwargs,
#
#         )
#
#         self.segmentation_head = SegmentationHead(
#             in_channels=decoder_channels[-1],
#             out_channels=num_classes,
#             activation=activation,
#             kernel_size=segmentation_head_kernel_size,
#         )
#
#         if aux_params is not None:
#             self.classification_head = ClassificationHead(
#                 in_channels=self.encoder.out_channels[-1], **aux_params
#             )
#         else:
#             self.classification_head = None
#
#         self.name = "u-{}".format(encoder_name)
#         self.initialize()
