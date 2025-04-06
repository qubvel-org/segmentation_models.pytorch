from typing import Any, Optional, Union, Callable, Sequence
import torch

from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.encoders.timm_vit import TimmViTEncoder
from segmentation_models_pytorch.base.utils import is_torch_compiling
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading
from .decoder import DPTDecoder, DPTSegmentationHead


class DPT(SegmentationModel):
    """
    DPT is a dense prediction architecture that leverages vision transformers in place of convolutional networks as
    a backbone for dense prediction tasks

    It assembles tokens from various stages of the vision transformer into image-like representations at various resolutions
    and progressively combines them into full-resolution predictions using a convolutional decoder.

    The transformer backbone processes representations at a constant and relatively high resolution and has a global receptive
    field at every stage. These properties allow the dense vision transformer to provide finer-grained and more globally coherent
    predictions when compared to fully-convolutional networks

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [1,4]. Each stage generate features
            smaller by a factor equal to the ViT model patch_size in spatial dimensions.
            Default is 4
        encoder_weights: One of **None** (random initialization), or other pretrained weights (see table with
            available weights for each encoder_name)
        encoder_output_indices: The indices of the encoder output features to use. If **None** will be sampled uniformly
            across the number of blocks in encoder, e.g. if number of blocks is 4 and encoder has 20 blocks, then
            encoder_output_indices will be (4, 9, 14, 19). If specified the number of indices should be equal to
            encoder_depth. Default is **None**.
        decoder_intermediate_channels: The number of channels for the intermediate decoder layers. Reduce if you
            want to reduce the number of parameters in the decoder. Default is (256, 512, 1024, 1024).
        decoder_fusion_channels: The latent dimension to which the encoder features will be projected to before fusion.
            Default is 256.
        in_channels: Number of input channels for the model, default is 3 (RGB images)
        classes: Number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models. Keys with
                ``None`` values are pruned before passing.
                allow_downsampling : Allow ViT encoder to have progressive spatial downsampling for it's representations.
                Set to False for DPT as the architecture requires all encoder feature outputs to have the same spatial shape.
                allow_output_stride_not_power_of_two : Allow ViT encoders with output_stride not being a power of 2. This
                    is set False for DPT as the architecture requires the encoder output features to have an output stride of
                    [1/32,1/16,1/8,1/4]

    Returns:
        ``torch.nn.Module``: DPT


    """

    _is_torch_scriptable = False
    _is_torch_compilable = False
    requires_divisible_input_shape = True

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "tu-vit_base_patch8_224",
        encoder_depth: int = 4,
        encoder_weights: Optional[str] = None,
        encoder_output_indices: Optional[list[int]] = None,
        decoder_intermediate_channels: Sequence[int] = (256, 512, 1024, 1024),
        decoder_fusion_channels: int = 256,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()
        if encoder_name.startswith("tu-"):
            encoder_name = encoder_name[3:]
        else:
            raise ValueError(
                f"Only Timm encoders are supported for DPT. Encoder name must start with 'tu-', got {encoder_name}"
            )

        self.encoder = TimmViTEncoder(
            name=encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            pretrained=encoder_weights is not None,
            output_indices=encoder_output_indices,
            **kwargs,
        )

        self.decoder = DPTDecoder(
            encoder_out_channels=self.encoder.out_channels,
            intermediate_channels=decoder_intermediate_channels,
            fusion_channels=decoder_fusion_channels,
            encoder_output_strides=self.encoder.output_strides,
            has_cls_token=self.encoder.has_class_token,
        )

        self.segmentation_head = DPTSegmentationHead(
            in_channels=decoder_fusion_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            upsampling=2,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "dpt-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        if not (
            torch.jit.is_scripting() or torch.jit.is_tracing() or is_torch_compiling()
        ):
            self.check_input_shape(x)

        features, cls_tokens = self.encoder(x)
        decoder_output = self.decoder(features, cls_tokens)
        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks
