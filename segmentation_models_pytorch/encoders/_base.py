import torch
import torch.nn as nn
from typing import List
from collections import OrderedDict


class EncoderMixin:
    """Add encoder functionality such as:
        - output channels specification of feature tensors (produced by encoder)
        - patching first convolution for arbitrary input channels
    """

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    def set_in_channels(self, in_channels):
        """Change first convolution chennels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        patch_first_conv(model=self, in_channels=in_channels)

    def replace_strides_with_dilation(self, stages: List):

        def get_layer_by_name(module, name):
            for name in name.split('.'):
                module = getattr(module, name)
            return module

        for i, stage in enumerate(stages, 1):
            stage_strided_layers = self._strided_layers[stage]
            stage_dilation_rate = (2 ** i, 2 ** i)
            for layer_name in stage_strided_layers:
                module = get_layer_by_name(self, layer_name)

                # replace strides with dilation
                module.dilation = stage_dilation_rate
                module.stride = (1, 1)

                # change padding
                k = module.kernel_size
                d = module.dilation
                module.padding = ((k[0] // 2) * d[0], (k[1] // 2) * d[1])


def patch_first_conv(model, in_channels):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            break

    # change input channels for first conv
    module.in_channels = in_channels
    weight = module.weight.detach()
    reset = False

    if in_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    else:
        reset = True
        weight = torch.Tensor(
            module.out_channels,
            module.in_channels // module.groups,
            *module.kernel_size
        )

    module.weight = nn.parameter.Parameter(weight)
    if reset:
        module.reset_parameters()
