import torch
from typing import Sequence, Dict

from . import _utils as utils


class EncoderMixin:
    """Add encoder functionality such as:
    - output channels specification of feature tensors (produced by encoder)
    - patching first convolution for arbitrary input channels
    """

    _is_torch_scriptable = True
    _is_torch_exportable = True
    _is_torch_compilable = True

    def __init__(self):
        self._depth = 5
        self._in_channels = 3
        self._output_stride = 32

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)

    def set_in_channels(self, in_channels, pretrained=True):
        """Change first convolution channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = [in_channels] + self._out_channels[1:]

        utils.patch_first_conv(
            model=self, new_in_channels=in_channels, pretrained=pretrained
        )

    def get_stages(self) -> Dict[int, Sequence[torch.nn.Module]]:
        """Override it in your implementation, should return a dictionary with keys as
        the output stride and values as the list of modules
        """
        raise NotImplementedError

    def make_dilated(self, output_stride):
        if output_stride not in [8, 16]:
            raise ValueError(f"Output stride should be 16 or 8, got {output_stride}.")

        stages = self.get_stages()
        for stage_stride, stage_modules in stages.items():
            if stage_stride <= output_stride:
                continue

            dilation_rate = stage_stride // output_stride
            for module in stage_modules:
                utils.replace_strides_with_dilation(module, dilation_rate)
