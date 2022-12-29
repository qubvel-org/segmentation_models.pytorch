import torch
import torch.nn as nn


def find_input_conv_layer(model: nn.Module, number_of_channels: int = 3):
    """Find first convolution layer in model with specified number of input channels"""
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == number_of_channels:
            return module
    raise ValueError("Can't find input conv layer in model")


def change_conv_in_channels(
    module: nn.Module,
    new_in_channels: int,
    pretrained: bool = True,
):
    """Change convolution layer input channels.
    In case:
        in_channels == 1: sum of original weights
        in_channels > 1: new_weight[i] = weight[i % initial_in_channels] * (initial_in_channels / new_in_channels)

    Args:
        module (nn.Module): Conv2d layer
        new_in_channels (int): new number of input channels
        pretrained (bool): if True, weights will be initialized with pretrained weights
    """

    if not pretrained:
        new_weight = torch.tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)
        module.weight = nn.parameter.Parameter(new_weight)
        module.reset_parameters()
        return

    weight = module.weight.detach()

    initial_in_channels = module.in_channels
    module.in_channels = new_in_channels

    if new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)

    else:
        new_weight = torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % initial_in_channels]

        new_weight = new_weight * (initial_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)


def replace_strides_with_dilation(module, dilation_rate):
    """Patch Conv2d modules replacing strides with dilation"""
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)

            # Kostyl for EfficientNet
            if hasattr(mod, "static_padding"):
                mod.static_padding = nn.Identity()
