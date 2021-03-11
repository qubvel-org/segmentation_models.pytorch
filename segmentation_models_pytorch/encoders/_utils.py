import torch
import torch.nn as nn


def _get_copied_weights(src_in_size, dst_in_size, module):
    """Copy initialization strategy
    See details in https://levir.buaa.edu.cn/publications/coinnet.pdf
    """

    weight = module.weight.detach()

    new_weight = torch.Tensor(
        module.out_channels,
        dst_in_size,
        *module.kernel_size
    )
    for i in range(0, dst_in_size, src_in_size):
        k = min(i + src_in_size, dst_in_size) - i
        new_weight[:, i:i+k] = weight[:, :k] * ((src_in_size + 0.0) / dst_in_size)

    return new_weight


def patch_first_conv(model, in_channels, weights_init_mode):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            break

    if module.in_channels != 3:
        raise ValueError(f"in_channels=3 is expected in the first Conv2d layer, got in_channels={in_channels}")

    # change input channels for first conv
    if in_channels % module.groups > 0:
        raise ValueError(f"Try to change layer input channels to {in_channels} that is not divisible by groups={module.groups}")

    module.in_channels = in_channels
    weight = module.weight.detach()
    reset = False

    dst_in_size = module.in_channels // module.groups

    # 3 is divisible by module.groups so module.groups is equal to 1 or 3
    if module.groups == 3:
        assert weights_init_mode == "copy_init", "Specify 'copy_init' as weights initialization mode"

        new_weight = _get_copied_weights(1, dst_in_size, module)
    else:  # module.groups == 1
        if in_channels == 1:
            new_weight = weight.sum(1, keepdim=True)
        elif in_channels == 2:
            new_weight = weight[:, :2] * (3.0 / 2.0)
        else:
            if weights_init_mode is None:
                reset = True
                new_weight = torch.Tensor(
                    module.out_channels,
                    dst_in_size,
                    *module.kernel_size
                )
            elif weights_init_mode == "copy_init":
                new_weight = _get_copied_weights(3, dst_in_size, module)
            else:
                raise ValueError("Wrong weights initialization mode. Available options are: 'copy_init' or None")

    module.weight = nn.parameter.Parameter(new_weight)
    if reset:
        module.reset_parameters()


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
