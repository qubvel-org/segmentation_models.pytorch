from segmentation_models_pytorch.base.modules import Conv2dReLU


def test_conv2drelu_batchnorm():
    module = Conv2dReLU(3, 16, kernel_size=3, padding=1, use_norm="batchnorm")

    expected = ('Conv2dReLU(\n  (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))'
                '\n  (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)'
                '\n  (2): ReLU(inplace=True)\n)')
    assert repr(module) == expected

def test_conv2drelu_batchnorm_with_keywords():
    module = Conv2dReLU(3, 16, kernel_size=3, padding=1, use_norm={"type": "batchnorm", "momentum": 1e-4, "affine": False})

    expected = ('Conv2dReLU(\n  (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))'
                '\n  (1): BatchNorm2d(16, eps=1e-05, momentum=0.0001, affine=False, track_running_stats=True)'
                '\n  (2): ReLU(inplace=True)\n)')
    assert repr(module) == expected


def test_conv2drelu_identity():
    module = Conv2dReLU(3, 16, kernel_size=3, padding=1, use_norm="identity")
    expected = ('Conv2dReLU(\n  (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))'
                '\n  (1): Identity()'
                '\n  (2): ReLU(inplace=True)\n)')
    assert repr(module) == expected


def test_conv2drelu_layernorm():
    module = Conv2dReLU(3, 16, kernel_size=3, padding=1, use_norm="layernorm")
    expected = ('Conv2dReLU(\n  (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))'
                '\n  (1): LayerNorm((16,), eps=1e-05, elementwise_affine=True)'
                '\n  (2): ReLU(inplace=True)\n)')
    assert repr(module) == expected

def test_conv2drelu_groupnorm():
    module = Conv2dReLU(3, 16, kernel_size=3, padding=1, use_norm="groupnorm")
    expected = ('Conv2dReLU(\n  (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))'
                '\n  (1): Identity()'
                '\n  (2): ReLU(inplace=True)\n)')
    assert repr(module) == expected

def test_conv2drelu_instancenorm():
    module = Conv2dReLU(3, 16, kernel_size=3, padding=1, use_norm="instancenorm")
    expected = ('Conv2dReLU(\n  (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))'
                '\n  (1): InstanceNorm2d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)'
                '\n  (2): ReLU(inplace=True)\n)')
    assert repr(module) == expected
