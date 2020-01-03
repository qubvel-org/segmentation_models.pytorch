Models 
======

Architectures 
^^^^^^^^^^^^^^

-  `Unet <https://arxiv.org/abs/1505.04597>`__
-  `Linknet <https://arxiv.org/abs/1707.03718>`__
-  `FPN <http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf>`__
-  `PSPNet <https://arxiv.org/abs/1612.01105>`__
-  `PAN <https://arxiv.org/abs/1805.10180>`__

.. autoclass:: segmentation_models_pytorch.Unet

Encoders 
^^^^^^^^^

=================== =========================== =========
Encoder             Weights                     Params, M
=================== =========================== =========
resnet18            imagenet                    11M
resnet34            imagenet                    21M
resnet50            imagenet                    23M
resnet101           imagenet                    42M
resnet152           imagenet                    58M
resnext50_32x4d     imagenet                    22M
resnext101_32x8d    imagenetinstagram           86M
resnext101_32x16d   instagram                   191M
resnext101_32x32d   instagram                   466M
resnext101_32x48d   instagram                   826M
dpn68               imagenet                    11M
dpn68b              imagenet+5k                 11M
dpn92               imagenet+5k                 34M
dpn98               imagenet                    58M
dpn107              imagenet+5k                 84M
dpn131              imagenet                    76M
vgg11               imagenet                    9M
vgg11_bn            imagenet                    9M
vgg13               imagenet                    9M
vgg13_bn            imagenet                    9M
vgg16               imagenet                    14M
vgg16_bn            imagenet                    14M
vgg19               imagenet                    20M
vgg19_bn            imagenet                    20M
senet154            imagenet                    113M
se_resnet50         imagenet                    26M
se_resnet101        imagenet                    47M
se_resnet152        imagenet                    64M
se_resnext50_32x4d  imagenet                    25M
se_resnext101_32x4d imagenet                    46M
densenet121         imagenet                    6M
densenet169         imagenet                    12M
densenet201         imagenet                    18M
densenet161         imagenet                    26M
inceptionresnetv2   imagenetimagenet+background 54M
inceptionv4         imagenetimagenet+background 41M
efficientnet-b0     imagenet                    4M
efficientnet-b1     imagenet                    6M
efficientnet-b2     imagenet                    7M
efficientnet-b3     imagenet                    10M
efficientnet-b4     imagenet                    17M
efficientnet-b5     imagenet                    28M
efficientnet-b6     imagenet                    40M
efficientnet-b7     imagenet                    63M
mobilenet_v2        imagenet                    2M
xception            imagenet                    22M
=================== =========================== =========
