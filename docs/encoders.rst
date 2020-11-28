🏔 Available Encoders
=====================

ResNets
~~~~~~~

==================== ================================== =========
Encoder              Weights                            Params, M
==================== ================================== =========
resnet18             imagenet / ssl / swsl              11M
resnet34             imagenet                           21M
resnet50             imagenet / ssl / swsl              23M
resnet101            imagenet                           42M
resnet152            imagenet                           58M
==================== ================================== =========

ResNeXts
~~~~~~~

==================== ================================== =========
Encoder              Weights                            Params, M
==================== ================================== =========
resnext50_32x4d      imagenet / ssl / swsl              22M
resnext101_32x4d     ssl / swsl                         42M
resnext101_32x8d     imagenet / instagram / ssl / swsl  86M
resnext101_32x16d    instagram / ssl / swsl             191M
resnext101_32x32d    instagram                          466M
resnext101_32x48d    instagram                          826M
==================== ================================== =========

DPNs
~~~~~~~

==================== ================================== =========
Encoder              Weights                            Params, M
==================== ================================== =========
dpn68                imagenet                           11M
dpn68b               imagenet+5k                        11M
dpn92                imagenet+5k                        34M
dpn98                imagenet                           58M
dpn107               imagenet+5k                        84M
dpn131               imagenet                           76M
==================== ================================== =========

VGGs
~~~~~~~

==================== ================================== =========
Encoder              Weights                            Params, M
==================== ================================== =========
vgg11                imagenet                           9M
vgg11_bn             imagenet                           9M
vgg13                imagenet                           9M
vgg13_bn             imagenet                           9M
vgg16                imagenet                           14M
vgg16_bn             imagenet                           14M
vgg19                imagenet                           20M
vgg19_bn             imagenet                           20M
==================== ================================== =========

SE-Nets
~~~~~~~

==================== ================================== =========
Encoder              Weights                            Params, M
==================== ================================== =========
senet154             imagenet                           113M
se_resnet50          imagenet                           26M
se_resnet101         imagenet                           47M
se_resnet101         imagenet                           47M
se_resnet152         imagenet                           64M
se_resnext50_32x4d   imagenet                           25M
se_resnext101_32x4d  imagenet                           46M
==================== ================================== =========

DenseNets
~~~~~~~

==================== ================================== =========
Encoder              Weights                            Params, M
==================== ================================== =========
densenet121          imagenet                           6M
densenet169          imagenet                           12M
densenet201          imagenet                           18M
densenet161          imagenet                           26M
==================== ================================== =========

Inceptions
~~~~~~~

==================== ================================== =========
Encoder              Weights                            Params, M
==================== ================================== =========
inceptionresnetv2    imagenet / imagenet+background     54M
inceptionv4          imagenet / imagenet+background     41M
xception             imagenet                           22M
==================== ================================== =========

EfficientNets
~~~~~~~

==================== ================================== =========
Encoder              Weights                            Params, M
==================== ================================== =========
efficientnet-b0      imagenet                           4M
efficientnet-b1      imagenet                           6M
efficientnet-b2      imagenet                           7M
efficientnet-b3      imagenet                           10M
efficientnet-b4      imagenet                           17M
efficientnet-b5      imagenet                           28M
efficientnet-b6      imagenet                           40M
efficientnet-b7      imagenet                           63M
timm-efficientnet-b0 imagenet / advprop / noisy-student 4M
timm-efficientnet-b1 imagenet / advprop / noisy-student 6M
timm-efficientnet-b2 imagenet / advprop / noisy-student 7M
timm-efficientnet-b3 imagenet / advprop / noisy-student 10M
timm-efficientnet-b4 imagenet / advprop / noisy-student 17M
timm-efficientnet-b5 imagenet / advprop / noisy-student 28M
timm-efficientnet-b6 imagenet / advprop / noisy-student 40M
timm-efficientnet-b7 imagenet / advprop / noisy-student 63M
timm-efficientnet-b8 imagenet / advprop /               84M
timm-efficientnet-l2 noisy-student                      474M
==================== ================================== =========

MobileNets
~~~~~~~

==================== ================================== =========
Encoder              Weights                            Params, M
==================== ================================== =========
mobilenet_v2         imagenet                           2M
==================== ================================== =========