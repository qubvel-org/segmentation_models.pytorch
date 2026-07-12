🔍 Available Encoders
=====================

**Segmentation Models PyTorch** provides support for a wide range of encoders.
This flexibility allows you to use these encoders with any model in the library by
specifying the encoder name in the ``encoder_name`` parameter during model initialization.

Here’s a quick example of using a ResNet34 encoder with the ``Unet`` model:

.. code-block:: python

    from segmentation_models_pytorch import Unet

    # Initialize Unet with ResNet34 encoder pre-trained on ImageNet
    model = Unet(encoder_name="resnet34", encoder_weights="imagenet")


The following encoder families are supported by the library, enabling you to choose the one that best fits your use case:

- **Mix Vision Transformer (mit)**
- **MobileOne**
- **MobileNet**
- **EfficientNet**
- **ResNet**
- **ResNeXt**
- **SENet**
- **DPN**
- **VGG**
- **DenseNet**
- **Xception**
- **Inception**

Choosing the Right Encoder
--------------------------

1. **Small Models for Edge Devices**
   Consider encoders like **MobileNet** or **MobileOne**, which have a smaller parameter count and are optimized for lightweight deployment.

2. **High Performance**
   If you require state-of-the-art accuracy **Mix Vision Transformer (mit)**, **EfficientNet** families offer excellent balance between performance and computational efficiency.

For each encoder, the table below provides detailed information:

1. **Pretrained Weights**
   Specifies the available pretrained weights (e.g., ``imagenet``, ``imagenet21k``).

2. **Params, M**:
   The total number of parameters in the encoder, measured in millions. This metric helps you assess the model's size and computational requirements.

3. **Script**:
   Indicates whether the encoder can be scripted with ``torch.jit.script``.

4. **Compile**:
   Indicates whether the encoder is compatible with ``torch.compile(model, fullgraph=True, dynamic=True, backend="eager")``.
   You may still get some issues with another backends, such as ``inductor``, depending on the torch/cuda/... dependencies version,
   but most of the time it will work.

5. **Export**:
   Indicates whether the encoder can be exported using ``torch.export.export``, making it suitable for deployment in different environments (e.g., ONNX).


============================ ==================================== =========== ======== ========= ========
Encoder                      Pretrained weights                   Params, M   Script   Compile   Export
============================ ==================================== =========== ======== ========= ========
resnet18                     imagenet / ssl / swsl                11M         ✅        ✅         ✅
resnet34                     imagenet                             21M         ✅        ✅         ✅
resnet50                     imagenet / ssl / swsl                23M         ✅        ✅         ✅
resnet101                    imagenet                             42M         ✅        ✅         ✅
resnet152                    imagenet                             58M         ✅        ✅         ✅
resnext50_32x4d              imagenet / ssl / swsl                22M         ✅        ✅         ✅
resnext101_32x4d             ssl / swsl                           42M         ✅        ✅         ✅
resnext101_32x8d             imagenet / instagram / ssl / swsl    86M         ✅        ✅         ✅
resnext101_32x16d            instagram / ssl / swsl               191M        ✅        ✅         ✅
resnext101_32x32d            instagram                            466M        ✅        ✅         ✅
resnext101_32x48d            instagram                            826M        ✅        ✅         ✅
dpn68                        imagenet                             11M         ❌        ✅         ✅
dpn68b                       imagenet+5k                          11M         ❌        ✅         ✅
dpn92                        imagenet+5k                          34M         ❌        ✅         ✅
dpn98                        imagenet                             58M         ❌        ✅         ✅
dpn107                       imagenet+5k                          84M         ❌        ✅         ✅
dpn131                       imagenet                             76M         ❌        ✅         ✅
vgg11                        imagenet                             9M          ✅        ✅         ✅
vgg11_bn                     imagenet                             9M          ✅        ✅         ✅
vgg13                        imagenet                             9M          ✅        ✅         ✅
vgg13_bn                     imagenet                             9M          ✅        ✅         ✅
vgg16                        imagenet                             14M         ✅        ✅         ✅
vgg16_bn                     imagenet                             14M         ✅        ✅         ✅
vgg19                        imagenet                             20M         ✅        ✅         ✅
vgg19_bn                     imagenet                             20M         ✅        ✅         ✅
senet154                     imagenet                             113M        ✅        ✅         ✅
se_resnet50                  imagenet                             26M         ✅        ✅         ✅
se_resnet101                 imagenet                             47M         ✅        ✅         ✅
se_resnet152                 imagenet                             64M         ✅        ✅         ✅
se_resnext50_32x4d           imagenet                             25M         ✅        ✅         ✅
se_resnext101_32x4d          imagenet                             46M         ✅        ✅         ✅
densenet121                  imagenet                             6M          ✅        ✅         ✅
densenet169                  imagenet                             12M         ✅        ✅         ✅
densenet201                  imagenet                             18M         ✅        ✅         ✅
densenet161                  imagenet                             26M         ✅        ✅         ✅
inceptionresnetv2            imagenet / imagenet+background       54M         ✅        ✅         ✅
inceptionv4                  imagenet / imagenet+background       41M         ✅        ✅         ✅
efficientnet-b0              imagenet / advprop                   4M          ✅        ✅         ✅
efficientnet-b1              imagenet / advprop                   6M          ✅        ✅         ✅
efficientnet-b2              imagenet / advprop                   7M          ✅        ✅         ✅
efficientnet-b3              imagenet / advprop                   10M         ✅        ✅         ✅
efficientnet-b4              imagenet / advprop                   17M         ✅        ✅         ✅
efficientnet-b5              imagenet / advprop                   28M         ✅        ✅         ✅
efficientnet-b6              imagenet / advprop                   40M         ✅        ✅         ✅
efficientnet-b7              imagenet / advprop                   63M         ✅        ✅         ✅
mobilenet_v2                 imagenet                             2M          ✅        ✅         ✅
xception                     imagenet                             20M         ✅        ✅         ✅
timm-efficientnet-b0         imagenet / advprop / noisy-student   4M          ✅        ✅         ✅
timm-efficientnet-b1         imagenet / advprop / noisy-student   6M          ✅        ✅         ✅
timm-efficientnet-b2         imagenet / advprop / noisy-student   7M          ✅        ✅         ✅
timm-efficientnet-b3         imagenet / advprop / noisy-student   10M         ✅        ✅         ✅
timm-efficientnet-b4         imagenet / advprop / noisy-student   17M         ✅        ✅         ✅
timm-efficientnet-b5         imagenet / advprop / noisy-student   28M         ✅        ✅         ✅
timm-efficientnet-b6         imagenet / advprop / noisy-student   40M         ✅        ✅         ✅
timm-efficientnet-b7         imagenet / advprop / noisy-student   63M         ✅        ✅         ✅
timm-efficientnet-b8         imagenet / advprop                   84M         ✅        ✅         ✅
timm-efficientnet-l2         noisy-student / noisy-student-475    474M        ✅        ✅         ✅
timm-tf_efficientnet_lite0   imagenet                             3M          ✅        ✅         ✅
timm-tf_efficientnet_lite1   imagenet                             4M          ✅        ✅         ✅
timm-tf_efficientnet_lite2   imagenet                             4M          ✅        ✅         ✅
timm-tf_efficientnet_lite3   imagenet                             6M          ✅        ✅         ✅
timm-tf_efficientnet_lite4   imagenet                             11M         ✅        ✅         ✅
timm-skresnet18              imagenet                             11M         ✅        ✅         ✅
timm-skresnet34              imagenet                             21M         ✅        ✅         ✅
timm-skresnext50_32x4d       imagenet                             23M         ✅        ✅         ✅
mit_b0                       imagenet                             3M          ✅        ✅         ✅
mit_b1                       imagenet                             13M         ✅        ✅         ✅
mit_b2                       imagenet                             24M         ✅        ✅         ✅
mit_b3                       imagenet                             44M         ✅        ✅         ✅
mit_b4                       imagenet                             60M         ✅        ✅         ✅
mit_b5                       imagenet                             81M         ✅        ✅         ✅
mobileone_s0                 imagenet                             4M          ✅        ✅         ✅
mobileone_s1                 imagenet                             3M          ✅        ✅         ✅
mobileone_s2                 imagenet                             5M          ✅        ✅         ✅
mobileone_s3                 imagenet                             8M          ✅        ✅         ✅
mobileone_s4                 imagenet                             12M         ✅        ✅         ✅
============================ ==================================== =========== ======== ========= ========
