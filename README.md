<div align="center">
 
![logo](https://i.ibb.co/dc1XdhT/Segmentation-Models-V2-Side-1-1.png)  
**Python library with Neural Networks for Image  
Segmentation based on [PyTorch](https://pytorch.org/).**  

![PyPI version](https://badge.fury.io/py/segmentation-models-pytorch.svg) [![Build Status](https://travis-ci.com/qubvel/segmentation_models.pytorch.svg?branch=master)](https://travis-ci.com/qubvel/segmentation_models.pytorch) [![Documentation Status](https://readthedocs.org/projects/smp/badge/?version=latest)](https://smp.readthedocs.io/en/latest/?badge=latest) <br> ![Downloads](https://pepy.tech/badge/segmentation-models-pytorch) [![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg)](https://shields.io/)

</div>

The main features of this library are:

 - High level API (just two lines to create a neural network)
 - 9 models architectures for binary and multi class segmentation (including legendary Unet)
 - 104 available encoders
 - All encoders have pre-trained weights for faster and better convergence
 
### [📚 Project Documentation 📚](http://smp.readthedocs.io/)

Visit [Read The Docs Project Page](https://smp.readthedocs.io/) or read following README to know more about Segmentation Models Pytorch (SMP for short) library

### 📋 Table of content
 1. [Quick start](#start)
 2. [Examples](#examples)
 3. [Models](#models)
    1. [Architectures](#architectures)
    2. [Encoders](#encoders)
 4. [Models API](#api)
    1. [Input channels](#input-channels)
    2. [Auxiliary classification output](#auxiliary-classification-output)
    3. [Depth](#depth)
 5. [Installation](#installation)
 6. [Competitions won with the library](#competitions-won-with-the-library)
 7. [Contributing](#contributing)
 8. [Citing](#citing)
 9. [License](#license)

### ⏳ Quick start <a name="start"></a>

#### 1. Create your first Segmentation model with SMP

Segmentation model is just a PyTorch nn.Module, which can be created as easy as:

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3,                      # model output channels (number of classes in your dataset)
)
```
 - see [table](#architectires) with available model architectures
 - see [table](#encoders) with available encoders and their corresponding weights

#### 2. Configure data preprocessing

All encoders have pretrained weights. Preparing your data the same way as during weights pre-training may give your better results (higher metric score and faster convergence). But it is relevant only for 1-2-3-channels images and **not necessary** in case you train the whole model, not only decoder.

```python
from segmentation_models_pytorch.encoders import get_preprocessing_fn

preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')
```

Congratulations! You are done! Now you can train your model with your favorite framework!

### 💡 Examples <a name="examples"></a>
 - Training model for cars segmentation on CamVid dataset [here](https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb).
 - Training SMP model with [Catalyst](https://github.com/catalyst-team/catalyst) (high-level framework for PyTorch), [TTAch](https://github.com/qubvel/ttach) (TTA library for PyTorch) and [Albumentations](https://github.com/albu/albumentations) (fast image augmentation library) - [here](https://github.com/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb)
 - Training SMP model with [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io) framework - [here](https://github.com/ternaus/cloths_segmentation) (clothes binary segmentation by [@teranus](https://github.com/ternaus)).

### 📦 Models <a name="models"></a>

#### Architectures <a name="architectures"></a>
 - Unet [[paper](https://arxiv.org/abs/1505.04597)] [[docs](https://smp.readthedocs.io/en/latest/models.html#unet)]
 - Unet++ [[paper](https://arxiv.org/pdf/1807.10165.pdf)] [[docs](https://smp.readthedocs.io/en/latest/models.html#id2)]
 - MAnet [[paper](https://ieeexplore.ieee.org/abstract/document/9201310)] [[docs](https://smp.readthedocs.io/en/latest/models.html#manet)]
 - Linknet [[paper](https://arxiv.org/abs/1707.03718)] [[docs](https://smp.readthedocs.io/en/latest/models.html#linknet)]
 - FPN [[paper](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)] [[docs](https://smp.readthedocs.io/en/latest/models.html#fpn)]
 - PSPNet [[paper](https://arxiv.org/abs/1612.01105)] [[docs](https://smp.readthedocs.io/en/latest/models.html#pspnet)]
 - PAN [[paper](https://arxiv.org/abs/1805.10180)] [[docs](https://smp.readthedocs.io/en/latest/models.html#pan)]
 - DeepLabV3 [[paper](https://arxiv.org/abs/1706.05587)] [[docs](https://smp.readthedocs.io/en/latest/models.html#deeplabv3)]
 - DeepLabV3+ [[paper](https://arxiv.org/abs/1802.02611)] [[docs](https://smp.readthedocs.io/en/latest/models.html#id9)]

#### Encoders <a name="encoders"></a>

The following is a list of supported encoders in the SMP. Select the appropriate family of encoders and click to expand the table and select a specific encoder and its pre-trained weights (`encoder_name` and `encoder_weights` parameters).

<details>
<summary style="margin-left: 25px;">ResNet</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|resnet18                        |imagenet / ssl / swsl           |11M                             |
|resnet34                        |imagenet                        |21M                             |
|resnet50                        |imagenet / ssl / swsl           |23M                             |
|resnet101                       |imagenet                        |42M                             |
|resnet152                       |imagenet                        |58M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">ResNeXt</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|resnext50_32x4d                 |imagenet / ssl / swsl           |22M                             |
|resnext101_32x4d                |ssl / swsl                      |42M                             |
|resnext101_32x8d                |imagenet / instagram / ssl / swsl|86M                         |
|resnext101_32x16d               |instagram / ssl / swsl          |191M                            |
|resnext101_32x32d               |instagram                       |466M                            |
|resnext101_32x48d               |instagram                       |826M                            |

</div>
</details>

<details>
<summary style="margin-left: 25px;">ResNeSt</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|timm-resnest14d                 |imagenet                        |8M                              |
|timm-resnest26d                 |imagenet                        |15M                             |
|timm-resnest50d                 |imagenet                        |25M                             |
|timm-resnest101e                |imagenet                        |46M                             |
|timm-resnest200e                |imagenet                        |68M                             |
|timm-resnest269e                |imagenet                        |108M                            |
|timm-resnest50d_4s2x40d         |imagenet                        |28M                             |
|timm-resnest50d_1s4x24d         |imagenet                        |23M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">Res2Ne(X)t</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|timm-res2net50_26w_4s           |imagenet                        |23M                             |
|timm-res2net101_26w_4s          |imagenet                        |43M                             |
|timm-res2net50_26w_6s           |imagenet                        |35M                             |
|timm-res2net50_26w_8s           |imagenet                        |46M                             |
|timm-res2net50_48w_2s           |imagenet                        |23M                             |
|timm-res2net50_14w_8s           |imagenet                        |23M                             |
|timm-res2next50                 |imagenet                        |22M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">RegNet(x/y)</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|timm-regnetx_002                |imagenet                        |2M                              |
|timm-regnetx_004                |imagenet                        |4M                              |
|timm-regnetx_006                |imagenet                        |5M                              |
|timm-regnetx_008                |imagenet                        |6M                              |
|timm-regnetx_016                |imagenet                        |8M                              |
|timm-regnetx_032                |imagenet                        |14M                             |
|timm-regnetx_040                |imagenet                        |20M                             |
|timm-regnetx_064                |imagenet                        |24M                             |
|timm-regnetx_080                |imagenet                        |37M                             |
|timm-regnetx_120                |imagenet                        |43M                             |
|timm-regnetx_160                |imagenet                        |52M                             |
|timm-regnetx_320                |imagenet                        |105M                            |
|timm-regnety_002                |imagenet                        |2M                              |
|timm-regnety_004                |imagenet                        |3M                              |
|timm-regnety_006                |imagenet                        |5M                              |
|timm-regnety_008                |imagenet                        |5M                              |
|timm-regnety_016                |imagenet                        |10M                             |
|timm-regnety_032                |imagenet                        |17M                             |
|timm-regnety_040                |imagenet                        |19M                             |
|timm-regnety_064                |imagenet                        |29M                             |
|timm-regnety_080                |imagenet                        |37M                             |
|timm-regnety_120                |imagenet                        |49M                             |
|timm-regnety_160                |imagenet                        |80M                             |
|timm-regnety_320                |imagenet                        |141M                            |

</div>
</details>

<details>
<summary style="margin-left: 25px;">SE-Net</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|senet154                        |imagenet                        |113M                            |
|se_resnet50                     |imagenet                        |26M                             |
|se_resnet101                    |imagenet                        |47M                             |
|se_resnet152                    |imagenet                        |64M                             |
|se_resnext50_32x4d              |imagenet                        |25M                             |
|se_resnext101_32x4d             |imagenet                        |46M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">SK-ResNe(X)t</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|timm-skresnet18                 |imagenet                        |11M                             |
|timm-skresnet34                 |imagenet                        |21M                             |
|timm-skresnext50_32x4d          |imagenet                        |25M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">DenseNet</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|densenet121                     |imagenet                        |6M                              |
|densenet169                     |imagenet                        |12M                             |
|densenet201                     |imagenet                        |18M                             |
|densenet161                     |imagenet                        |26M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">Inception</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|inceptionresnetv2               |imagenet /  imagenet+background |54M                             |
|inceptionv4                     |imagenet /  imagenet+background |41M                             |
|xception                        |imagenet                        |22M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">EfficientNet</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|efficientnet-b0                 |imagenet                        |4M                              |
|efficientnet-b1                 |imagenet                        |6M                              |
|efficientnet-b2                 |imagenet                        |7M                              |
|efficientnet-b3                 |imagenet                        |10M                             |
|efficientnet-b4                 |imagenet                        |17M                             |
|efficientnet-b5                 |imagenet                        |28M                             |
|efficientnet-b6                 |imagenet                        |40M                             |
|efficientnet-b7                 |imagenet                        |63M                             |
|timm-efficientnet-b0            |imagenet / advprop / noisy-student|4M                              |
|timm-efficientnet-b1            |imagenet / advprop / noisy-student|6M                              |
|timm-efficientnet-b2            |imagenet / advprop / noisy-student|7M                              |
|timm-efficientnet-b3            |imagenet / advprop / noisy-student|10M                             |
|timm-efficientnet-b4            |imagenet / advprop / noisy-student|17M                             |
|timm-efficientnet-b5            |imagenet / advprop / noisy-student|28M                             |
|timm-efficientnet-b6            |imagenet / advprop / noisy-student|40M                             |
|timm-efficientnet-b7            |imagenet / advprop / noisy-student|63M                             |
|timm-efficientnet-b8            |imagenet / advprop             |84M                             |
|timm-efficientnet-l2            |noisy-student                   |474M                            |
|timm-efficientnet-lite0         |imagenet                        |4M                              |
|timm-efficientnet-lite1         |imagenet                        |5M                              |
|timm-efficientnet-lite2         |imagenet                        |6M                              |
|timm-efficientnet-lite3         |imagenet                        |8M                             |
|timm-efficientnet-lite4         |imagenet                        |13M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">MobileNet</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|mobilenet_v2                    |imagenet                        |2M                              |

</div>
</details>

<details>
<summary style="margin-left: 25px;">DPN</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|dpn68                           |imagenet                        |11M                             |
|dpn68b                          |imagenet+5k                     |11M                             |
|dpn92                           |imagenet+5k                     |34M                             |
|dpn98                           |imagenet                        |58M                             |
|dpn107                          |imagenet+5k                     |84M                             |
|dpn131                          |imagenet                        |76M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">VGG</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|vgg11                           |imagenet                        |9M                              |
|vgg11_bn                        |imagenet                        |9M                              |
|vgg13                           |imagenet                        |9M                              |
|vgg13_bn                        |imagenet                        |9M                              |
|vgg16                           |imagenet                        |14M                             |
|vgg16_bn                        |imagenet                        |14M                             |
|vgg19                           |imagenet                        |20M                             |
|vgg19_bn                        |imagenet                        |20M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">timm-universal (0.4.5)</summary>
<div style="margin-left: 25px;">

|Encoder                                 |Weights                                 |Params, M                               |
|----------------------------------------|:--------------------------------------:|:--------------------------------------:|
|timm-u-adv_inception_v3                 |?                                       |21M                                     |
|timm-u-cspdarknet53                     |?                                       |9M                                      |
|timm-u-cspdarknet53_iabn                |?                                       |9M                                      |
|timm-u-cspresnet50                      |?                                       |20M                                     |
|timm-u-cspresnet50d                     |?                                       |20M                                     |
|timm-u-cspresnet50w                     |?                                       |26M                                     |
|timm-u-cspresnext50                     |?                                       |18M                                     |
|timm-u-cspresnext50_iabn                |?                                       |18M                                     |
|timm-u-darknet53                        |?                                       |14M                                     |
|timm-u-densenet121                      |?                                       |6M                                      |
|timm-u-densenet121d                     |?                                       |6M                                      |
|timm-u-densenet161                      |?                                       |26M                                     |
|timm-u-densenet169                      |?                                       |12M                                     |
|timm-u-densenet201                      |?                                       |18M                                     |
|timm-u-densenet264                      |?                                       |68M                                     |
|timm-u-densenet264d_iabn                |?                                       |68M                                     |
|timm-u-densenetblur121d                 |?                                       |6M                                      |
|timm-u-dla34                            |?                                       |6M                                      |
|timm-u-dla46_c                          |?                                       |0.4M                                    |
|timm-u-dla46x_c                         |?                                       |0.3M                                    |
|timm-u-dla60                            |?                                       |11M                                     |
|timm-u-dla60_res2net                    |?                                       |10M                                     |
|timm-u-dla60_res2next                   |?                                       |8M                                      |
|timm-u-dla60x                           |?                                       |8M                                      |
|timm-u-dla60x_c                         |?                                       |0.5M                                    |
|timm-u-dla102                           |?                                       |22M                                     |
|timm-u-dla102x                          |?                                       |17M                                     |
|timm-u-dla102x2                         |?                                       |28M                                     |
|timm-u-dla169                           |?                                       |42M                                     |
|timm-u-dm_nfnet_f0                      |?                                       |68M                                     |
|timm-u-dm_nfnet_f1                      |?                                       |129M                                    |
|timm-u-dm_nfnet_f2                      |?                                       |190M                                    |
|timm-u-dm_nfnet_f3                      |?                                       |251M                                    |
|timm-u-dm_nfnet_f4                      |?                                       |312M                                    |
|timm-u-dm_nfnet_f5                      |?                                       |374M                                    |
|timm-u-dm_nfnet_f6                      |?                                       |435M                                    |
|timm-u-dpn68                            |?                                       |11M                                     |
|timm-u-dpn68b                           |?                                       |11M                                     |
|timm-u-dpn92                            |?                                       |34M                                     |
|timm-u-dpn98                            |?                                       |58M                                     |
|timm-u-dpn107                           |?                                       |84M                                     |
|timm-u-dpn131                           |?                                       |76M                                     |
|timm-u-eca_vovnet39b                    |?                                       |21M                                     |
|timm-u-ecaresnet26t                     |?                                       |13M                                     |
|timm-u-ecaresnet50d                     |?                                       |23M                                     |
|timm-u-ecaresnet50d_pruned              |?                                       |17M                                     |
|timm-u-ecaresnet50t                     |?                                       |23M                                     |
|timm-u-ecaresnet101d                    |?                                       |42M                                     |
|timm-u-ecaresnet101d_pruned             |?                                       |22M                                     |
|timm-u-ecaresnet200d                    |?                                       |62M                                     |
|timm-u-ecaresnet269d                    |?                                       |100M                                    |
|timm-u-ecaresnetlight                   |?                                       |28M                                     |
|timm-u-ecaresnext26t_32x4d              |?                                       |13M                                     |
|timm-u-ecaresnext50t_32x4d              |?                                       |13M                                     |
|timm-u-efficientnet_b0                  |?                                       |3.5M                                    |
|timm-u-efficientnet_b1                  |?                                       |6M                                      |
|timm-u-efficientnet_b1_pruned           |?                                       |4.6M                                    |
|timm-u-efficientnet_b2                  |?                                       |7M                                      |
|timm-u-efficientnet_b2_pruned           |?                                       |6M                                      |
|timm-u-efficientnet_b2a                 |?                                       |7M                                      |
|timm-u-efficientnet_b3                  |?                                       |10M                                     |
|timm-u-efficientnet_b3_pruned           |?                                       |7M                                      |
|timm-u-efficientnet_b3a                 |?                                       |10M                                     |
|timm-u-efficientnet_b4                  |?                                       |16M                                     |
|timm-u-efficientnet_b5                  |?                                       |27M                                     |
|timm-u-efficientnet_b6                  |?                                       |39M                                     |
|timm-u-efficientnet_b7                  |?                                       |62M                                     |
|timm-u-efficientnet_b8                  |?                                       |82M                                     |
|timm-u-efficientnet_cc_b0_4e            |?                                       |11M                                     |
|timm-u-efficientnet_cc_b0_8e            |?                                       |22M                                     |
|timm-u-efficientnet_cc_b1_8e            |?                                       |38M                                     |
|timm-u-efficientnet_el                  |?                                       |8M                                      |
|timm-u-efficientnet_em                  |?                                       |5.3M                                    |
|timm-u-efficientnet_es                  |?                                       |3.9M                                    |
|timm-u-efficientnet_l2                  |?                                       |467M                                    |
|timm-u-efficientnet_lite0               |?                                       |2.9M                                    |
|timm-u-efficientnet_lite1               |?                                       |3.7M                                    |
|timm-u-efficientnet_lite2               |?                                       |4.3M                                    |
|timm-u-efficientnet_lite3               |?                                       |6M                                      |
|timm-u-efficientnet_lite4               |?                                       |11M                                     |
|timm-u-ens_adv_inception_resnet_v2      |?                                       |54M                                     |
|timm-u-ese_vovnet19b_dw                 |?                                       |5.5M                                    |
|timm-u-ese_vovnet19b_slim               |?                                       |2.6M                                    |
|timm-u-ese_vovnet19b_slim_dw            |?                                       |1.3M                                    |
|timm-u-ese_vovnet39b                    |?                                       |23M                                     |
|timm-u-ese_vovnet39b_evos               |?                                       |23M                                     |
|timm-u-ese_vovnet57b                    |?                                       |37M                                     |
|timm-u-ese_vovnet99b                    |?                                       |62M                                     |
|timm-u-ese_vovnet99b_iabn               |?                                       |62M                                     |
|timm-u-fbnetc_100                       |?                                       |2.8M                                    |
|timm-u-gernet_l                         |?                                       |28M                                     |
|timm-u-gernet_m                         |?                                       |18M                                     |
|timm-u-gernet_s                         |?                                       |6M                                      |
|timm-u-gluon_inception_v3               |?                                       |21M                                     |
|timm-u-gluon_resnet18_v1b               |?                                       |11M                                     |
|timm-u-gluon_resnet34_v1b               |?                                       |21M                                     |
|timm-u-gluon_resnet50_v1b               |?                                       |23M                                     |
|timm-u-gluon_resnet50_v1c               |?                                       |23M                                     |
|timm-u-gluon_resnet50_v1d               |?                                       |23M                                     |
|timm-u-gluon_resnet50_v1s               |?                                       |23M                                     |
|timm-u-gluon_resnet101_v1b              |?                                       |42M                                     |
|timm-u-gluon_resnet101_v1c              |?                                       |42M                                     |
|timm-u-gluon_resnet101_v1d              |?                                       |42M                                     |
|timm-u-gluon_resnet101_v1s              |?                                       |42M                                     |
|timm-u-gluon_resnet152_v1b              |?                                       |58M                                     |
|timm-u-gluon_resnet152_v1c              |?                                       |58M                                     |
|timm-u-gluon_resnet152_v1d              |?                                       |58M                                     |
|timm-u-gluon_resnet152_v1s              |?                                       |58M                                     |
|timm-u-gluon_resnext50_32x4d            |?                                       |22M                                     |
|timm-u-gluon_resnext101_32x4d           |?                                       |42M                                     |
|timm-u-gluon_resnext101_64x4d           |?                                       |81M                                     |
|timm-u-gluon_senet154                   |?                                       |113M                                    |
|timm-u-gluon_seresnext50_32x4d          |?                                       |25M                                     |
|timm-u-gluon_seresnext101_32x4d         |?                                       |46M                                     |
|timm-u-gluon_seresnext101_64x4d         |?                                       |86M                                     |
|timm-u-gluon_xception65                 |?                                       |37M                                     |
|timm-u-hrnet_w18                        |?                                       |10M                                     |
|timm-u-hrnet_w18_small                  |?                                       |2.8M                                    |
|timm-u-hrnet_w18_small_v2               |?                                       |5.2M                                    |
|timm-u-hrnet_w30                        |?                                       |27M                                     |
|timm-u-hrnet_w32                        |?                                       |30M                                     |
|timm-u-hrnet_w40                        |?                                       |47M                                     |
|timm-u-hrnet_w44                        |?                                       |56M                                     |
|timm-u-hrnet_w48                        |?                                       |67M                                     |
|timm-u-hrnet_w64                        |?                                       |117M                                    |
|timm-u-ig_resnext101_32x8d              |?                                       |86M                                     |
|timm-u-ig_resnext101_32x16d             |?                                       |191M                                    |
|timm-u-ig_resnext101_32x32d             |?                                       |466M                                    |
|timm-u-ig_resnext101_32x48d             |?                                       |826M                                    |
|timm-u-inception_resnet_v2              |?                                       |54M                                     |
|timm-u-inception_v3                     |?                                       |21M                                     |
|timm-u-inception_v4                     |?                                       |41M                                     |
|timm-u-legacy_senet154                  |?                                       |113M                                    |
|timm-u-legacy_seresnet18                |?                                       |11M                                     |
|timm-u-legacy_seresnet34                |?                                       |21M                                     |
|timm-u-legacy_seresnet50                |?                                       |26M                                     |
|timm-u-legacy_seresnet101               |?                                       |47M                                     |
|timm-u-legacy_seresnet152               |?                                       |64M                                     |
|timm-u-legacy_seresnext26_32x4d         |?                                       |14M                                     |
|timm-u-legacy_seresnext50_32x4d         |?                                       |25M                                     |
|timm-u-legacy_seresnext101_32x4d        |?                                       |46M                                     |
|timm-u-mixnet_l                         |?                                       |5.3M                                    |
|timm-u-mixnet_m                         |?                                       |3.1M                                    |
|timm-u-mixnet_s                         |?                                       |2.2M                                    |
|timm-u-mixnet_xl                        |?                                       |9M                                      |
|timm-u-mixnet_xxl                       |?                                       |21M                                     |
|timm-u-mnasnet_050                      |?                                       |0.7M                                    |
|timm-u-mnasnet_075                      |?                                       |1.5M                                    |
|timm-u-mnasnet_100                      |?                                       |2.6M                                    |
|timm-u-mnasnet_140                      |?                                       |5.2M                                    |
|timm-u-mnasnet_a1                       |?                                       |2.1M                                    |
|timm-u-mnasnet_b1                       |?                                       |2.6M                                    |
|timm-u-mnasnet_small                    |?                                       |0.5M                                    |
|timm-u-mobilenetv2_100                  |?                                       |1.8M                                    |
|timm-u-mobilenetv2_110d                 |?                                       |2.7M                                    |
|timm-u-mobilenetv2_120d                 |?                                       |4.0M                                    |
|timm-u-mobilenetv2_140                  |?                                       |3.5M                                    |
|timm-u-mobilenetv3_large_075            |?                                       |1.7M                                    |
|timm-u-mobilenetv3_large_100            |?                                       |2.9M                                    |
|timm-u-mobilenetv3_rw                   |?                                       |2.9M                                    |
|timm-u-mobilenetv3_small_075            |?                                       |0.5M                                    |
|timm-u-mobilenetv3_small_100            |?                                       |0.9M                                    |
|timm-u-nasnetalarge                     |?                                       |84M                                     |
|timm-u-nf_ecaresnet26                   |?                                       |13M                                     |
|timm-u-nf_ecaresnet50                   |?                                       |23M                                     |
|timm-u-nf_ecaresnet101                  |?                                       |42M                                     |
|timm-u-nf_regnet_b0                     |?                                       |7M                                      |
|timm-u-nf_regnet_b1                     |?                                       |9M                                      |
|timm-u-nf_regnet_b2                     |?                                       |13M                                     |
|timm-u-nf_regnet_b3                     |?                                       |17M                                     |
|timm-u-nf_regnet_b4                     |?                                       |28M                                     |
|timm-u-nf_regnet_b5                     |?                                       |48M                                     |
|timm-u-nf_resnet26                      |?                                       |13M                                     |
|timm-u-nf_resnet50                      |?                                       |23M                                     |
|timm-u-nf_resnet101                     |?                                       |42M                                     |
|timm-u-nf_seresnet26                    |?                                       |15M                                     |
|timm-u-nf_seresnet50                    |?                                       |26M                                     |
|timm-u-nf_seresnet101                   |?                                       |47M                                     |
|timm-u-nfnet_f0                         |?                                       |68M                                     |
|timm-u-nfnet_f0s                        |?                                       |68M                                     |
|timm-u-nfnet_f1                         |?                                       |129M                                    |
|timm-u-nfnet_f1s                        |?                                       |129M                                    |
|timm-u-nfnet_f2                         |?                                       |190M                                    |
|timm-u-nfnet_f2s                        |?                                       |190M                                    |
|timm-u-nfnet_f3                         |?                                       |251M                                    |
|timm-u-nfnet_f3s                        |?                                       |251M                                    |
|timm-u-nfnet_f4                         |?                                       |312M                                    |
|timm-u-nfnet_f4s                        |?                                       |312M                                    |
|timm-u-nfnet_f5                         |?                                       |374M                                    |
|timm-u-nfnet_f5s                        |?                                       |374M                                    |
|timm-u-nfnet_f6                         |?                                       |435M                                    |
|timm-u-nfnet_f6s                        |?                                       |435M                                    |
|timm-u-nfnet_f7                         |?                                       |496M                                    |
|timm-u-nfnet_f7s                        |?                                       |496M                                    |
|timm-u-nfnet_l0a                        |?                                       |27M                                     |
|timm-u-nfnet_l0b                        |?                                       |32M                                     |
|timm-u-nfnet_l0c                        |?                                       |21M                                     |
|timm-u-pnasnet5large                    |?                                       |81M                                     |
|timm-u-regnetx_002                      |?                                       |2.3M                                    |
|timm-u-regnetx_004                      |?                                       |4.7M                                    |
|timm-u-regnetx_006                      |?                                       |5.6M                                    |
|timm-u-regnetx_008                      |?                                       |6M                                      |
|timm-u-regnetx_016                      |?                                       |8M                                      |
|timm-u-regnetx_032                      |?                                       |14M                                     |
|timm-u-regnetx_040                      |?                                       |20M                                     |
|timm-u-regnetx_064                      |?                                       |24M                                     |
|timm-u-regnetx_080                      |?                                       |37M                                     |
|timm-u-regnetx_120                      |?                                       |43M                                     |
|timm-u-regnetx_160                      |?                                       |52M                                     |
|timm-u-regnetx_320                      |?                                       |105M                                    |
|timm-u-regnety_002                      |?                                       |2.7M                                    |
|timm-u-regnety_004                      |?                                       |3.9M                                    |
|timm-u-regnety_006                      |?                                       |5.4M                                    |
|timm-u-regnety_008                      |?                                       |5.4M                                    |
|timm-u-regnety_016                      |?                                       |10M                                     |
|timm-u-regnety_032                      |?                                       |17M                                     |
|timm-u-regnety_040                      |?                                       |19M                                     |
|timm-u-regnety_064                      |?                                       |29M                                     |
|timm-u-regnety_080                      |?                                       |37M                                     |
|timm-u-regnety_120                      |?                                       |49M                                     |
|timm-u-regnety_160                      |?                                       |80M                                     |
|timm-u-regnety_320                      |?                                       |141M                                    |
|timm-u-repvgg_a2                        |?                                       |26M                                     |
|timm-u-repvgg_b0                        |?                                       |14M                                     |
|timm-u-repvgg_b1                        |?                                       |55M                                     |
|timm-u-repvgg_b1g4                      |?                                       |37M                                     |
|timm-u-repvgg_b2                        |?                                       |86M                                     |
|timm-u-repvgg_b2g4                      |?                                       |59M                                     |
|timm-u-repvgg_b3                        |?                                       |120M                                    |
|timm-u-repvgg_b3g4                      |?                                       |81M                                     |
|timm-u-res2net50_14w_8s                 |?                                       |23M                                     |
|timm-u-res2net50_26w_4s                 |?                                       |23M                                     |
|timm-u-res2net50_26w_6s                 |?                                       |35M                                     |
|timm-u-res2net50_26w_8s                 |?                                       |46M                                     |
|timm-u-res2net50_48w_2s                 |?                                       |23M                                     |
|timm-u-res2net101_26w_4s                |?                                       |43M                                     |
|timm-u-res2next50                       |?                                       |22M                                     |
|timm-u-resnest14d                       |?                                       |8M                                      |
|timm-u-resnest26d                       |?                                       |15M                                     |
|timm-u-resnest50d                       |?                                       |25M                                     |
|timm-u-resnest50d_1s4x24d               |?                                       |23M                                     |
|timm-u-resnest50d_4s2x40d               |?                                       |28M                                     |
|timm-u-resnest101e                      |?                                       |46M                                     |
|timm-u-resnest200e                      |?                                       |68M                                     |
|timm-u-resnest269e                      |?                                       |108M                                    |
|timm-u-resnet18                         |?                                       |11M                                     |
|timm-u-resnet18d                        |?                                       |11M                                     |
|timm-u-resnet26                         |?                                       |13M                                     |
|timm-u-resnet26d                        |?                                       |13M                                     |
|timm-u-resnet34                         |?                                       |21M                                     |
|timm-u-resnet34d                        |?                                       |21M                                     |
|timm-u-resnet50                         |?                                       |23M                                     |
|timm-u-resnet50d                        |?                                       |23M                                     |
|timm-u-resnet101                        |?                                       |42M                                     |
|timm-u-resnet101d                       |?                                       |42M                                     |
|timm-u-resnet152                        |?                                       |58M                                     |
|timm-u-resnet152d                       |?                                       |58M                                     |
|timm-u-resnet200                        |?                                       |62M                                     |
|timm-u-resnet200d                       |?                                       |62M                                     |
|timm-u-resnetblur18                     |?                                       |11M                                     |
|timm-u-resnetblur50                     |?                                       |23M                                     |
|timm-u-resnetv2_50x1_bitm               |?                                       |23M                                     |
|timm-u-resnetv2_50x1_bitm_in21k         |?                                       |23M                                     |
|timm-u-resnetv2_50x3_bitm               |?                                       |211M                                    |
|timm-u-resnetv2_50x3_bitm_in21k         |?                                       |211M                                    |
|timm-u-resnetv2_101x1_bitm              |?                                       |42M                                     |
|timm-u-resnetv2_101x1_bitm_in21k        |?                                       |42M                                     |
|timm-u-resnetv2_101x3_bitm              |?                                       |381M                                    |
|timm-u-resnetv2_101x3_bitm_in21k        |?                                       |381M                                    |
|timm-u-resnetv2_152x2_bitm              |?                                       |232M                                    |
|timm-u-resnetv2_152x2_bitm_in21k        |?                                       |232M                                    |
|timm-u-resnetv2_152x4_bitm              |?                                       |928M                                    |
|timm-u-resnext50d_32x4d                 |?                                       |22M                                     |
|timm-u-resnext101_32x4d                 |?                                       |42M                                     |
|timm-u-resnext101_32x8d                 |?                                       |86M                                     |
|timm-u-resnext101_64x4d                 |?                                       |81M                                     |
|timm-u-rexnet_100                       |?                                       |3.2M                                    |
|timm-u-rexnet_130                       |?                                       |5.4M                                    |
|timm-u-rexnet_150                       |?                                       |7M                                      |
|timm-u-rexnet_200                       |?                                       |12M                                     |
|timm-u-rexnetr_100                      |?                                       |3.3M                                    |
|timm-u-rexnetr_130                      |?                                       |5.5M                                    |
|timm-u-rexnetr_150                      |?                                       |7M                                      |
|timm-u-rexnetr_200                      |?                                       |13M                                     |
|timm-u-selecsls42                       |?                                       |18M                                     |
|timm-u-selecsls42b                      |?                                       |18M                                     |
|timm-u-selecsls60                       |?                                       |18M                                     |
|timm-u-selecsls60b                      |?                                       |18M                                     |
|timm-u-selecsls84                       |?                                       |28M                                     |
|timm-u-semnasnet_050                    |?                                       |0.5M                                    |
|timm-u-semnasnet_075                    |?                                       |1.3M                                    |
|timm-u-semnasnet_100                    |?                                       |2.1M                                    |
|timm-u-semnasnet_140                    |?                                       |4.2M                                    |
|timm-u-senet154                         |?                                       |113M                                    |
|timm-u-seresnet18                       |?                                       |11M                                     |
|timm-u-seresnet34                       |?                                       |21M                                     |
|timm-u-seresnet50                       |?                                       |26M                                     |
|timm-u-seresnet50t                      |?                                       |26M                                     |
|timm-u-seresnet101                      |?                                       |47M                                     |
|timm-u-seresnet152                      |?                                       |64M                                     |
|timm-u-seresnet152d                     |?                                       |64M                                     |
|timm-u-seresnet200d                     |?                                       |69M                                     |
|timm-u-seresnet269d                     |?                                       |111M                                    |
|timm-u-seresnext26d_32x4d               |?                                       |14M                                     |
|timm-u-seresnext26t_32x4d               |?                                       |14M                                     |
|timm-u-seresnext26tn_32x4d              |?                                       |14M                                     |
|timm-u-seresnext50_32x4d                |?                                       |25M                                     |
|timm-u-seresnext101_32x4d               |?                                       |46M                                     |
|timm-u-seresnext101_32x8d               |?                                       |91M                                     |
|timm-u-skresnet18                       |?                                       |11M                                     |
|timm-u-skresnet34                       |?                                       |21M                                     |
|timm-u-skresnet50                       |?                                       |23M                                     |
|timm-u-skresnet50d                      |?                                       |23M                                     |
|timm-u-skresnext50_32x4d                |?                                       |25M                                     |
|timm-u-spnasnet_100                     |?                                       |2.7M                                    |
|timm-u-ssl_resnet18                     |?                                       |11M                                     |
|timm-u-ssl_resnet50                     |?                                       |23M                                     |
|timm-u-ssl_resnext50_32x4d              |?                                       |22M                                     |
|timm-u-ssl_resnext101_32x4d             |?                                       |42M                                     |
|timm-u-ssl_resnext101_32x8d             |?                                       |86M                                     |
|timm-u-ssl_resnext101_32x16d            |?                                       |191M                                    |
|timm-u-swsl_resnet18                    |?                                       |11M                                     |
|timm-u-swsl_resnet50                    |?                                       |23M                                     |
|timm-u-swsl_resnext50_32x4d             |?                                       |22M                                     |
|timm-u-swsl_resnext101_32x4d            |?                                       |42M                                     |
|timm-u-swsl_resnext101_32x8d            |?                                       |86M                                     |
|timm-u-swsl_resnext101_32x16d           |?                                       |191M                                    |
|timm-u-tf_efficientnet_b0               |?                                       |3.5M                                    |
|timm-u-tf_efficientnet_b0_ap            |?                                       |3.5M                                    |
|timm-u-tf_efficientnet_b0_ns            |?                                       |3.5M                                    |
|timm-u-tf_efficientnet_b1               |?                                       |6M                                      |
|timm-u-tf_efficientnet_b1_ap            |?                                       |6M                                      |
|timm-u-tf_efficientnet_b1_ns            |?                                       |6M                                      |
|timm-u-tf_efficientnet_b2               |?                                       |7M                                      |
|timm-u-tf_efficientnet_b2_ap            |?                                       |7M                                      |
|timm-u-tf_efficientnet_b2_ns            |?                                       |7M                                      |
|timm-u-tf_efficientnet_b3               |?                                       |10M                                     |
|timm-u-tf_efficientnet_b3_ap            |?                                       |10M                                     |
|timm-u-tf_efficientnet_b3_ns            |?                                       |10M                                     |
|timm-u-tf_efficientnet_b4               |?                                       |16M                                     |
|timm-u-tf_efficientnet_b4_ap            |?                                       |16M                                     |
|timm-u-tf_efficientnet_b4_ns            |?                                       |16M                                     |
|timm-u-tf_efficientnet_b5               |?                                       |27M                                     |
|timm-u-tf_efficientnet_b5_ap            |?                                       |27M                                     |
|timm-u-tf_efficientnet_b5_ns            |?                                       |27M                                     |
|timm-u-tf_efficientnet_b6               |?                                       |39M                                     |
|timm-u-tf_efficientnet_b6_ap            |?                                       |39M                                     |
|timm-u-tf_efficientnet_b6_ns            |?                                       |39M                                     |
|timm-u-tf_efficientnet_b7               |?                                       |62M                                     |
|timm-u-tf_efficientnet_b7_ap            |?                                       |62M                                     |
|timm-u-tf_efficientnet_b7_ns            |?                                       |62M                                     |
|timm-u-tf_efficientnet_b8               |?                                       |82M                                     |
|timm-u-tf_efficientnet_b8_ap            |?                                       |82M                                     |
|timm-u-tf_efficientnet_cc_b0_4e         |?                                       |11M                                     |
|timm-u-tf_efficientnet_cc_b0_8e         |?                                       |22M                                     |
|timm-u-tf_efficientnet_cc_b1_8e         |?                                       |38M                                     |
|timm-u-tf_efficientnet_el               |?                                       |8M                                      |
|timm-u-tf_efficientnet_em               |?                                       |5.3M                                    |
|timm-u-tf_efficientnet_es               |?                                       |3.9M                                    |
|timm-u-tf_efficientnet_l2_ns            |?                                       |467M                                    |
|timm-u-tf_efficientnet_l2_ns_475        |?                                       |467M                                    |
|timm-u-tf_efficientnet_lite0            |?                                       |2.9M                                    |
|timm-u-tf_efficientnet_lite1            |?                                       |3.7M                                    |
|timm-u-tf_efficientnet_lite2            |?                                       |4.3M                                    |
|timm-u-tf_efficientnet_lite3            |?                                       |6M                                      |
|timm-u-tf_efficientnet_lite4            |?                                       |11M                                     |
|timm-u-tf_inception_v3                  |?                                       |21M                                     |
|timm-u-tf_mixnet_l                      |?                                       |5.3M                                    |
|timm-u-tf_mixnet_m                      |?                                       |3.1M                                    |
|timm-u-tf_mixnet_s                      |?                                       |2.2M                                    |
|timm-u-tf_mobilenetv3_large_075         |?                                       |1.7M                                    |
|timm-u-tf_mobilenetv3_large_100         |?                                       |2.9M                                    |
|timm-u-tf_mobilenetv3_large_minimal_100 |?                                       |1.4M                                    |
|timm-u-tf_mobilenetv3_small_075         |?                                       |0.5M                                    |
|timm-u-tf_mobilenetv3_small_100         |?                                       |0.9M                                    |
|timm-u-tf_mobilenetv3_small_minimal_100 |?                                       |0.4M                                    |
|timm-u-tresnet_l                        |?                                       |?                                       |
|timm-u-tresnet_l_448                    |?                                       |?                                       |
|timm-u-tresnet_m                        |?                                       |?                                       |
|timm-u-tresnet_m_448                    |?                                       |?                                       |
|timm-u-tresnet_xl                       |?                                       |?                                       |
|timm-u-tresnet_xl_448                   |?                                       |?                                       |
|timm-u-tv_densenet121                   |?                                       |6M                                      |
|timm-u-tv_resnet34                      |?                                       |21M                                     |
|timm-u-tv_resnet50                      |?                                       |23M                                     |
|timm-u-tv_resnet101                     |?                                       |42M                                     |
|timm-u-tv_resnet152                     |?                                       |58M                                     |
|timm-u-tv_resnext50_32x4d               |?                                       |22M                                     |
|timm-u-vgg11                            |?                                       |9M                                      |
|timm-u-vgg11_bn                         |?                                       |9M                                      |
|timm-u-vgg13                            |?                                       |9M                                      |
|timm-u-vgg13_bn                         |?                                       |9M                                      |
|timm-u-vgg16                            |?                                       |14M                                     |
|timm-u-vgg16_bn                         |?                                       |14M                                     |
|timm-u-vgg19                            |?                                       |20M                                     |
|timm-u-vgg19_bn                         |?                                       |20M                                     |
|timm-u-vit_base_patch16_224             |?                                       |?                                       |
|timm-u-vit_base_patch16_224_in21k       |?                                       |?                                       |
|timm-u-vit_base_patch16_384             |?                                       |?                                       |
|timm-u-vit_base_patch32_224             |?                                       |?                                       |
|timm-u-vit_base_patch32_224_in21k       |?                                       |?                                       |
|timm-u-vit_base_patch32_384             |?                                       |?                                       |
|timm-u-vit_base_resnet26d_224           |?                                       |?                                       |
|timm-u-vit_base_resnet50_224_in21k      |?                                       |?                                       |
|timm-u-vit_base_resnet50_384            |?                                       |?                                       |
|timm-u-vit_base_resnet50d_224           |?                                       |?                                       |
|timm-u-vit_deit_base_distilled_patch16_224|?                                       |?                                       |
|timm-u-vit_deit_base_distilled_patch16_384|?                                       |?                                       |
|timm-u-vit_deit_base_patch16_224        |?                                       |?                                       |
|timm-u-vit_deit_base_patch16_384        |?                                       |?                                       |
|timm-u-vit_deit_small_distilled_patch16_224|?                                       |?                                       |
|timm-u-vit_deit_small_patch16_224       |?                                       |?                                       |
|timm-u-vit_deit_tiny_distilled_patch16_224|?                                       |?                                       |
|timm-u-vit_deit_tiny_patch16_224        |?                                       |?                                       |
|timm-u-vit_huge_patch14_224_in21k       |?                                       |?                                       |
|timm-u-vit_large_patch16_224            |?                                       |?                                       |
|timm-u-vit_large_patch16_224_in21k      |?                                       |?                                       |
|timm-u-vit_large_patch16_384            |?                                       |?                                       |
|timm-u-vit_large_patch32_224            |?                                       |?                                       |
|timm-u-vit_large_patch32_224_in21k      |?                                       |?                                       |
|timm-u-vit_large_patch32_384            |?                                       |?                                       |
|timm-u-vit_small_patch16_224            |?                                       |?                                       |
|timm-u-vit_small_resnet26d_224          |?                                       |?                                       |
|timm-u-vit_small_resnet50d_s3_224       |?                                       |?                                       |
|timm-u-vovnet39a                        |?                                       |21M                                     |
|timm-u-vovnet57a                        |?                                       |35M                                     |
|timm-u-wide_resnet50_2                  |?                                       |66M                                     |
|timm-u-wide_resnet101_2                 |?                                       |124M                                    |
|timm-u-xception                         |?                                       |20M                                     |
|timm-u-xception41                       |?                                       |24M                                     |
|timm-u-xception65                       |?                                       |37M                                     |
|timm-u-xception71                       |?                                       |40M                                     |

</div>
</details>


\* `ssl`, `swsl` - semi-supervised and weakly-supervised learning on ImageNet ([repo](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models)).


### 🔁 Models API <a name="api"></a>

 - `model.encoder` - pretrained backbone to extract features of different spatial resolution
 - `model.decoder` - depends on models architecture (`Unet`/`Linknet`/`PSPNet`/`FPN`)
 - `model.segmentation_head` - last block to produce required number of mask channels (include also optional upsampling and activation)
 - `model.classification_head` - optional block which create classification head on top of encoder
 - `model.forward(x)` - sequentially pass `x` through model\`s encoder, decoder and segmentation head (and classification head if specified)

##### Input channels
Input channels parameter allows you to create models, which process tensors with arbitrary number of channels.
If you use pretrained weights from imagenet - weights of first convolution will be reused for
1- or 2- channels inputs, for input channels > 4 weights of first convolution will be initialized randomly.
```python
model = smp.FPN('resnet34', in_channels=1)
mask = model(torch.ones([1, 1, 64, 64]))
```

##### Auxiliary classification output  
All models support `aux_params` parameters, which is default set to `None`. 
If `aux_params = None` then classification auxiliary output is not created, else
model produce not only `mask`, but also `label` output with shape `NC`.
Classification head consists of GlobalPooling->Dropout(optional)->Linear->Activation(optional) layers, which can be 
configured by `aux_params` as follows:
```python
aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation='sigmoid',      # activation function, default is None
    classes=4,                 # define number of output labels
)
model = smp.Unet('resnet34', classes=4, aux_params=aux_params)
mask, label = model(x)
```

##### Depth
Depth parameter specify a number of downsampling operations in encoder, so you can make
your model lighter if specify smaller `depth`.
```python
model = smp.Unet('resnet34', encoder_depth=4)
```


### 🛠 Installation <a name="installation"></a>
PyPI version:
```bash
$ pip install segmentation-models-pytorch
````
Latest version from source:
```bash
$ pip install git+https://github.com/qubvel/segmentation_models.pytorch
````

### 🏆 Competitions won with the library

`Segmentation Models` package is widely used in the image segmentation competitions.
[Here](https://github.com/qubvel/segmentation_models.pytorch/blob/master/HALLOFFAME.md) you can find competitions, names of the winners and links to their solutions.

### 🤝 Contributing

##### Run test
```bash
$ docker build -f docker/Dockerfile.dev -t smp:dev . && docker run --rm smp:dev pytest -p no:cacheprovider
```
##### Generate table
```bash
$ docker build -f docker/Dockerfile.dev -t smp:dev . && docker run --rm smp:dev python misc/generate_table.py
```

### 📝 Citing
```
@misc{Yakubovskiy:2019,
  Author = {Pavel Yakubovskiy},
  Title = {Segmentation Models Pytorch},
  Year = {2020},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/qubvel/segmentation_models.pytorch}}
}
```

### 🛡️ License <a name="license"></a>
Project is distributed under [MIT License](https://github.com/qubvel/segmentation_models.pytorch/blob/master/LICENSE)
