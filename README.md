<div align="center">
 
![logo](https://i.ibb.co/dc1XdhT/Segmentation-Models-V2-Side-1-1.png)  
**Python library with Neural Networks for Image  
Segmentation based on [PyTorch](https://pytorch.org/).**  

[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/qubvel/segmentation_models.pytorch/blob/master/LICENSE) 
[![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/qubvel/segmentation_models.pytorch/CI/master?style=for-the-badge&logo=github)](https://github.com/qubvel/segmentation_models.pytorch/actions/workflows/tests.yml) 
[![Read the Docs](https://img.shields.io/readthedocs/smp?style=for-the-badge&logo=readthedocs&logoColor=white)](https://smp.readthedocs.io/en/latest/) 
<br>
[![PyPI](https://img.shields.io/pypi/v/segmentation-models-pytorch?color=blue&style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/search/?q=segmentation-models-pytorch) 
[![PyPI - Downloads](https://img.shields.io/pypi/dm/segmentation-models-pytorch?style=for-the-badge&color=blue)](https://pepy.tech/project/segmentation-models-pytorch) 
<br>
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.4+-red?style=for-the-badge&logo=pytorch)](https://pepy.tech/project/segmentation-models-pytorch) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.6+-red?style=for-the-badge&logo=python&logoColor=white)](https://pepy.tech/project/segmentation-models-pytorch) 

</div>

The main features of this library are:

 - High level API (just two lines to create a neural network)
 - 9 models architectures for binary and multi class segmentation (including legendary Unet)
 - 113 available encoders (and 400+ encoders from [timm](https://github.com/rwightman/pytorch-image-models))
 - All encoders have pre-trained weights for faster and better convergence
 - Popular metrics and losses for training routines
 
### [📚 Project Documentation 📚](http://smp.readthedocs.io/)

Visit [Read The Docs Project Page](https://smp.readthedocs.io/) or read following README to know more about Segmentation Models Pytorch (SMP for short) library

### 📋 Table of content
 1. [Quick start](#start)
 2. [Examples](#examples)
 3. [Models](#models)
    1. [Architectures](#architectures)
    2. [Encoders](#encoders)
    3. [Timm Encoders](#timm)
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
 - see [table](#architectures) with available model architectures
 - see [table](#encoders) with available encoders and their corresponding weights

#### 2. Configure data preprocessing

All encoders have pretrained weights. Preparing your data the same way as during weights pre-training may give your better results (higher metric score and faster convergence). It is **not necessary** in case you train the whole model, not only decoder.

```python
from segmentation_models_pytorch.encoders import get_preprocessing_fn

preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')
```

Congratulations! You are done! Now you can train your model with your favorite framework!

### 💡 Examples <a name="examples"></a>
 - Training model for pets binary segmentation with Pytorch-Lightning [notebook](https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb) and [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/examples/binary_segmentation_intro.ipynb)
 - Training model for cars segmentation on CamVid dataset [here](https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb).
 - Training SMP model with [Catalyst](https://github.com/catalyst-team/catalyst) (high-level framework for PyTorch), [TTAch](https://github.com/qubvel/ttach) (TTA library for PyTorch) and [Albumentations](https://github.com/albu/albumentations) (fast image augmentation library) - [here](https://github.com/catalyst-team/catalyst/blob/v21.02rc0/examples/notebooks/segmentation-tutorial.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/v21.02rc0/examples/notebooks/segmentation-tutorial.ipynb)
 - Training SMP model with [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io) framework - [here](https://github.com/ternaus/cloths_segmentation) (clothes binary segmentation by [@ternaus](https://github.com/ternaus)).

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
<summary style="margin-left: 25px;">GERNet</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|timm-gernet_s                   |imagenet                        |6M                              |
|timm-gernet_m                   |imagenet                        |18M                             |
|timm-gernet_l                   |imagenet                        |28M                             |

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
|timm-mobilenetv3_large_075      |imagenet                        |1.78M                       |
|timm-mobilenetv3_large_100      |imagenet                        |2.97M                       |
|timm-mobilenetv3_large_minimal_100|imagenet                        |1.41M                       |
|timm-mobilenetv3_small_075      |imagenet                        |0.57M                        |
|timm-mobilenetv3_small_100      |imagenet                        |0.93M                       |
|timm-mobilenetv3_small_minimal_100|imagenet                        |0.43M                       |

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


\* `ssl`, `swsl` - semi-supervised and weakly-supervised learning on ImageNet ([repo](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models)).

#### Timm Encoders <a name="timm"></a>

[docs](https://smp.readthedocs.io/en/latest/encoders_timm.html)

Pytorch Image Models (a.k.a. timm) has a lot of pretrained models and interface which allows using these models as encoders in smp, however, not all models are supported

 - transformer models do not have ``features_only`` functionality implemented
 - some models do not have appropriate strides

Total number of supported encoders: 467
 - [table with available encoders](https://smp.readthedocs.io/en/latest/encoders_timm.html)

### 🔁 Models API <a name="api"></a>

 - `model.encoder` - pretrained backbone to extract features of different spatial resolution
 - `model.decoder` - depends on models architecture (`Unet`/`Linknet`/`PSPNet`/`FPN`)
 - `model.segmentation_head` - last block to produce required number of mask channels (include also optional upsampling and activation)
 - `model.classification_head` - optional block which create classification head on top of encoder
 - `model.forward(x)` - sequentially pass `x` through model\`s encoder, decoder and segmentation head (and classification head if specified)

##### Input channels
Input channels parameter allows you to create models, which process tensors with arbitrary number of channels.
If you use pretrained weights from imagenet - weights of first convolution will be reused. For
1-channel case it would be a sum of weights of first convolution layer, otherwise channels would be 
populated with weights like `new_weight[:, i] = pretrained_weight[:, i % 3]` and than scaled with `new_weight * 3 / new_in_channels`.
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

##### Install linting and formatting pre-commit hooks
```bash
pip install pre-commit black flake8
pre-commit install
```

##### Run tests
```bash
pytest -p no:cacheprovider
```

##### Run tests in docker
```bash
$ docker build -f docker/Dockerfile.dev -t smp:dev . && docker run --rm smp:dev pytest -p no:cacheprovider
```

##### Generate table with encoders (in case you add a new encoder)
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
