<div align="center">
 
![logo](https://i.ibb.co/dc1XdhT/Segmentation-Models-V2-Side-1-1.png)  
**Python library with Neural Networks for Image  
Segmentation based on [PyTorch](https://pytorch.org/).**  

![PyPI version](https://badge.fury.io/py/segmentation-models-pytorch.svg) [![Build Status](https://travis-ci.com/qubvel/segmentation_models.pytorch.svg?branch=master)](https://travis-ci.com/qubvel/segmentation_models.pytorch) [![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg)](https://shields.io/)

</div>

The main features of this library are:

 - High level API (just two lines to create neural network)
 - 5 models architectures for binary and multi class segmentation (including legendary Unet)
 - 46 available encoders for each architecture
 - All encoders have pre-trained weights for faster and better convergence

### Table of content
 1. [Quick start](#start)
 2. [Examples](#examples)
 3. [Models](#models)
    1. [Architectures](#architectires)
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

### Quick start <a name="start"></a>
Since the library is built on the PyTorch framework, created segmentation model is just a PyTorch nn.Module, which can be created as easy as:
```python
import segmentation_models_pytorch as smp

model = smp.Unet()
```
Depending on the task, you can change the network architecture by choosing backbones with fewer or more parameters and use pretrainded weights to initialize it:

```python
model = smp.Unet('resnet34', encoder_weights='imagenet')
```

Change number of output classes in the model:

```python
model = smp.Unet('resnet34', classes=3, activation='softmax')
```

All models have pretrained encoders, so you have to prepare your data the same way as during weights pretraining:
```python
from segmentation_models_pytorch.encoders import get_preprocessing_fn

preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')
```
### Examples <a name="examples"></a>
 - Training model for cars segmentation on CamVid dataset [here](https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb).
 - Training SMP model with [Catalyst](https://github.com/catalyst-team/catalyst) (high-level framework for PyTorch), [Ttach](https://github.com/qubvel/ttach) (TTA library for PyTorch) and [Albumentations](https://github.com/albu/albumentations) (fast image augmentation library) - [here](https://github.com/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb)

### Models <a name="models"></a>

#### Architectures <a name="architectires"></a>
 - [Unet](https://arxiv.org/abs/1505.04597)
 - [Linknet](https://arxiv.org/abs/1707.03718)
 - [FPN](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
 - [PSPNet](https://arxiv.org/abs/1612.01105)
 - [PAN](https://arxiv.org/abs/1805.10180)
 - [DeepLabV3](https://arxiv.org/abs/1706.05587)

#### Encoders <a name="encoders"></a>

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|resnet18                        |imagenet<br>ssl*<br>swsl*        |11M                             |
|resnet34                        |imagenet                        |21M                             |
|resnet50                        |imagenet<br>ssl*<br>swsl*        |23M                             |
|resnet101                       |imagenet                        |42M                             |
|resnet152                       |imagenet                        |58M                             |
|resnext50_32x4d                 |imagenet<br>ssl*<br>swsl*        |22M                             |
|resnext101_32x4d                |ssl<br>swsl                     |42M                             |
|resnext101_32x8d                |imagenet<br>instagram<br>ssl*<br>swsl*|86M                         |
|resnext101_32x16d               |instagram<br>ssl*<br>swsl*        |191M                            |
|resnext101_32x32d               |instagram                       |466M                            |
|resnext101_32x48d               |instagram                       |826M                            |
|dpn68                           |imagenet                        |11M                             |
|dpn68b                          |imagenet+5k                     |11M                             |
|dpn92                           |imagenet+5k                     |34M                             |
|dpn98                           |imagenet                        |58M                             |
|dpn107                          |imagenet+5k                     |84M                             |
|dpn131                          |imagenet                        |76M                             |
|vgg11                           |imagenet                        |9M                              |
|vgg11_bn                        |imagenet                        |9M                              |
|vgg13                           |imagenet                        |9M                              |
|vgg13_bn                        |imagenet                        |9M                              |
|vgg16                           |imagenet                        |14M                             |
|vgg16_bn                        |imagenet                        |14M                             |
|vgg19                           |imagenet                        |20M                             |
|vgg19_bn                        |imagenet                        |20M                             |
|senet154                        |imagenet                        |113M                            |
|se_resnet50                     |imagenet                        |26M                             |
|se_resnet101                    |imagenet                        |47M                             |
|se_resnet152                    |imagenet                        |64M                             |
|se_resnext50_32x4d              |imagenet                        |25M                             |
|se_resnext101_32x4d             |imagenet                        |46M                             |
|densenet121                     |imagenet                        |6M                              |
|densenet169                     |imagenet                        |12M                             |
|densenet201                     |imagenet                        |18M                             |
|densenet161                     |imagenet                        |26M                             |
|inceptionresnetv2               |imagenet<br>imagenet+background |54M                             |
|inceptionv4                     |imagenet<br>imagenet+background |41M                             |
|efficientnet-b0                 |imagenet                        |4M                              |
|efficientnet-b1                 |imagenet                        |6M                              |
|efficientnet-b2                 |imagenet                        |7M                              |
|efficientnet-b3                 |imagenet                        |10M                             |
|efficientnet-b4                 |imagenet                        |17M                             |
|efficientnet-b5                 |imagenet                        |28M                             |
|efficientnet-b6                 |imagenet                        |40M                             |
|efficientnet-b7                 |imagenet                        |63M                             |
|mobilenet_v2                    |imagenet                        |2M                              |
|xception                        |imagenet                        |22M                             |
|timm-efficientnet-b0            |imagenet<br>advprop<br>noisy-student|4M                              |
|timm-efficientnet-b1            |imagenet<br>advprop<br>noisy-student|6M                              |
|timm-efficientnet-b2            |imagenet<br>advprop<br>noisy-student|7M                              |
|timm-efficientnet-b3            |imagenet<br>advprop<br>noisy-student|10M                             |
|timm-efficientnet-b4            |imagenet<br>advprop<br>noisy-student|17M                             |
|timm-efficientnet-b5            |imagenet<br>advprop<br>noisy-student|28M                             |
|timm-efficientnet-b6            |imagenet<br>advprop<br>noisy-student|40M                             |
|timm-efficientnet-b7            |imagenet<br>advprop<br>noisy-student|63M                             |
|timm-efficientnet-b8            |imagenet<br>advprop             |84M                             |
|timm-efficientnet-l2            |noisy-student                   |474M                            |

\* `ssl`, `wsl` from [here](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models).

### Models API <a name="api"></a>

 - `model.encoder` - pretrained backbone to extract features of different spatial resolution
 - `model.decoder` - depends on models architecture (`Unet`/`Linknet`/`PSPNet`/`FPN`)
 - `model.segmentation_head` - last block to produce required number of mask channels (include also optional upsampling and activation)
 - `model.classification_head` - optional block which create classification head on top of encoder
 - `model.forward(x)` - sequentially pass `x` through model\`s encoder, decoder and segmentation head (and classification head if specified)

##### Input channels
Input channels parameter allow you to create models, which process tensors with arbitrary number of channels.
If you use pretrained weights from imagenet - weights of first convolution will be reused for
1- or 2- channels inputs, for input channels > 4 weights of first convolution will be initialized randomly.
```python
model = smp.FPN('resnet34', in_channels=1)
mask = model(torch.ones([1, 1, 64, 64]))
```

##### Auxiliary classification output  
All models support `aux_params` parameters, which is default set to `None`. 
If `aux_params = None` than classification auxiliary output is not created, else
model produce not only `mask`, but also `label` output with shape `NC`.
Classification head consist of GlobalPooling->Dropout(optional)->Linear->Activation(optional) layers, which can be 
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
your model lighted if specify smaller `depth`.
```python
model = smp.Unet('resnet34', encoder_depth=4)
```


### Installation <a name="installation"></a>
PyPI version:
```bash
$ pip install segmentation-models-pytorch
````
Latest version from source:
```bash
$ pip install git+https://github.com/qubvel/segmentation_models.pytorch
````

### Competitions won with the library

`Segmentation Models` package is widely used in the image segmentation competitions.
[Here](https://github.com/qubvel/segmentation_models.pytorch/blob/master/HALLOFFAME.md) you can find competitions, names of the winners and links to their solutions.

### Contributing

##### Run test
```bash
$ docker build -f docker/Dockerfile.dev -t smp:dev . && docker run --rm smp:dev pytest -p no:cacheprovider
```
##### Generate table
```bash
$ docker build -f docker/Dockerfile.dev -t smp:dev . && docker run --rm smp:dev python misc/generate_table.py
```

### Citing
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

### License <a name="license"></a>
Project is distributed under [MIT License](https://github.com/qubvel/segmentation_models.pytorch/blob/master/LICENSE)
