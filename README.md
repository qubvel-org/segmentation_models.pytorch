# Segmentation models
[![Build Status](https://travis-ci.com/qubvel/segmentation_models.pytorch.svg?branch=master)](https://travis-ci.com/qubvel/segmentation_models.pytorch) [![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg)](https://shields.io/)

Segmentation models is python library with Neural Networks for Image Segmentation based on PyTorch.

The main features of this library are:

 - High level API (just two lines to create neural network)
 - 4 models architectures for binary and multi class segmentation (including legendary Unet)
 - 46 available encoders for each architecture
 - All encoders have pre-trained weights for faster and better convergence

### Table of content
 1. [Quick start](#start)
 2. [Examples](#examples)
 3. [Models](#models)
    1. [Architectures](#architectires)
    2. [Encoders](#encoders)
 4. [Models API](#api)
 5. [Installation](#installation)
 6. [License](#license)

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
 - Training model with [Catalyst](https://github.com/catalyst-team/catalyst) (high-level framework for PyTorch) - [here](https://colab.research.google.com/gist/Scitator/e3fd90eec05162e16b476de832500576/cars-segmentation-camvid.ipynb).

### Models <a name="models"></a>

#### Architectures <a name="architectires"></a>
 - [Unet](https://arxiv.org/abs/1505.04597)
 - [Linknet](https://arxiv.org/abs/1707.03718)
 - [FPN](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
 - [PSPNet](https://arxiv.org/abs/1612.01105)

#### Encoders <a name="encoders"></a>

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|resnet18                        |imagenet                        |11M                             |
|resnet34                        |imagenet                        |21M                             |
|resnet50                        |imagenet                        |23M                             |
|resnet101                       |imagenet                        |42M                             |
|resnet152                       |imagenet                        |58M                             |
|resnext50_32x4d                 |imagenet                        |22M                             |
|resnext101_32x8d                |imagenet<br>instagram           |86M                             |
|resnext101_32x16d               |instagram                       |191M                            |
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
|xception                    |imagenet                        |22M                              |

### Models API <a name="api"></a>
 - `model.encoder` - pretrained backbone to extract features of different spatial resolution
 - `model.decoder` - segmentation head, depends on models architecture (`Unet`/`Linknet`/`PSPNet`/`FPN`)
 - `model.activation` - output activation function, one of `sigmoid`, `softmax`
 - `model.forward(x)` - sequentially pass `x` through model\`s encoder and decoder (return logits!)
 - `model.predict(x)` - inference method, switch model to `.eval()` mode, call `.forward(x)` and apply activation function with `torch.no_grad()`

### Installation <a name="installation"></a>
PyPI version:
```bash
$ pip install segmentation-models-pytorch
````
Latest version from source:
```bash
$ pip install git+https://github.com/qubvel/segmentation_models.pytorch
````
### License <a name="license"></a>
Project is distributed under [MIT License](https://github.com/qubvel/segmentation_models.pytorch/blob/master/LICENSE)

### Contributing

##### Run test
```bash
$ docker build -f docker/Dockerfile.dev -t smp:dev . && docker run --rm smp:dev pytest -p no:cacheprovider
```
##### Generate table
```bash
$ docker build -f docker/Dockerfile.dev -t smp:dev . && docker run --rm smp:dev python misc/generate_table.py
```
