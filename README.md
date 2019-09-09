# Segmentation models
[![Build Status](https://travis-ci.com/qubvel/segmentation_models.pytorch.svg?branch=master)](https://travis-ci.com/qubvel/segmentation_models.pytorch) [![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg)](https://shields.io/)

Segmentation models is python library with Neural Networks for Image Segmentation based on PyTorch.

The main features of this library are:

 - High level API (just two lines to create neural network)
 - 4 models architectures for binary and multi class segmentation (including legendary Unet)
 - 30 available encoders for each architecture
 - All encoders have pre-trained weights for faster and better convergence

### Table of content
 1. [Quick start](#start)
 2. [Examples](#examples)
 3. [Models](#models) 
    1. [Architectures](#architectires)
    2. [Encoders](#encoders)
    3. [Pretrained weights](#weights)
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

| Type       | Encoder names                                                                               |
|------------|---------------------------------------------------------------------------------------------|
| VGG        | vgg11, vgg13, vgg16, vgg19, vgg11bn,  vgg13bn, vgg16bn, vgg19bn                             |
| DenseNet   | densenet121, densenet169, densenet201, densenet161                                          |
| DPN        | dpn68, dpn68b, dpn92, dpn98, dpn107, dpn131                                                 |
| Inception  | inceptionresnetv2                                                                           |
| ResNet     | resnet18, resnet34, resnet50, resnet101, resnet152                                          |
| ResNeXt    | resnext50_32x4d, resnext101_32x8d, resnext101_32x16d, resnext101_32x32d, resnext101_32x48d  |
| SE-ResNet  | se_resnet50, se_resnet101, se_resnet152                                                     |
| SE-ResNeXt | se_resnext50_32x4d,  se_resnext101_32x4d                                                    |
| SENet      | senet154                                                                                    |  

#### Weights <a name="weights"></a>

| Weights name                                                              | Encoder names                                                                                                                                                                                                                                                                                                                                                                       |
|---------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| imagenet+5k                                                               | dpn68b, dpn92, dpn107                                                                                                                                                                                                                                                                                                                                                               |
| imagenet                                                                  | vgg11, vgg13, vgg16, vgg19, vgg11bn,  vgg13bn, vgg16bn, vgg19bn, <br> densenet121, densenet169, densenet201, densenet161, dpn68, dpn98, dpn131, <br> inceptionresnetv2, <br> resnet18, resnet34, resnet50, resnet101, resnet152, <br> resnext50_32x4d, resnext101_32x8d, <br> se_resnet50, se_resnet101, se_resnet152, <br> se_resnext50_32x4d,  se_resnext101_32x4d, <br> senet154 |
| [instagram](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/) | resnext101_32x8d, resnext101_32x16d, resnext101_32x32d, resnext101_32x48d                                                                                                                                                                                                                                                                                                           |

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

### Run tests
```bash
$ docker build -f docker/Dockerfile.dev -t smp:dev .
$ docker run --rm smp:dev pytest -p no:cacheprovider
```
