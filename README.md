# Segmentation models

Segmentation models is python library with Neural Networks for Image Segmentation based on PyTorch.

The main features of this library are:

 - High level API (just two lines to create NN)
 - 4 models architectures for binary and multi class segmentation (including legendary Unet)
 - 30 available encoders for each architecture
 - All backbones have pre-trained weights for faster and better convergence

### Table of content
 1. [Quick start](#start)
 2. [Models](#models) 
    1. [Architectires](#architectires)
    2. [Encoders](#encoders)
    3. [Pretrained weights](#weights)
 3. [Installation](#installation)
 4. [License](#license)

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

As all models have pretrained weights, so you have to prepare your data the same way as during weights pretraining:
```python
from segmentation_models_pytorch.encoders import get_preprocessing_fn

preprocess_input = get_preprocessing_fn('renset18', pretrained='imagenet')
```

### Models <a name="models"></a>

#### Architectires <a name="architectires"></a>
 - [Unet](https://arxiv.org/abs/1505.04597)
 - [Linknet](https://arxiv.org/abs/1707.03718)
 - [FPN](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
 - [PSPNet](https://arxiv.org/abs/1612.01105)
 
#### Encoders <a name="encoders"></a>

| Type       | Encoder names                                                 |
|------------|-----------------------------------------------------------------|
| VGG        | vgg11, vgg13, vgg16, vgg19, vgg11bn,  vgg13bn, vgg16bn, vgg19bn |
| DenseNet   | densenet121, densenet169, densenet201, densenet161              |
| DPN        | dpn68, dpn68b, dpn92, dpn98, dpn107, dpn131                     |
| Inception  | inceptionresnetv2                                               |
| ResNet     | resnet18, resnet34, resnet50, resnet101, resnet152              |
| SE-ResNet  | se_resnet50, se_resnet101, se_resnet152                         |
| SE-ResNeXt | se_resnext50_32x4d,  se_resnext101_32x4d                        |
| SENet      | senet154                                                        |                                                           |

#### Weights <a name="weights"></a>

| Weights name | Encoder names         |
|--------------|-----------------------|
| imagenet+5k  | dpn68b, dpn92, dpn107 |
| imagenet     | * all other encoders  |


### Installation <a name="installation"></a>
PyPI version is not released yet, to install package from source run following command:
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
