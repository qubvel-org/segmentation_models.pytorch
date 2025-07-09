<div align="center">
 
![logo](https://i.ibb.co/dc1XdhT/Segmentation-Models-V2-Side-1-1.png)  
**Python library with Neural Networks for Image Semantic  
Segmentation based on [PyTorch](https://pytorch.org/).**  

 
[![GitHub Workflow Status (branch)](https://img.shields.io/github/actions/workflow/status/qubvel/segmentation_models.pytorch/tests.yml?branch=main&style=for-the-badge)](https://github.com/qubvel/segmentation_models.pytorch/actions/workflows/tests.yml) 
![Codecov](https://img.shields.io/codecov/c/github/qubvel-org/segmentation_models.pytorch?style=for-the-badge)
[![Read the Docs](https://img.shields.io/readthedocs/smp?style=for-the-badge&logo=readthedocs&logoColor=white)](https://smp.readthedocs.io/en/latest/) 
<br>
[![PyPI](https://img.shields.io/pypi/v/segmentation-models-pytorch?color=red&style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/segmentation-models-pytorch/) 
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.9+-red?style=for-the-badge&logo=pytorch)](https://pepy.tech/project/segmentation-models-pytorch) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.9+-red?style=for-the-badge&logo=python&logoColor=white)](https://pepy.tech/project/segmentation-models-pytorch) 
<br>
[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge&color=blue)](https://github.com/qubvel/segmentation_models.pytorch/blob/main/LICENSE)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/segmentation-models-pytorch?style=for-the-badge&color=blue)](https://pepy.tech/project/segmentation-models-pytorch) 

</div>

The main features of the library are:

 - Super simple high-level API (just two lines to create a neural network)
 - 12 encoder-decoder model architectures (Unet, Unet++, Segformer, DPT, ...)
 - 800+ **pretrained** convolution- and transform-based encoders, including [timm](https://github.com/huggingface/pytorch-image-models) support
 - Popular metrics and losses for training routines (Dice, Jaccard, Tversky, ...)
 - ONNX export and torch script/trace/compile friendly

### Community-Driven Project, Supported By
<table>
  <tr>
    <td align="center" vertical-align="center">
      <a href="https://withoutbg.com/?utm_source=smp&utm_medium=github_readme&utm_campaign=sponsorship" >
        <img src="https://withoutbg.com/images/logo-social.png" width="70px;" alt="withoutBG API Logo" />
      </a>
    </td>
    <td align="center" vertical-align="center">
      <b>withoutBG API</b>
      <br />
      <a href="https://withoutbg.com/?utm_source=smp&utm_medium=github_readme&utm_campaign=sponsorship">https://withoutbg.com</a>
      <br />
      <p width="200px">
      High-quality background removal API
        <br/>
      </p>
    </td>
  </tr>
</table>
 
### [📚 Project Documentation 📚](http://smp.readthedocs.io/)

Visit [Read The Docs Project Page](https://smp.readthedocs.io/) or read the following README to know more about Segmentation Models Pytorch (SMP for short) library

### 📋 Table of content
 1. [Quick start](#start)
 2. [Examples](#examples)
 3. [Models and encoders](#models-and-encoders)
 4. [Models API](#api)
    1. [Input channels](#input-channels)
    2. [Auxiliary classification output](#auxiliary-classification-output)
    3. [Depth](#depth)
 5. [Installation](#installation)
 6. [Competitions won with the library](#competitions)
 7. [Contributing](#contributing)
 8. [Citing](#citing)
 9. [License](#license)

## ⏳ Quick start <a name="start"></a>

#### 1. Create your first Segmentation model with SMP

The segmentation model is just a PyTorch `torch.nn.Module`, which can be created as easy as:

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

All encoders have pretrained weights. Preparing your data the same way as during weights pre-training may give you better results (higher metric score and faster convergence). It is **not necessary** in case you train the whole model, not only the decoder.

```python
from segmentation_models_pytorch.encoders import get_preprocessing_fn

preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')
```

Congratulations! You are done! Now you can train your model with your favorite framework!

## 💡 Examples <a name="examples"></a>

| Name                                      | Link                                                                                           | Colab                                                                                           |
|-------------------------------------------|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| **Train** pets binary segmentation on OxfordPets     | [Notebook](https://github.com/qubvel/segmentation_models.pytorch/blob/main/examples/binary_segmentation_intro.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/main/examples/binary_segmentation_intro.ipynb) |
| **Train** cars binary segmentation on CamVid       | [Notebook](https://github.com/qubvel/segmentation_models.pytorch/blob/main/examples/cars%20segmentation%20(camvid).ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/main/examples/cars%20segmentation%20(camvid).ipynb) |
| **Train** multiclass segmentation on CamVid          | [Notebook](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/examples/camvid_segmentation_multiclass.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qubvel-org/segmentation_models.pytorch/blob/main/examples/camvid_segmentation_multiclass.ipynb) |
| **Train** clothes binary segmentation by @ternaus   | [Repo](https://github.com/ternaus/cloths_segmentation)                                        |     |
| **Load and inference** pretrained Segformer | [Notebook](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/examples/segformer_inference_pretrained.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/main/examples/segformer_inference_pretrained.ipynb) |
| **Load and inference** pretrained DPT | [Notebook](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/examples/dpt_inference_pretrained.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/main/examples/dpt_inference_pretrained.ipynb) |
| **Load and inference** pretrained UPerNet | [Notebook](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/examples/upernet_inference_pretrained.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/main/examples/upernet_inference_pretrained.ipynb) |
| **Save and load** models locally / to HuggingFace Hub |[Notebook](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/examples/save_load_model_and_share_with_hf_hub.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/main/examples/save_load_model_and_share_with_hf_hub.ipynb)
| **Export** trained model to ONNX              | [Notebook](https://github.com/qubvel/segmentation_models.pytorch/blob/main/examples/convert_to_onnx.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/main/examples/convert_to_onnx.ipynb) |


## 📦 Models and encoders <a name="models-and-encoders"></a>

### Architectures <a name="architectures"></a>
| Architecture | Paper | Documentation | Checkpoints |
|--------------|-------|---------------|------------|
| Unet | [paper](https://arxiv.org/abs/1505.04597) | [docs](https://smp.readthedocs.io/en/latest/models.html#unet) | |
| Unet++ | [paper](https://arxiv.org/pdf/1807.10165.pdf) | [docs](https://smp.readthedocs.io/en/latest/models.html#unetplusplus) | |
| MAnet | [paper](https://ieeexplore.ieee.org/abstract/document/9201310) | [docs](https://smp.readthedocs.io/en/latest/models.html#manet) | |
| Linknet | [paper](https://arxiv.org/abs/1707.03718) | [docs](https://smp.readthedocs.io/en/latest/models.html#linknet) | |
| FPN | [paper](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf) | [docs](https://smp.readthedocs.io/en/latest/models.html#fpn) | |
| PSPNet | [paper](https://arxiv.org/abs/1612.01105) | [docs](https://smp.readthedocs.io/en/latest/models.html#pspnet) | |
| PAN | [paper](https://arxiv.org/abs/1805.10180) | [docs](https://smp.readthedocs.io/en/latest/models.html#pan) | |
| DeepLabV3 | [paper](https://arxiv.org/abs/1706.05587) | [docs](https://smp.readthedocs.io/en/latest/models.html#deeplabv3) | |
| DeepLabV3+ | [paper](https://arxiv.org/abs/1802.02611) | [docs](https://smp.readthedocs.io/en/latest/models.html#deeplabv3plus) | |
| UPerNet | [paper](https://arxiv.org/abs/1807.10221) | [docs](https://smp.readthedocs.io/en/latest/models.html#upernet) | [checkpoints](https://huggingface.co/collections/smp-hub/upernet-67fadcdbe08418c6ea94f768) |
| Segformer | [paper](https://arxiv.org/abs/2105.15203) | [docs](https://smp.readthedocs.io/en/latest/models.html#segformer) | [checkpoints](https://huggingface.co/collections/smp-hub/segformer-6749eb4923dea2c355f29a1f) |
| DPT | [paper](https://arxiv.org/abs/2103.13413) | [docs](https://smp.readthedocs.io/en/latest/models.html#dpt) | [checkpoints](https://huggingface.co/collections/smp-hub/dpt-67f30487327c0599a0c62d68) |

### Encoders <a name="encoders"></a>

The library provides a wide range of **pretrained** encoders (also known as backbones) for segmentation models. Instead of using features from the final layer of a classification model, we extract **intermediate features** and feed them into the decoder for segmentation tasks.  

All encoders come with **pretrained weights**, which help achieve **faster and more stable convergence** when training segmentation models.  

Given the extensive selection of supported encoders, you can choose the best one for your specific use case, for example:  
- **Lightweight encoders** for low-latency applications or real-time inference on edge devices (mobilenet/mobileone).  
- **High-capacity architectures** for complex tasks involving a large number of segmented classes, providing superior accuracy (convnext/swin/mit).  

By selecting the right encoder, you can balance **efficiency, performance, and model complexity** to suit your project needs.  

All encoders and corresponding pretrained weight are listed in the documentation:
 - [table](https://smp.readthedocs.io/en/latest/encoders.html) with natively ported encoders 
 - [table](https://smp.readthedocs.io/en/latest/encoders_timm.html) with [timm](https://github.com/huggingface/pytorch-image-models) encoders supported

## 🔁 Models API <a name="api"></a>

### Input channels

The input channels parameter allows you to create a model that can process a tensor with an arbitrary number of channels. 
If you use pretrained weights from ImageNet, the weights of the first convolution will be reused:
 - For the 1-channel case, it would be a sum of the weights of the first convolution layer.
 - Otherwise, channels would be populated with weights like `new_weight[:, i] = pretrained_weight[:, i % 3]`, and then scaled with `new_weight * 3 / new_in_channels`.

```python
model = smp.FPN('resnet34', in_channels=1)
mask = model(torch.ones([1, 1, 64, 64]))
```

### Auxiliary classification output

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

### Depth

Depth parameter specify a number of downsampling operations in encoder, so you can make
your model lighter if specify smaller `depth`.
```python
model = smp.Unet('resnet34', encoder_depth=4)
```

## 🛠 Installation <a name="installation"></a>
PyPI version:

```bash
$ pip install segmentation-models-pytorch
````

The latest version from GitHub:

```bash
$ pip install git+https://github.com/qubvel/segmentation_models.pytorch
````

## 🏆 Competitions won with the library <a name="competitions"></a>

`Segmentation Models` package is widely used in image segmentation competitions.
[Here](https://github.com/qubvel/segmentation_models.pytorch/blob/main/HALLOFFAME.md) you can find competitions, names of the winners and links to their solutions.

## 🤝 Contributing <a name="contributing"></a>

1. Install SMP in dev mode

```bash
make install_dev  # Create .venv, install SMP in dev mode
```

2. Run tests and code checks

```bash
make test          # Run tests suite with pytest
make fixup         # Ruff for formatting and lint checks
```

3. Update a table (in case you added an encoder)

```bash
make table        # Generates a table with encoders and print to stdout
```

## 📝 Citing <a name="citing"></a>
```
@misc{Iakubovskii:2019,
  Author = {Pavel Iakubovskii},
  Title = {Segmentation Models Pytorch},
  Year = {2019},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/qubvel/segmentation_models.pytorch}}
}
```

## 🛡️ License <a name="license"></a>
The project is primarily distributed under [MIT License](https://github.com/qubvel/segmentation_models.pytorch/blob/main/LICENSE), while some files are subject to other licenses. Please refer to [LICENSES](licenses/LICENSES.md) and license statements in each file for careful check, especially for commercial use.

<!-- GitAds-Verify: T1452I5Y1X6LC6PFHM9EMDOETGQEU72P -->