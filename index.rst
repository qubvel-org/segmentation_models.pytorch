.. segmentation_models_pytorch documentation master file, created by
   sphinx-quickstart on Fri Jan  3 13:36:52 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to segmentation_models_pytorch's documentation!
=======================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   docs/api
   docs/models

.. container::

   | |logo|
   | Python library with Neural Networks for Image
     Segmentation based on  `PyTorch <https://pytorch.org/>`__ 

   |PyPI version| |Build Status| |Generic badge|

The main features of this library are:

-  High level API (just two lines to create neural network)
-  5 models architectures for binary and multi class segmentation
   (including legendary Unet)
-  46 available encoders for each architecture
-  All encoders have pre-trained weights for faster and better
   convergence

Table of content
~~~~~~~~~~~~~~~~

1. `Quick start <#start>`__
2. `Examples <#examples>`__
3. `Models <#models>`__

   1. `Architectures <#architectires>`__
   2. `Encoders <#encoders>`__

4. `Models API <#api>`__

   1. `Input channels <#input-channels>`__
   2. `Auxiliary classification
      output <#auxiliary-classification-output>`__
   3. `Depth <#depth>`__

5. `Installation <#installation>`__
6. `Competitions won with the
   library <#competitions-won-with-the-library>`__
7. `License <#license>`__
8. `Contributing <#contributing>`__

Quick start 
~~~~~~~~~~~~

Since the library is built on the PyTorch framework, created
segmentation model is just a PyTorch nn.Module, which can be created as
easy as:

.. code:: python

   import segmentation_models_pytorch as smp

   model = smp.Unet()

Depending on the task, you can change the network architecture by
choosing backbones with fewer or more parameters and use pretrainded
weights to initialize it:

.. code:: python

   model = smp.Unet('resnet34', encoder_weights='imagenet')

Change number of output classes in the model:

.. code:: python

   model = smp.Unet('resnet34', classes=3, activation='softmax')

All models have pretrained encoders, so you have to prepare your data
the same way as during weights pretraining:

.. code:: python

   from segmentation_models_pytorch.encoders import get_preprocessing_fn

   preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')

Examples 
~~~~~~~~~

-  Training model for cars segmentation on CamVid dataset
   `here <https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb>`__.
-  Training SMP model with
   `Catalyst <https://github.com/catalyst-team/catalyst>`__ (high-level
   framework for PyTorch), `Ttach <https://github.com/qubvel/ttach>`__
   (TTA library for PyTorch) and
   `Albumentations <https://github.com/albu/albumentations>`__ (fast
   image augmentation library) -
   `here <https://github.com/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb>`__
   |Open In Colab|


Installation 
~~~~~~~~~~~~~

PyPI version:

.. code:: bash

   $ pip install segmentation-models-pytorch

Latest version from source:

.. code:: bash

   $ pip install git+https://github.com/qubvel/segmentation_models.pytorch

Competitions won with the library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Segmentation Models`` package is widely used in the image segmentation
competitions.
`Here <https://github.com/qubvel/segmentation_models.pytorch/blob/master/HALLOFFAME.md>`__
you can find competitions, names of the winners and links to their
solutions.

License 
~~~~~~~~

Project is distributed under `MIT
License <https://github.com/qubvel/segmentation_models.pytorch/blob/master/LICENSE>`__

Contributing
~~~~~~~~~~~~

Run test
''''''''

.. code:: bash

   $ docker build -f docker/Dockerfile.dev -t smp:dev . && docker run --rm smp:dev pytest -p no:cacheprovider

Generate table
''''''''''''''

.. code:: bash

   $ docker build -f docker/Dockerfile.dev -t smp:dev . && docker run --rm smp:dev python misc/generate_table.py

.. |logo| image:: https://i.ibb.co/dc1XdhT/Segmentation-Models-V2-Side-1-1.png
.. |PyPI version| image:: https://badge.fury.io/py/segmentation-models-pytorch.svg
.. |Build Status| image:: https://travis-ci.com/qubvel/segmentation_models.pytorch.svg?branch=master
   :target: https://travis-ci.com/qubvel/segmentation_models.pytorch
.. |Generic badge| image:: https://img.shields.io/badge/License-MIT-%3CCOLOR%3E.svg
   :target: https://shields.io/
.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
