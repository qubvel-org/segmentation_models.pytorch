📉 Losses
=========

Collection of popular semantic segmentation losses. Adapted from 
an awesome repo with pytorch utils https://github.com/BloodAxe/pytorch-toolbelt

Constants
~~~~~~~~~
.. automodule:: segmentation_models_pytorch.losses.constants
        :members:

JaccardLoss
~~~~~~~~~~~
.. autoclass:: segmentation_models_pytorch.losses.JaccardLoss

DiceLoss
~~~~~~~~
.. autoclass:: segmentation_models_pytorch.losses.DiceLoss

TverskyLoss
~~~~~~~~
.. autoclass:: segmentation_models_pytorch.losses.TverskyLoss

FocalLoss
~~~~~~~~~
.. autoclass:: segmentation_models_pytorch.losses.FocalLoss

LovaszLoss
~~~~~~~~~~
.. autoclass:: segmentation_models_pytorch.losses.LovaszLoss

SoftBCEWithLogitsLoss
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: segmentation_models_pytorch.losses.SoftBCEWithLogitsLoss

SoftCrossEntropyLoss
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: segmentation_models_pytorch.losses.SoftCrossEntropyLoss

MCCLoss
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: segmentation_models_pytorch.losses.MCCLoss
        :members: forward
