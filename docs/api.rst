Models API 
==========

-  ``model.encoder`` - pretrained backbone to extract features of
   different spatial resolution
-  ``model.decoder`` - depends on models architecture
   (``Unet``/``Linknet``/``PSPNet``/``FPN``)
-  ``model.segmentation_head`` - last block to produce required number
   of mask channels (include also optional upsampling and activation)
-  ``model.classification_head`` - optional block which create
   classification head on top of encoder
-  ``model.forward(x)`` - sequentially pass ``x`` through model`s
   encoder, decoder and segmentation head (and classification head if
   specified)

Input channels
--------------

Input channels parameter allow you to create models, which process
tensors with arbitrary number of channels. If you use pretrained weights
from imagenet - weights of first convolution will be reused for 1- or 2-
channels inputs, for input channels > 4 weights of first convolution
will be initialized randomly.

.. code:: python

   model = smp.FPN('resnet34', in_channels=1)
   mask = model(torch.ones([1, 1, 64, 64]))

Auxiliary classification output
-------------------------------

All models support ``aux_params`` parameters, which is default set to
``None``. If ``aux_params = None`` than classification auxiliary output
is not created, else model produce not only ``mask``, but also ``label``
output with shape ``NC``. Classification head consist of
GlobalPooling->Dropout(optional)->Linear->Activation(optional) layers,
which can be configured by ``aux_params`` as follows:

.. code:: python

   aux_params=dict(
       pooling='avg',             # one of 'avg', 'max'
       dropout=0.5,               # dropout ratio, default is None
       activation='sigmoid',      # activation function, default is None
       classes=4,                 # define number of output labels
   )
   model = smp.Unet('resnet34', classes=4, aux_params=aux_params)
   mask, label = model(x)

Depth
-----

Depth parameter specify a number of downsampling operations in encoder,
so you can make your model lighted if specify smaller ``depth``.

.. code:: python

   model = smp.Unet('resnet34', encoder_depth=4)
