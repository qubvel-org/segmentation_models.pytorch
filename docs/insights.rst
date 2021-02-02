🔧 Insights
===========

1. Models architecture
~~~~~~~~~~~~~~~~~~~~~~

All segmentation models in SMP (this library short name) are made of:

- encoder (feature extractor, a.k.a backbone)
- decoder (features fusion block to create segmentation *mask*)
- segmentation head (final head to reduce number of channels from decoder and upsample mask to preserve input-output spatial resolution identity)
- classification head (optional head which build on top of deepest encoder features)


2. Creating your own encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Encoder is a "classification model" which extract features from image and pass it to decoder.
Each encoder should have following attributes and methods and be inherited from `segmentation_models_pytorch.encoders._base.EncoderMixin`

.. code-block:: python

    class MyEncoder(torch.nn.Module, EncoderMixin):
        
        def __init__(self, **kwargs):
            super().__init__()
            
            # A number of channels for each encoder feature tensor, list of integers
            self._out_channels: List[int] = [3, 16, 64, 128, 256, 512]

            # A number of stages in decoder (in other words number of downsampling operations), integer
            # use in in forward pass to reduce number of returning features
            self._depth: int = 5 

            # Default number of input channels in first Conv2d layer for encoder (usually 3)
            self._in_channels: int = 3 
            
            # Define encoder modules below
            ...

        def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
            """Produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
            shape NCHW (features should be sorted in descending order according to spatial resolution, starting
            with resolution same as input `x` tensor).

            Input: `x` with shape (1, 3, 64, 64)
            Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                    [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                    (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

            also should support number of features according to specified depth, e.g. if depth = 5,
            number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
            depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
            """

            return [feat1, feat2, feat3, feat4, feat5, feat6]

When you write your own Encoder class register its build parameters

.. code-block:: python

    smp.encoders.encoders["my_awesome_encoder"] = {
        "encoder": MyEncoder, # encoder class here
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://some-url.com/my-model-weights",
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": {
            # init params for encoder if any
        },
    },

Now you can use your encoder

.. code-block:: python

    model = smp.Unet(encoder_name="my_awesome_encoder")

For better understanding see more examples of encoder in smp.encoders module.

.. note::

    If it works fine, don`t forget to contribute your work and make a PR to SMP 😉

3. Aux classification output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All models support ``aux_params`` parameter, which is default set to ``None``. 
If ``aux_params = None`` than classification auxiliary output is not created, else
model produce not only ``mask``, but also ``label`` output with shape ``(N, C)``.

Classification head consist of following layers:
    
1. GlobalPooling
2. Dropout (optional)
3. Linear
4. Activation (optional)

Example:

.. code-block:: python
    
    aux_params=dict(
        pooling='avg',             # one of 'avg', 'max'
        dropout=0.5,               # dropout ratio, default is None
        activation='sigmoid',      # activation function, default is None
        classes=4,                 # define number of output labels
    )

    model = smp.Unet('resnet34', classes=4, aux_params=aux_params)
    mask, label = model(x)

    mask.shape, label.shape
    # (N, 4, H, W), (N, 4)
