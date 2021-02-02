⏳ Quick Start
==============

**1. Create segmentation model**

Segmentation model is just a PyTorch nn.Module, which can be created as easy as:

.. code-block:: python
    
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,                      # model output channels (number of classes in your dataset)
    )

- see table with available model architectures
- see table with avaliable encoders and its corresponding weights

**2. Configure data preprocessing**

All encoders have pretrained weights. Preparing your data the same way as during weights pre-training may give your better results (higher metric score and faster convergence). But it is relevant only for 1-2-3-channels images and **not necessary** in case you train the whole model, not only decoder.

.. code-block:: python

    from segmentation_models_pytorch.encoders import get_preprocessing_fn

    preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')


**3. Congratulations!** 🎉


You are done! Now you can train your model with your favorite framework!
