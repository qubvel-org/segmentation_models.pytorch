📂 Saving and Loading
=====================

In this section, we will discuss how to save a trained model, push it to the Hugging Face Hub, and load it back for later use.

Saving and Sharing a Model
--------------------------

Once you have trained your model, you can save it using the `.save_pretrained` method. This method saves the model configuration and weights to a directory of your choice.
And, optionally, you can push the model to the Hugging Face Hub by setting the `push_to_hub` parameter to `True`.

For example:

.. code:: python

    import segmentation_models_pytorch as smp

    model = smp.Unet('resnet34', encoder_weights='imagenet')

    # After training your model, save it to a directory
    model.save_pretrained('./my_model')

    # Or saved and pushed to the Hub simultaneously
    model.save_pretrained('username/my-model', push_to_hub=True)

Loading Trained Model
---------------------

Once your model is saved and pushed to the Hub, you can load it back using the `smp.from_pretrained` method. This method allows you to load the model weights and configuration from a directory or directly from the Hub.

For example:

.. code:: python

    import segmentation_models_pytorch as smp

    # Load the model from the local directory
    model = smp.from_pretrained('./my_model')

    # Alternatively, load the model directly from the Hugging Face Hub
    model = smp.from_pretrained('username/my-model')

Saving model Metrics and Dataset Name
-------------------------------------

You can simply pass the `metrics` and `dataset` parameters to the `save_pretrained` method to save the model metrics and dataset name in Model Card along with the model configuration and weights.

For example:

.. code:: python

    import segmentation_models_pytorch as smp

    model = smp.Unet('resnet34', encoder_weights='imagenet')

    # After training your model, save it to a directory
    model.save_pretrained('./my_model', metrics={'accuracy': 0.95}, dataset='my_dataset')

    # Or saved and pushed to the Hub simultaneously
    model.save_pretrained('username/my-model', push_to_hub=True, metrics={'accuracy': 0.95}, dataset='my_dataset')


Conclusion
----------

By following these steps, you can easily save, share, and load your models, facilitating collaboration and reproducibility in your projects. Don't forget to replace the placeholders with your actual model paths and names.

|colab-badge|

.. |colab-badge| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/main/examples/binary_segmentation_intro.ipynb
    :alt: Open In Colab


