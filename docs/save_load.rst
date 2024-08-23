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

Saving with preprocessing transform (Albumentations)
----------------------------------------------------

You can save the preprocessing transform along with the model and push it to the Hub. 
This can be useful when you want to share the model with the preprocessing transform that was used during training, 
to make sure that the inference pipeline is consistent with the training pipeline.

.. code:: python

    import albumentations as A
    import segmentation_models_pytorch as smp

    # Define a preprocessing transform for image that would be used during inference
    preprocessing_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize()
    ])

    model = smp.Unet()

    directory_or_repo_on_the_hub = "qubvel-hf/unet-with-transform"  # <username>/<repo-name>

    # Save the model and transform (and pus ot hub, if needed)
    model.save_pretrained(directory_or_repo_on_the_hub, push_to_hub=True)
    preprocessing_transform.save_pretrained(directory_or_repo_on_the_hub, push_to_hub=True)

    # Loading transform and model
    restored_model = smp.from_pretrained(directory_or_repo_on_the_hub)
    restored_transform = A.Compose.from_pretrained(directory_or_repo_on_the_hub)

    print(restored_transform)

Conclusion
----------

By following these steps, you can easily save, share, and load your models, facilitating collaboration and reproducibility in your projects. Don't forget to replace the placeholders with your actual model paths and names.

|colab-badge|

.. |colab-badge| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/main/examples/binary_segmentation_intro.ipynb
    :alt: Open In Colab

.. |colab-badge| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/main/examples/save_load_model_and_share_with_hf_hub.ipynb
    :alt: Open In Colab
