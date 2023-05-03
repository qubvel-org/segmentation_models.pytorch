from typing import Optional, Union, List, Tuple

import torch
from segment_anything.modeling import MaskDecoder, TwoWayTransformer, PromptEncoder
from torch.nn import functional as F

from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
)
from segmentation_models_pytorch.encoders import get_encoder


class SAM(SegmentationModel):
    """SAM_ (Segment Anything Model) is a visual transformer based encoder-decoder segmentation
    model that can be used to produce high quality segmentation masks from images and prompts.
    Consists of *image encoder*, *prompt encoder* and *mask decoder*. *Segmentation head* is
    added after the *mask decoder* to define the final number of classes for the output mask.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [6, 24]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"sa-1b"** (pre-training on SA-1B dataset).
        decoder_channels: How many output channels image encoder will have. Default is 256.
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: SAM

    .. _SAM:
        https://github.com/facebookresearch/segment-anything

    """

    def __init__(
        self,
        encoder_name: str = "sam-vit_h",
        encoder_depth: int = None,
        encoder_weights: Optional[str] = "sam-vit_h",
        decoder_channels: List[int] = 256,
        decoder_multimask_output: bool = True,
        in_channels: int = 3,
        image_size: int = 1024,
        vit_patch_size: int = 16,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            img_size=image_size,
            patch_size=vit_patch_size,
            out_chans=decoder_channels,
        )

        image_embedding_size = image_size // vit_patch_size
        self.prompt_encoder = PromptEncoder(
            embed_dim=decoder_channels,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )

        self.decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=decoder_channels,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=decoder_channels,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self._decoder_multiclass_output = decoder_multimask_output

        self.segmentation_head = SegmentationHead(
            in_channels=3 if decoder_multimask_output else 1,
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            raise NotImplementedError("Auxiliary output is not supported yet")
        self.classification_head = None

        self.name = encoder_name
        self.initialize()

    def preprocess(self, x):
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.encoder.img_size - h
        padw = self.encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.encoder.img_size, self.encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def forward(self, x):
        img_size = x.shape[-2:]
        x = torch.stack([self.preprocess(img) for img in x])
        features = self.encoder(x)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None, boxes=None, masks=None)
        low_res_masks, iou_predictions = self.decoder(
            image_embeddings=features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self._decoder_multiclass_output,
        )
        masks = self.postprocess_masks(low_res_masks, input_size=img_size, original_size=img_size)
        output = self.segmentation_head(masks)
        return output
