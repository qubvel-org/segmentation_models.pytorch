import torch

import segmentation_models_pytorch as smp

model = smp.Unet("resnet18",
                 encoder_weights=None,
                 decoder_channels=(64, 32, 16),
                 decoder_depth=3,
                 add_segmentation_head=False)

model.eval()


images = torch.rand(1,3,1024,1024)
print("Image shape: ",images.shape)
with torch.inference_mode():
    mask = model(images)

print("Mask Shape: ",mask.shape)
