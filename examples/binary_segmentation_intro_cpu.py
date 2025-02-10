import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import cv2
import numpy as np
import logging

# ----------------------------
# Set the device to GPU if available
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# ----------------------------
# Download the CamVid dataset, if needed
# ----------------------------
root = os.getcwd()

DATA_DIR = os.path.join(root, 'data')
if not os.path.exists(DATA_DIR):
    logging.info('Loading data...')
    os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
    logging.info('Done!')

# ----------------------------
# Define a custom dataset class for the CamVid dataset
# ----------------------------
class Dataset(BaseDataset):

    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # Always map background ('unlabelled') to 0
        self.background_class = 0 
        self.class_values = [0, 1] 

        self.augmentation = augmentation

    def __getitem__(self, i):
        """
        Retrieves the image and corresponding mask at index `i`.
        Args:
            i (int): Index of the image and mask to retrieve.
        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): The preprocessed image tensor of shape (1, 160, 160) normalized to [0, 1].
                - mask_remap (torch.Tensor): The preprocessed mask tensor of shape (160, 160) with values 0 or 1.
        """
        # Read the image
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        
        # resize image to 160x160
        image = cv2.resize(image, (160, 160))

        # Read the mask in grayscale mode
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # Update the mask: Keep 0 as is, set all other values to 1
        mask_remap = np.where(mask == 0, 0, 1).astype(np.uint8)
        
        # resize mask to 160x160
        mask_remap = cv2.resize(mask_remap, (160, 160))

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]

        # Convert to PyTorch tensors
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)  # Add channel dimension if missing
        image = torch.tensor(image).float().permute(2, 0, 1) / 255.0  # HWC -> CHW and normalize to [0, 1]
        mask_remap = torch.tensor(mask_remap).long()  # Ensure mask is LongTensor

        return image, mask_remap

    def __len__(self):
        return len(self.ids)

# ----------------------------
# Define a function to visualize images and masks
# ----------------------------
def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())

        # If it's an image, plot it as RGB
        if name == "image":
            # Convert CHW to HWC for plotting
            image = image.numpy().transpose(1, 2, 0)

            plt.imshow(image)
        else:
            plt.imshow(image, cmap="tab20")
    plt.show()

# ----------------------------
# Define the data directories and create the datasets
# ----------------------------
x_train_dir = os.path.join(DATA_DIR, 'CamVid', 'train')
y_train_dir = os.path.join(DATA_DIR, 'CamVid', 'trainannot')

x_val_dir = os.path.join(DATA_DIR, 'CamVid', 'val')
y_val_dir = os.path.join(DATA_DIR, 'CamVid', 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'CamVid', 'test')
y_test_dir = os.path.join(DATA_DIR, 'CamVid', 'testannot')

train_dataset = Dataset(x_train_dir, y_train_dir)
valid_dataset = Dataset(x_val_dir, y_val_dir)
test_dataset = Dataset(x_test_dir, y_test_dir)

image, mask = train_dataset[0]
logging.info(f"Unique values in mask: {np.unique(mask)}")
logging.info(f"Image shape: {image.shape}")
logging.info(f"Mask shape: {mask.shape}")
visualize(image=image, mask=mask)

# ----------------------------
# Create the dataloaders using the datasets
# ----------------------------
logging.info(f"Train size: {len(train_dataset)}")
logging.info(f"Valid size: {len(valid_dataset)}")
logging.info(f"Test size: {len(test_dataset)}")

train_dataloader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, num_workers=64
)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=8, shuffle=False, num_workers=64
)
test_dataloader = DataLoader(
    test_dataset, batch_size=8, shuffle=False, num_workers=64
)

# Print the size of the first image in train_dataloader
logging.info(f"First image in train_dataloader shape: {next(iter(train_dataloader))[0].shape}")

# Print the size of the first mask in train_dataloader
logging.info(f"First mask in train_dataloader shape: {next(iter(train_dataloader))[1].shape}")

# Print the size of the first image in valid_dataloader
logging.info(f"First image in valid_dataloader shape: {next(iter(valid_dataloader))[0].shape}")

# Print the size of the first mask in valid_dataloader
logging.info(f"First mask in valid_dataloader shape: {next(iter(valid_dataloader))[1].shape}")

# Print the size of the first image in test_dataloader
logging.info(f"First image in test_dataloader shape: {next(iter(test_dataloader))[0].shape}")

# Print the size of the first mask in test_dataloader
logging.info(f"First mask in test_dataloader shape: {next(iter(test_dataloader))[1].shape}")

# ----------------------------
# Lets look at some samples
# ----------------------------
sample = train_dataset[0]
plt.subplot(1, 2, 1)
# for visualization we have to transpose back to HWC
plt.imshow(sample[0].numpy().transpose(1, 2, 0))
plt.subplot(1, 2, 2)
# for visualization we have to remove 3rd dimension of mask
plt.imshow(sample[1].squeeze())
plt.show()

sample = valid_dataset[0]
plt.subplot(1, 2, 1)
# for visualization we have to transpose back to HWC
plt.imshow(sample[0].numpy().transpose(1, 2, 0))
plt.subplot(1, 2, 2)
# for visualization we have to remove 3rd dimension of mask
plt.imshow(sample[1].squeeze())
plt.show()

sample = test_dataset[0]
plt.subplot(1, 2, 1)
# for visualization we have to transpose back to HWC
plt.imshow(sample[0].numpy().transpose(1, 2, 0))
plt.subplot(1, 2, 2)
# for visualization we have to remove 3rd dimension of mask
plt.imshow(sample[1].squeeze())
plt.show()


# ----------------------------
# Create the model
# ----------------------------

# Some training hyperparameters
EPOCHS = 10
T_MAX = EPOCHS * len(train_dataloader)
OUT_CLASSES = 1
ADAM_LEARNING_RATE = 2e-4


# Define a class for the CamVid model as an instance of a PyTorch Lightning module
class CamVidModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, out_classes=1, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            classes=out_classes,
            **kwargs,
        )

        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.number_of_classes = out_classes
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Loss function for binary segmentation
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch

        # Ensure that image dimensions are correct
        assert image.ndim == 4  # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()

        # Mask shape
        assert mask.ndim == 3  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using binary Dice loss (pass original mask, not one-hot encoded)
        loss = self.loss_fn(logits_mask, mask)

        # Apply sigmoid to get probabilities for binary segmentation
        prob_mask = logits_mask.sigmoid()

        # Convert probabilities to predicted class labels
        pred_mask = (prob_mask > 0.5).long()

        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="binary"
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # Per-image IoU and dataset IoU calculations
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=ADAM_LEARNING_RATE)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# ----------------------------
# Train the model
# ----------------------------
model = CamVidModel("Unet", "resnet34", out_classes=OUT_CLASSES)

# print the number of images in train_dataloader and the size of the first image
logging.info(train_dataloader.dataset.__len__())
logging.info(next(iter(train_dataloader))[0].shape)

# print the number of images in valid_dataloader and the size of the first image
logging.info(valid_dataloader.dataset.__len__())
logging.info(next(iter(valid_dataloader))[0].shape)


# Training loop
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

# Use multiple CPUs in parallel
os.system("export OMP_NUM_THREADS=64")
torch.set_num_threads(os.cpu_count())

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch in train_dataloader:
        images, masks = batch
        # images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        logging.info(f"Train Loss: {loss.item()}")
    
    scheduler.step()
    avg_train_loss = train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    logging.info(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in valid_dataloader:
            images, masks = batch
            # images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            val_loss += loss.item()
            logging.info(f"Validation Loss: {loss.item()}")
    
    avg_val_loss = val_loss / len(valid_dataloader)
    val_losses.append(avg_val_loss)
    logging.info(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {avg_val_loss}")

# Store the training history
history = {
    "train_losses": train_losses,
    "val_losses": val_losses,
}

# ----------------------------
# Test the model
# ----------------------------
model.eval()
test_loss = 0
with torch.no_grad():
    for batch in test_dataloader:
        images, masks = batch
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, masks)
        
        # Save the output segmentation mask
        if not os.path.exists(os.path.join(root, "output_masks")):
            os.makedirs(os.path.join(root, "output_masks"))
        for i, output in enumerate(outputs):
            output = output.squeeze().cpu().numpy()
            cv2.imwrite(f"{os.path.join(root, "output_masks")}/{i}.png", output)
            

        test_loss += loss.item()
        logging.info(f"Test Loss: {loss.item()}")
        
# Save the test loss in a text file
with open(os.path.join(root, "test_loss.txt"), "w") as f:
    f.write(f"Test Loss: {test_loss}")
    
  
# Read the output masks and save them again using plt.savefig
output_masks = os.listdir(os.path.join(root, "output_masks"))
for i, output_mask in enumerate(output_masks):
    output_mask = cv2.imread(f"output_masks/{output_mask}", cv2.IMREAD_GRAYSCALE)
    output_mask = (output_mask / output_mask.max()) * 255  # Normalize to [0, 255]
    plt.imsave(f"output_masks/{i}_2.png", output_mask, cmap="gray")
          