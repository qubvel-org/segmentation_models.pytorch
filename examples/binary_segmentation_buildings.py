"""
This script demonstrates how to train a binary segmentation model using the
CamVid dataset and segmentation_models_pytorch. The CamVid dataset is a
collection of videos with pixel-level annotations for semantic segmentation.
The dataset includes 367 training images, 101 validation images, and 233 test.
Each training image has a corresponding mask that labels each pixel as belonging
to these classes with the numerical labels as follows:
- Sky: 0
- Building: 1
- Pole: 2
- Road: 3
- Pavement: 4
- Tree: 5
- SignSymbol: 6
- Fence: 7
- Car: 8
- Pedestrian: 9
- Bicyclist: 10
- Unlabelled: 11

In this script, we focus on binary segmentation, where the goal is to classify
each pixel as whether belonging to a certain class (Foregorund) or
not (Background).

Class Labels:
- 0: Background
- 1: Foreground

The script includes the following steps:
1. Set the device to GPU if available, otherwise use CPU.
2. Download the CamVid dataset if it is not already present.
3. Define hyperparameters for training.
4. Define a custom dataset class for loading and preprocessing the CamVid
     dataset.
5. Define a function to visualize images and masks.
6. Create datasets and dataloaders for training, validation, and testing.
7. Define a model class for the segmentation task.
8. Train the model using the training and validation datasets.
9. Evaluate the model using the test dataset and save the output masks and
     metrics.
"""

import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm

import segmentation_models_pytorch as smp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d:%m:%Y %H:%M:%S",
)

# ----------------------------
# Set the device to GPU if available
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")
if device == "cpu":
    os.system("export OMP_NUM_THREADS=64")
    torch.set_num_threads(os.cpu_count())

# ----------------------------
# Download the CamVid dataset, if needed
# ----------------------------
# Change this to your desired directory
main_dir = "./examples/binary_segmentation_data/"

data_dir = os.path.join(main_dir, "dataset")
if not os.path.exists(data_dir):
    logging.info("Loading data...")
    os.system(f"git clone https://github.com/alexgkendall/SegNet-Tutorial {data_dir}")
    logging.info("Done!")

# Create a directory to store the output masks
output_dir = os.path.join(main_dir, "output_images")
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# Define the hyperparameters
# ----------------------------
epochs_max = 200  # Number of epochs to train the model
adam_lr = 2e-4  # Learning rate for the Adam optimizer
eta_min = 1e-5  # Minimum learning rate for the scheduler
batch_size = 8  # Batch size for training
input_image_reshape = (320, 320)  # Desired shape for the input images and masks
foreground_class = 1  # 1 for binary segmentation


# ----------------------------
# Define a custom dataset class for the CamVid dataset
# ----------------------------
class Dataset(BaseDataset):
    """
    A custom dataset class for binary segmentation tasks.

    Parameters:
    ----------

    - images_dir (str): Directory containing the input images.
    - masks_dir (str): Directory containing the corresponding masks.
    - input_image_reshape (tuple, optional): Desired shape for the input
      images and masks. Default is (320, 320).
    - foreground_class (int, optional): The class value in the mask to be
      considered as the foreground. Default is 1.
    - augmentation (callable, optional): A function/transform to apply to the
      images and masks for data augmentation.
    """

    def __init__(
        self,
        images_dir,
        masks_dir,
        input_image_reshape=(320, 320),
        foreground_class=1,
        augmentation=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_filepaths = [
            os.path.join(images_dir, image_id) for image_id in self.ids
        ]
        self.masks_filepaths = [
            os.path.join(masks_dir, image_id) for image_id in self.ids
        ]

        self.input_image_reshape = input_image_reshape
        self.foreground_class = foreground_class
        self.augmentation = augmentation

    def __getitem__(self, i):
        """
        Retrieves the image and corresponding mask at index `i`.

        Parameters:
        ----------

        - i (int): Index of the image and mask to retrieve.
        Returns:
        - A tuple containing:
            - image (torch.Tensor): The preprocessed image tensor of shape
            (1, input_image_reshape) - e.g., (1, 320, 320) - normalized to [0, 1].
            - mask_remap (torch.Tensor): The preprocessed mask tensor of
            shape input_image_reshape with values 0 or 1.
        """
        # Read the image
        image = cv2.imread(
            self.images_filepaths[i], cv2.IMREAD_GRAYSCALE
        )  # Read image as grayscale
        image = np.expand_dims(image, axis=-1)  # Add channel dimension

        # resize image to input_image_reshape
        image = cv2.resize(image, self.input_image_reshape)

        # Read the mask in grayscale mode
        mask = cv2.imread(self.masks_filepaths[i], 0)

        # Update the mask: Set foreground_class to 1 and the rest to 0
        mask_remap = np.where(mask == self.foreground_class, 1, 0).astype(np.uint8)

        # resize mask to input_image_reshape
        mask_remap = cv2.resize(mask_remap, self.input_image_reshape)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]

        # Convert to PyTorch tensors
        # Add channel dimension if missing
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        # HWC -> CHW and normalize to [0, 1]
        image = torch.tensor(image).float().permute(2, 0, 1) / 255.0

        # Ensure mask is LongTensor
        mask_remap = torch.tensor(mask_remap).long()

        return image, mask_remap

    def __len__(self):
        return len(self.ids)


# Define a class for the CamVid model
class CamVidModel(torch.nn.Module):
    """
    A PyTorch model for binary segmentation using the Segmentation Models
    PyTorch library.

    Parameters:
    ----------

    - arch (str): The architecture name of the segmentation model
       (e.g., 'Unet', 'FPN').
    - encoder_name (str): The name of the encoder to use
       (e.g., 'resnet34', 'vgg16').
    - in_channels (int, optional): Number of input channels (e.g., 3 for RGB).
    - out_classes (int, optional): Number of output classes (e.g., 1 for binary)
    **kwargs: Additional keyword arguments to pass to the model
    creation function.
    """

    def __init__(self, arch, encoder_name, in_channels=3, out_classes=1, **kwargs):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask


def visualize(output_dir, image_filename, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()
    plt.savefig(os.path.join(output_dir, image_filename))
    plt.close()


# Use multiple CPUs in parallel
def train_and_evaluate_one_epoch(
    model, train_dataloader, valid_dataloader, optimizer, scheduler, loss_fn, device
):
    # Set the model to training mode
    model.train()
    train_loss = 0
    for batch in tqdm(train_dataloader, desc="Training"):
        images, masks = batch
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    avg_train_loss = train_loss / len(train_dataloader)

    # Set the model to evaluation mode
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc="Evaluating"):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(valid_dataloader)
    return avg_train_loss, avg_val_loss


def train_model(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epochs,
):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        avg_train_loss, avg_val_loss = train_and_evaluate_one_epoch(
            model,
            train_dataloader,
            valid_dataloader,
            optimizer,
            scheduler,
            loss_fn,
            device,
        )
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        logging.info(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.2f}, Validation Loss: {avg_val_loss:.2f}"
        )

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    return history


def test_model(model, output_dir, test_dataloader, loss_fn, device):
    # Set the model to evaluation mode
    model.eval()
    test_loss = 0
    tp, fp, fn, tn = 0, 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            for i, output in enumerate(outputs):
                input = images[i].cpu().numpy().transpose(1, 2, 0)
                output = output.squeeze().cpu().numpy()

                visualize(
                    output_dir,
                    f"output_{i}.png",
                    input_image=input,
                    output_mask=output,
                    binary_mask=output > 0.5,
                )

            test_loss += loss.item()

            prob_mask = outputs.sigmoid().squeeze(1)
            pred_mask = (prob_mask > 0.5).long()
            batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                pred_mask, masks, mode="binary"
            )
            tp += batch_tp.sum().item()
            fp += batch_fp.sum().item()
            fn += batch_fn.sum().item()
            tn += batch_tn.sum().item()

        test_loss_mean = test_loss / len(test_dataloader)
        logging.info(f"Test Loss: {test_loss_mean:.2f}")

    iou_score = smp.metrics.iou_score(
        torch.tensor([tp]),
        torch.tensor([fp]),
        torch.tensor([fn]),
        torch.tensor([tn]),
        reduction="micro",
    )

    return test_loss_mean, iou_score.item()


# ----------------------------
# Define the data directories and create the datasets
# ----------------------------
x_train_dir = os.path.join(data_dir, "CamVid", "train")
y_train_dir = os.path.join(data_dir, "CamVid", "trainannot")

x_val_dir = os.path.join(data_dir, "CamVid", "val")
y_val_dir = os.path.join(data_dir, "CamVid", "valannot")

x_test_dir = os.path.join(data_dir, "CamVid", "test")
y_test_dir = os.path.join(data_dir, "CamVid", "testannot")

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    input_image_reshape=input_image_reshape,
    foreground_class=foreground_class,
)
valid_dataset = Dataset(
    x_val_dir,
    y_val_dir,
    input_image_reshape=input_image_reshape,
    foreground_class=foreground_class,
)
test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    input_image_reshape=input_image_reshape,
    foreground_class=foreground_class,
)

image, mask = train_dataset[0]
logging.info(f"Unique values in mask: {np.unique(mask)}")
logging.info(f"Image shape: {image.shape}")
logging.info(f"Mask shape: {mask.shape}")

# ----------------------------
# Create the dataloaders using the datasets
# ----------------------------
logging.info(f"Train size: {len(train_dataset)}")
logging.info(f"Valid size: {len(valid_dataset)}")
logging.info(f"Test size: {len(test_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------------------
# Lets look at some samples
# ----------------------------
# Visualize and save train sample
sample = train_dataset[0]
visualize(
    output_dir,
    "train_sample.png",
    train_image=sample[0].numpy().transpose(1, 2, 0),
    train_mask=sample[1].squeeze(),
)

# Visualize and save validation sample
sample = valid_dataset[0]
visualize(
    output_dir,
    "validation_sample.png",
    validation_image=sample[0].numpy().transpose(1, 2, 0),
    validation_mask=sample[1].squeeze(),
)

# Visualize and save test sample
sample = test_dataset[0]
visualize(
    output_dir,
    "test_sample.png",
    test_image=sample[0].numpy().transpose(1, 2, 0),
    test_mask=sample[1].squeeze(),
)

# ----------------------------
# Create and train the model
# ----------------------------
max_iter = epochs_max * len(train_dataloader)  # Total number of iterations

model = CamVidModel("Unet", "resnet34", in_channels=3, out_classes=1)

# Training loop
model = model.to(device)

# Define the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)

# Define the learning rate scheduler
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=eta_min)

# Define the loss function
loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

# Train the model
history = train_model(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epochs_max,
)

# Visualize the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(history["train_losses"], label="Train Loss")
plt.plot(history["val_losses"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()
plt.savefig(os.path.join(output_dir, "train_val_losses.png"))
plt.close()


# Evaluate the model
test_loss = test_model(model, output_dir, test_dataloader, loss_fn, device)

logging.info(f"Test Loss: {test_loss[0]}, IoU Score: {test_loss[1]}")
logging.info(f"The output masks are saved in {output_dir}.")
