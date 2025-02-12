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

# ----------------------------
# Set the device to GPU if available
# ----------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info("Using GPU")
else:
    device = torch.device("cpu")
    logging.info("Using CPU")
    os.system("export OMP_NUM_THREADS=64")
    torch.set_num_threads(os.cpu_count())

# ----------------------------
# Download the CamVid dataset, if needed
# ----------------------------
main_dir = os.getcwd()

DATA_DIR = os.path.join(main_dir, "data")
if not os.path.exists(DATA_DIR):
    logging.info("Loading data...")
    os.system(f"git clone https://github.com/alexgkendall/SegNet-Tutorial {DATA_DIR}")
    logging.info("Done!")

# Create a directory to store the output masks
output_dir = os.path.join(main_dir, "output_images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    os.system(f"rm -rf {output_dir}")
    os.makedirs(output_dir)

# ----------------------------
# Define the hyperparameters
# ----------------------------
EPOCHS = 100
ADAM_LEARNING_RATE = 2e-4
INPUT_IMAGE_RESHAPE = (320, 320)
FOREGROUND_CLASS = 1  # Binary segmentation of buildings


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
    - INPUT_IMAGE_RESHAPE (tuple, optional): Desired shape for the input
      images and masks. Default is (320, 320).
    - FOREGROUND_CLASS (int, optional): The class value in the mask to be
      considered as the foreground. Default is 1.
    - augmentation (callable, optional): A function/transform to apply to the
      images and masks for data augmentation.
    """

    def __init__(
        self,
        images_dir,
        masks_dir,
        INPUT_IMAGE_RESHAPE=(320, 320),
        FOREGROUND_CLASS=1,
        augmentation=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.INPUT_IMAGE_RESHAPE = INPUT_IMAGE_RESHAPE
        self.FOREGROUND_CLASS = FOREGROUND_CLASS
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
            (1, INPUT_IMAGE_RESHAPE) - e.g., (1, 320, 320) - normalized to [0, 1].
            - mask_remap (torch.Tensor): The preprocessed mask tensor of
            shape INPUT_IMAGE_RESHAPE with values 0 or 1.
        """
        # Read the image
        image = cv2.imread(
            self.images_fps[i], cv2.IMREAD_GRAYSCALE
        )  # Read image as grayscale
        image = np.expand_dims(image, axis=-1)  # Add channel dimension

        # resize image to INPUT_IMAGE_RESHAPE
        image = cv2.resize(image, self.INPUT_IMAGE_RESHAPE)

        # Read the mask in grayscale mode
        mask = cv2.imread(self.masks_fps[i], 0)

        # Update the mask: Set FOREGROUND_CLASS to 1 and the rest to 0
        mask_remap = np.where(mask == self.FOREGROUND_CLASS, 1, 0).astype(np.uint8)

        # resize mask to INPUT_IMAGE_RESHAPE
        mask_remap = cv2.resize(mask_remap, self.INPUT_IMAGE_RESHAPE)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]

        # Convert to PyTorch tensors
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)  # Add channel dimension if missing
        image = (
            torch.tensor(image).float().permute(2, 0, 1) / 255.0
        )  # HWC -> CHW and normalize to [0, 1]
        mask_remap = torch.tensor(mask_remap).long()  # Ensure mask is LongTensor

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
    """
    Trains a given model using the provided training and validation
    dataloaders, optimizer, scheduler, and loss function.

    Parameters:
    ----------

    - model (torch.nn.Module): The model to be trained.
    - train_dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
    - valid_dataloader (torch.utils.data.DataLoader): DataLoader for the validation data.
    - optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
    - scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
    - loss_fn (callable): Loss function to compute the loss between the outputs and targets.
    - device (torch.device): Device to run the training on (e.g., 'cpu' or 'cuda').
    - epochs (int): Number of epochs to train the model.

    Returns:
    ----------

    - dict: A dictionary containing the training and validation loss
        history with keys 'train_losses' and 'val_losses'.
    """
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} - Training"
        ):
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
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(
                valid_dataloader, desc=f"Epoch {epoch + 1}/{epochs} - Validation"
            ):
                images, masks = batch
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, masks)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(valid_dataloader)
        val_losses.append(avg_val_loss)

        logging.info(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: \
                {avg_train_loss:.2f}, Validation Loss: {avg_val_loss:.2f}"
        )

    # Store the training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    return history


def evaluate_model(model, output_dir, test_dataloader, loss_fn, device):
    """
    Evaluate the given model on the test dataset and compute the loss
    and IoU score.

    Parameters:
    ----------

    - model (torch.nn.Module): The model to evaluate.
    - output_dir (str): Directory to save the output masks.
    - test_dataloader (torch.utils.data.DataLoader):
      DataLoader for the test dataset.
    - loss_fn (callable):
      Loss function to compute the loss.
    - device (torch.device):
      Device to perform the evaluation on (e.g., 'cpu' or 'cuda').

    Returns:
    ----------

    - A tuple containing:
        - test_loss (float): The average test loss.
        - iou_score (float): The Intersection over Union (IoU) score.
    """
    model.eval()
    test_loss = 0
    tp, fp, fn, tn = 0, 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            # Save the output segmentation mask
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
            # print(f"Test Loss: {loss.item()}")

            # Compute metrics
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

    # Calculate IoU
    iou_score = smp.metrics.iou_score(
        torch.tensor([tp]),
        torch.tensor([fp]),
        torch.tensor([fn]),
        torch.tensor([tn]),
        reduction="micro",
    )

    return test_loss, iou_score.item()


# ----------------------------
# Define the data directories and create the datasets
# ----------------------------
x_train_dir = os.path.join(DATA_DIR, "CamVid", "train")
y_train_dir = os.path.join(DATA_DIR, "CamVid", "trainannot")

x_val_dir = os.path.join(DATA_DIR, "CamVid", "val")
y_val_dir = os.path.join(DATA_DIR, "CamVid", "valannot")

x_test_dir = os.path.join(DATA_DIR, "CamVid", "test")
y_test_dir = os.path.join(DATA_DIR, "CamVid", "testannot")

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    INPUT_IMAGE_RESHAPE=INPUT_IMAGE_RESHAPE,
    FOREGROUND_CLASS=FOREGROUND_CLASS,
)
valid_dataset = Dataset(
    x_val_dir,
    y_val_dir,
    INPUT_IMAGE_RESHAPE=INPUT_IMAGE_RESHAPE,
    FOREGROUND_CLASS=FOREGROUND_CLASS,
)
test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    INPUT_IMAGE_RESHAPE=INPUT_IMAGE_RESHAPE,
    FOREGROUND_CLASS=FOREGROUND_CLASS,
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

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=64)

valid_dataloader = DataLoader(
    valid_dataset, batch_size=8, shuffle=False, num_workers=64
)

test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=64)

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
T_MAX = EPOCHS * len(train_dataloader)  # Total number of iterations

model = CamVidModel("Unet", "resnet34", in_channels=3, out_classes=1)

# Training loop
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
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
    EPOCHS,
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
test_loss = evaluate_model(model, output_dir, test_dataloader, loss_fn, device)

logging.info(f"Test Loss: {test_loss[0]}, IoU Score: {test_loss[1]}")
logging.info(f"The outout masks are saved in {output_dir}.")
