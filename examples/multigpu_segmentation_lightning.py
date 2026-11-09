"""
Multi-GPU (DDP) fine-tuning of a segmentation_models_pytorch model with
PyTorch Lightning on the CamVid dataset (multiclass segmentation).

The script shows how to:
1. Wrap an smp model in a ``LightningModule`` so Lightning handles DDP,
   mixed precision, checkpointing and logging.
2. Compute metrics with correct gathering of results across processes
   using ``torchmetrics`` (confusion-matrix based IoU, synced on epoch end).
3. Visualize predictions on the validation set (rank 0 only).

Install the extra dependencies:

    pip install segmentation-models-pytorch lightning torchmetrics matplotlib opencv-python

Launch examples:

    # all visible GPUs on one machine (Lightning spawns the DDP processes)
    python examples/multigpu_segmentation_lightning.py --devices -1

    # 2 GPUs, different architecture / encoder
    python examples/multigpu_segmentation_lightning.py --devices 2 \
        --arch FPN --encoder resnet50 --batch-size 8

    # quick smoke test on CPU with 2 DDP processes, no pretrained weights
    python examples/multigpu_segmentation_lightning.py --accelerator cpu \
        --devices 2 --epochs 1 --encoder-weights none --limit-batches 2

Note: ``--batch-size`` is per process (per GPU), so the effective batch size is
``batch_size * num_gpus``.

CamVid (https://github.com/alexgkendall/SegNet-Tutorial) has 367 train,
101 val and 233 test images and 11 classes + 1 "unlabelled" class which is
ignored in the loss and in the metrics.
"""

import argparse
import os
import subprocess

import cv2
import lightning as L
import matplotlib

matplotlib.use("Agg")  # no display needed on training machines

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import segmentation_models_pytorch as smp

CLASSES = [
    "sky",
    "building",
    "pole",
    "road",
    "pavement",
    "tree",
    "signsymbol",
    "fence",
    "car",
    "pedestrian",
    "bicyclist",
]
UNLABELLED = 11  # index of the "unlabelled" class in the CamVid masks
NUM_CLASSES = len(CLASSES)


# ----------------------------
# Data
# ----------------------------
def download_camvid(main_dir):
    """Clone the CamVid copy used by the other smp examples (once)."""
    data_dir = os.path.join(main_dir, "CamVid")
    if not os.path.exists(data_dir):
        os.makedirs(main_dir, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/alexgkendall/SegNet-Tutorial",
                os.path.join(main_dir, "SegNet-Tutorial"),
            ],
            check=True,
        )
        os.rename(os.path.join(main_dir, "SegNet-Tutorial", "CamVid"), data_dir)
    return data_dir


class CamVidDataset(BaseDataset):
    def __init__(self, images_dir, masks_dir, image_size=(320, 320), augment=False):
        self.ids = sorted(os.listdir(images_dir))
        self.images = [os.path.join(images_dir, i) for i in self.ids]
        self.masks = [os.path.join(masks_dir, i) for i in self.ids]
        self.image_size = image_size  # (H, W), must be divisible by 32
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(self.images[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE)
        h, w = self.image_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.augment and np.random.rand() < 0.5:  # horizontal flip
            image, mask = image[:, ::-1], mask[:, ::-1]

        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return image / 255.0, mask


class CamVidDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, image_size, limit=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.limit = limit

    def setup(self, stage=None):
        def make(split, augment):
            ds = CamVidDataset(
                os.path.join(self.data_dir, split),
                os.path.join(self.data_dir, split + "annot"),
                self.image_size,
                augment,
            )
            if self.limit:
                ds.ids = ds.ids[: self.limit]
                ds.images = ds.images[: self.limit]
                ds.masks = ds.masks[: self.limit]
            return ds

        self.train_ds = make("train", augment=True)
        self.val_ds = make("val", augment=False)

    # Lightning replaces the default sampler with a DistributedSampler under
    # DDP, so every process sees a different shard of the data.
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


# ----------------------------
# Model
# ----------------------------
class SegmentationModule(L.LightningModule):
    def __init__(self, arch, encoder, encoder_weights, lr, epochs):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder,
            encoder_weights=None if encoder_weights == "none" else encoder_weights,
            in_channels=3,
            classes=NUM_CLASSES,
        )
        params = smp.encoders.get_preprocessing_params(encoder)
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))

        # Unlabelled pixels (index 11) are excluded from loss and metrics.
        self.loss_fn = smp.losses.DiceLoss(
            smp.losses.MULTICLASS_MODE, from_logits=True, ignore_index=UNLABELLED
        )
        # torchmetrics keeps per-process state and all-reduces it in compute(),
        # so the epoch value is computed over the *whole* validation set, not
        # averaged over per-rank / per-batch values.
        self.val_iou = torchmetrics.classification.MulticlassJaccardIndex(
            num_classes=NUM_CLASSES, ignore_index=UNLABELLED, average="macro"
        )
        self.val_iou_per_class = torchmetrics.classification.MulticlassJaccardIndex(
            num_classes=NUM_CLASSES, ignore_index=UNLABELLED, average="none"
        )

    def forward(self, image):
        return self.model((image - self.mean) / self.std)

    def training_step(self, batch, batch_idx):
        image, mask = batch
        loss = self.loss_fn(self(image), mask)
        # sync_dist averages the logged value across processes
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        logits = self(image)
        self.log("val_loss", self.loss_fn(logits, mask), sync_dist=True)
        pred = logits.argmax(dim=1)
        self.val_iou.update(pred, mask)
        self.val_iou_per_class.update(pred, mask)

    def on_validation_epoch_end(self):
        # Passing the Metric object lets Lightning call compute() (which syncs
        # state over all ranks) and reset() for us.
        self.log("val_iou", self.val_iou, prog_bar=True)
        per_class = self.val_iou_per_class.compute()  # synced across ranks
        self.val_iou_per_class.reset()
        for name, value in zip(CLASSES, per_class):
            self.log(f"val_iou/{name}", value, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.epochs, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": sched}


# ----------------------------
# Visualization (rank 0 only)
# ----------------------------
class VisualizePredictions(Callback):
    def __init__(self, out_dir, num_samples=4):
        self.out_dir = out_dir
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        ds = trainer.datamodule.val_ds
        idx = np.linspace(0, len(ds) - 1, self.num_samples).astype(int)
        image, mask = map(torch.stack, zip(*[ds[i] for i in idx]))
        pl_module.eval()
        with torch.no_grad():
            pred = pl_module(image.to(pl_module.device)).argmax(1).cpu()
        pl_module.train()

        fig, axes = plt.subplots(self.num_samples, 3, figsize=(9, 3 * self.num_samples))
        for row, (im, gt, pr) in enumerate(zip(image, mask, pred)):
            axes[row, 0].imshow(im.permute(1, 2, 0))
            axes[row, 1].imshow(gt, vmin=0, vmax=UNLABELLED, cmap="tab20")
            axes[row, 2].imshow(pr, vmin=0, vmax=UNLABELLED, cmap="tab20")
        for ax, title in zip(axes[0], ["image", "ground truth", "prediction"]):
            ax.set_title(title)
        for ax in axes.ravel():
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(
            os.path.join(self.out_dir, f"epoch_{trainer.current_epoch:03d}.png")
        )
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", default="./examples/camvid_data")
    p.add_argument("--arch", default="Unet")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--encoder-weights", default="imagenet", help="or 'none'")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8, help="per process")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--image-size", type=int, nargs=2, default=(320, 320))
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices", type=int, default=-1, help="-1 = all GPUs")
    p.add_argument(
        "--precision", default="16-mixed" if torch.cuda.is_available() else "32"
    )
    p.add_argument("--limit-batches", type=int, default=None, help="debug")
    p.add_argument("--out-dir", default="./examples/camvid_lightning_output")
    args = p.parse_args()

    L.seed_everything(42, workers=True)

    # Only one process should download; the others wait at the barrier below.
    # (Under the default "subprocess" DDP launch every rank runs this script.)
    data_dir = os.path.join(args.data_dir, "CamVid")
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        data_dir = download_camvid(args.data_dir)
    else:
        import time

        while not os.path.exists(os.path.join(data_dir, "testannot")):
            time.sleep(2)

    dm = CamVidDataModule(
        data_dir,
        args.batch_size,
        args.num_workers,
        tuple(args.image_size),
        limit=args.limit_batches and args.limit_batches * args.batch_size * 2,
    )
    module = SegmentationModule(
        args.arch, args.encoder, args.encoder_weights, args.lr, args.epochs
    )
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy="ddp_find_unused_parameters_false",
        precision=args.precision,
        max_epochs=args.epochs,
        default_root_dir=args.out_dir,
        callbacks=[
            ModelCheckpoint(monitor="val_iou", mode="max", save_top_k=1),
            VisualizePredictions(os.path.join(args.out_dir, "predictions")),
        ],
        log_every_n_steps=10,
    )
    trainer.fit(module, datamodule=dm)

    if trainer.is_global_zero:
        best = trainer.checkpoint_callback.best_model_path
        print(f"Best checkpoint: {best}")
        print(f"Best val_iou: {trainer.checkpoint_callback.best_model_score}")


if __name__ == "__main__":
    main()
