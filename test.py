import os  # isort: skip

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from utils import configuration
from utils.dataset import SegmDataset
from utils.augmentations import apply_preprocessing, apply_validation_augmentation


def setup_system(system_config: configuration.System) -> None:
    torch.manual_seed(system_config.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic


def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def main(system_config=configuration.System):
    setup_system(system_config)

    img_dir = os.path.join(configuration.DataSet.root_dir, configuration.DataSet.img_dir)
    gt_dir = os.path.join(configuration.DataSet.root_dir, configuration.DataSet.mask_dir)
    set_dir = configuration.DataSet.set_dir

    net = smp.Unet(
        encoder_name=configuration.Model.encoder,
        encoder_weights=configuration.Model.encoder_weights,
        activation=configuration.Model.activation,
        classes=configuration.DataSet.number_of_classes
    )
    net.load_state_dict(
        torch.load('./snapshots/best_model_config_se_resnet50+unet_2020-02-11 15:19:37.434428.pth')
        ['model_state_dict']
    )
    val_dataset = SegmDataset(
        img_dir, gt_dir, os.path.join(set_dir, 'val.txt'), apply_validation_augmentation(),
        apply_preprocessing()
    )

    dataloaders = {
        'val':
            DataLoader(
                val_dataset,
                batch_size=configuration.DataLoader.batch_size,
                shuffle=True,
                num_workers=configuration.DataLoader.num_workers
            )
    }
    criterion = smp.utils.losses.DiceLoss(activation='softmax2d')
    # criterion = smp.utils.losses.JaccardLoss(activation='softmax2d')
    metrics = [smp.utils.metrics.IoU(threshold=0.5)]

    valid_epoch = smp.utils.train.ValidEpoch(
        net, loss=criterion, metrics=metrics, device=configuration.Trainer.device, verbose=True
    )
    valid_logs = valid_epoch.run(dataloaders['val'])
    print(valid_logs['iou_score'])
    imgs_num = 1
    ids = np.random.choice(np.arange(len(val_dataset)), size=imgs_num)

    for i in ids:
        image, gt_mask = val_dataset[i]
        image = image.to('cuda')
        image = image.unsqueeze(0)
        pr_masks = net.predict(image)
        image = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # pr_masks = pr_masks.softmax(dim=1).detach().cpu()
        # pr_masks = torch.where(pr_masks > 0.5, torch.tensor(1), torch.tensor(0))
        pr_masks = pr_masks.squeeze(0).permute(1, 2, 0).detach().cpu()
        for j in range(pr_masks.shape[2]):
            plt.subplot(1, 3, 1)
            plt.imshow(denormalize(image))
            plt.subplot(1, 3, 2)
            mask = torch.where(pr_masks[:, :, j].sigmoid() > 0.55, torch.Tensor(1), torch.Tensor(0))
            plt.imshow(mask)
            plt.subplot(1, 3, 3)
            plt.imshow(gt_mask[j, :, :])
            plt.show()


if __name__ == '__main__':
    main()
