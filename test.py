import os  # isort: skip

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from utils import configuration
from utils.dataset import SegmDataset
from utils.build_model import build_model
from utils.augmentations import get_preprocessing, apply_validation_augmentation  # apply_preprocessing,


def setup_system(system_config: configuration.System) -> None:
    torch.manual_seed(system_config.seed)


def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def main(system_config=configuration.System):
    setup_system(system_config)

    img_dir = os.path.join(configuration.DataSet.root_dir, configuration.DataSet.img_val_dir)
    gt_dir = os.path.join(configuration.DataSet.root_dir, configuration.DataSet.mask_val_dir)

    net = build_model(configuration)

    net.load_state_dict(
        torch.load('./outputs/se_resnet50+fpn_2020-05-06 22:06:54.561792/best_model.pth')['model_state_dict']
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        configuration.Model.encoder, configuration.Model.encoder_weights
    )
    val_dataset = SegmDataset(
        img_dir,
        gt_dir,
        classes=configuration.DataSet.classes,
        augmentations=apply_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn)
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
    criterion = smp.utils.losses.DiceLoss(activation=configuration.Model.activation, ignore_channels=(0, ))
    # criterion = smp.utils.losses.JaccardLoss(activation='softmax2d')
    metrics = [
        smp.utils.metrics.IoU(
            threshold=0.5, activation=configuration.Model.activation, ignore_channels=(0, )
        ),
        smp.utils.metrics.Fscore(
            threshold=0.5, activation=configuration.Model.activation, ignore_channels=(0, )
        )
    ]

    valid_epoch = smp.utils.train.ValidEpoch(
        net, loss=criterion, metrics=metrics, device=configuration.Trainer.device, verbose=True
    )
    valid_logs = valid_epoch.run(dataloaders['val'])
    print(valid_logs['iou_score'])
    # imgs_num = 1
    # ids = np.random.choice(np.arange(len(val_dataset)), size=imgs_num)
    #
    # for i in ids:
    #     image, gt_mask = val_dataset[i]
    #     image = image.to('cuda')
    #     image = image.unsqueeze(0)
    #     pr_masks = net.predict(image)
    #     image = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    #     # pr_masks = pr_masks.softmax(dim=1).detach().cpu()
    #     # pr_masks = torch.where(pr_masks > 0.5, torch.tensor(1), torch.tensor(0))
    #     pr_masks = pr_masks.squeeze(0).permute(1, 2, 0).detach().cpu()
    #     for j in range(pr_masks.shape[2]):
    #         plt.subplot(1, 3, 1)
    #         plt.imshow(denormalize(image))
    #         plt.subplot(1, 3, 2)
    #         mask = torch.where(pr_masks[:, :, j].sigmoid() > 0.55, torch.Tensor(1), torch.Tensor(0))
    #         plt.imshow(mask)
    #         plt.subplot(1, 3, 3)
    #         plt.imshow(gt_mask[j, :, :])
    #         plt.show()


if __name__ == '__main__':
    main()
