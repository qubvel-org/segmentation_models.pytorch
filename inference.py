import os  # isort: skip

from time import time

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
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

    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    setup_system(system_config)

    img_dir = os.path.join(configuration.DataSet.root_dir, configuration.DataSet.img_val_dir)
    gt_dir = os.path.join(configuration.DataSet.root_dir, configuration.DataSet.mask_val_dir)

    net = build_model(configuration)

    net.load_state_dict(
        torch.load('./outputs/mobilenet_v2+deeplab_v3+_2020-05-25-22-18/best_model.pth')['model_state_dict']
    )
    net.to(configuration.Trainer.device)
    print(net)
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

    dataloaders = DataLoader(
        val_dataset, batch_size=1, shuffle=True, num_workers=configuration.DataLoader.num_workers
    )
    count = 0
    possible_classes = (
        'sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian',
        'bicyclist', 'unlabelled'
    )
    start = time()
    for image, gt_mask in tqdm(dataloaders):
        count += 1
        image = image.to(configuration.Trainer.device)
        pr_masks = net.predict(image)
        image = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        pr_masks = pr_masks.squeeze(0).permute(1, 2, 0).detach().cpu()
        pr_masks = pr_masks.softmax(dim=-1)
        gt_mask = gt_mask.squeeze(0).permute(1, 2, 0).detach().cpu()
        pr_masks = pr_masks.argmax(dim=-1)
        gt_mask = gt_mask.argmax(dim=-1)
        plt.subplot(131)
        plt.imshow(denormalize(image))
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.title('Image')
        plt.subplot(132)
        plt.imshow(gt_mask.numpy())
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.title('GT Mask')
        plt.subplot(133)
        plt.imshow(pr_masks.numpy())
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.title('Predicted')
        plt.savefig(f'./vis_images/inference/{count}.png')
    #     i = 0
    #     plt.figure(figsize=(30, 70))
    #     for j in range(12):
    #         plt.subplot(7, 2, j + 1)
    #         if j % 2 == 0:
    #             plt.title(f'GT {possible_classes[i]}', fontsize=48)
    #             plt.xticks([])
    #             plt.yticks([])
    #             plt.imshow(gt_mask[..., i])
    #         else:
    #             plt.title(f'Predicted {possible_classes[i]}', fontsize=48)
    #             plt.xticks([])
    #             plt.yticks([])
    #             pr_masks[..., i][pr_masks[..., i] >= 0.5] = 1
    #             pr_masks[..., i][pr_masks[..., i] < 0.5] = 0
    #             plt.imshow(pr_masks[..., i])
    #             i += 1
    #     plt.savefig(f'./vis_images/inference/{count}_1.png')
    #     plt.figure(figsize=(30, 70))
    #     for j in range(12):
    #         plt.subplot(7, 2, j + 1)
    #         if j % 2 == 0:
    #             plt.title(f'GT {possible_classes[i]}', fontsize=48)
    #             plt.xticks([])
    #             plt.yticks([])
    #             plt.imshow(gt_mask[..., i])
    #         else:
    #             plt.title(f'Predicted {possible_classes[i]}', fontsize=48)
    #             plt.xticks([])
    #             plt.yticks([])
    #             pr_masks[..., i][pr_masks[..., i] >= 0.5] = 1
    #             pr_masks[..., i][pr_masks[..., i] < 0.5] = 0
    #             plt.imshow(pr_masks[..., i])
    #             i += 1
    #     plt.savefig(f'./vis_images/inference/{count}_2.png')
    # print('FPS: ', (time() - start) / len(dataloaders))


if __name__ == '__main__':
    main()
