import os  # isort: skip

from shutil import copyfile
from datetime import datetime

import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import segmentation_models_pytorch as smp

from utils import configuration
from utils.dataset import SegmDataset
from utils.build_model import build_model
from utils.augmentations import apply_preprocessing, apply_training_augmentation, apply_validation_augmentation


def setup_system(system_config: configuration.System) -> None:
    torch.manual_seed(system_config.seed)


def console_initial_log(config, time):
    print(
        f'Initial LR: {config.Optimizer.learning_rate}\n', f'Batch Size: {config.DataLoader.batch_size}\n',
        f'Model Name: {config.Model.model_name}\n', f'Encoder: {config.Model.encoder}\n',
        f'Start_time: {time}'
    )


def main(system_config=configuration.System):
    time = datetime.now()
    copyfile(
        './utils/configuration.py', f'./configs/{configuration.Model.encoder}+'
        f'{configuration.Model.model_name}_{time}.py'
    )
    setup_system(system_config)
    writer = SummaryWriter(f'runs/{configuration.Model.encoder}+{configuration.Model.model_name}_{time}')
    img_dir = os.path.join(configuration.DataSet.root_dir, configuration.DataSet.img_dir)
    gt_dir = os.path.join(configuration.DataSet.root_dir, configuration.DataSet.mask_dir)
    set_dir = configuration.DataSet.set_dir
    console_initial_log(configuration, time)
    net = build_model(configuration)
    # snapshot = torch.load('./snapshots/best_model_config_dpn92+unet_2020-02-14 14:58:33.803400.pth')
    # net.load_state_dict(snapshot['model_state_dict'])
    print(net)
    train_dataset = SegmDataset(
        img_dir,
        gt_dir,
        os.path.join(set_dir, 'train.txt'),
        # apply_light_training_augmentation(),
        apply_training_augmentation(),
        apply_preprocessing()
    )
    val_dataset = SegmDataset(
        img_dir, gt_dir, os.path.join(set_dir, 'val.txt'), apply_validation_augmentation(),
        apply_preprocessing()
    )

    dataloaders = {
        'train':
            DataLoader(
                train_dataset,
                batch_size=configuration.DataLoader.batch_size,
                shuffle=True,
                num_workers=configuration.DataLoader.num_workers
            ),
        'val':
            DataLoader(
                val_dataset,
                batch_size=configuration.DataLoader.batch_size,
                shuffle=True,
                num_workers=configuration.DataLoader.num_workers
            )
    }
    criterion = smp.utils.losses.DiceLoss(activation=configuration.Model.activation)
    metrics = [smp.utils.metrics.IoU(threshold=0.5, activation=configuration.Model.activation)]

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=configuration.Optimizer.learning_rate,
        weight_decay=configuration.Optimizer.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=configuration.Optimizer.lr_step_milestones
    )

    train_epoch = smp.utils.train.TrainEpoch(
        net,
        loss=criterion,
        metrics=metrics,
        optimizer=optimizer,
        device=configuration.Trainer.device,
        verbose=True
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        net, loss=criterion, metrics=metrics, device=configuration.Trainer.device, verbose=True
    )
    max_score = 0

    for epoch in range(configuration.Trainer.epoch_num):

        print('\nEpoch: {}'.format(epoch))
        train_logs = train_epoch.run(dataloaders['train'])
        valid_logs = valid_epoch.run(dataloaders['val'])
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metric': max_score
            }, f'./snapshots/best_model_config_{configuration.Model.encoder}+{configuration.Model.model_name}_{time}.pth'
                      )
            print('Model saved!')
        for tag, value in train_logs.items():
            writer.add_scalar(tag + '_train', value, epoch)
        for tag, value in valid_logs.items():
            writer.add_scalar(tag + '_val', value, epoch)
        lr_scheduler.step(epoch)


if __name__ == '__main__':
    main()
