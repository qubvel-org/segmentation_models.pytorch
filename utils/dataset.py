import os

import cv2
import numpy as np

from torch.utils.data import Dataset


class SegmDataset(Dataset):
    def __init__(
        self,
        imgs_dir,
        masks_dir,
        classes=None,
        augmentations=None,
        preprocessing=None,
    ):
        self.possible_classes = [
            'sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian',
            'bicyclist', 'unlabelled'
        ]
        super().__init__()
        imgs_dir = imgs_dir if imgs_dir[-1] == '/' else imgs_dir + '/'
        masks_dir = masks_dir if masks_dir[-1] == '/' else masks_dir + '/'
        self.ids = os.listdir(imgs_dir)
        self.images_filenames = [os.path.join(imgs_dir, image_id) for image_id in self.ids]
        self.masks_filenames = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.class_values = [self.possible_classes.index(cls.lower()) for cls in classes]
        self.augmentations = augmentations
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        img = cv2.imread(self.images_filenames[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_filenames[i], 0)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((background, mask), axis=-1)

        if self.augmentations:
            sample = self.augmentations(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        return img, mask

    def __len__(self):
        return len(self.ids)
