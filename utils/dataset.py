import os

from glob import glob

import cv2
import numpy as np

from torch.utils.data import Dataset


class SegmDataset(Dataset):
    def __init__(
        self,
        imgs_dir,
        masks_dir,
        imgs_set,
        augmentations=None,
        preprocessing=None,
    ):
        classes = ['Epithelial', 'Lymphocyte', 'Macrophage', 'Neutrophil']
        super().__init__()
        with open(imgs_set, 'r') as file:
            self.ids = [line.strip() for line in file.readlines()]
        self.imgs = [
            path for img_id in self.ids for path in glob(os.path.join(imgs_dir, '*', f'{img_id}.tif'))
        ]
        self.masks = {img_id: [None, None, None, None] for img_id in self.ids}
        for img_id in self.ids:
            masks_paths = glob(os.path.join(masks_dir, '*', img_id, '*', '*.tif'))
            for mask_path in masks_paths:
                for i, cls in enumerate(classes):
                    if cls in mask_path:
                        self.masks[img_id][i] = mask_path
        self.augmentations = augmentations
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # Read image
        img = cv2.imread(self.imgs[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = [
            cv2.imread(class_mask, cv2.IMREAD_ANYDEPTH)
            if class_mask is not None else np.zeros(img.shape[:-1]) for class_mask in self.masks[self.ids[i]]
        ]
        mask = np.asarray(mask).transpose(1, 2, 0)
        # background = 1 - mask.max(axis=-1, keepdims=True, initial=0)
        sample = {'image': img, 'mask': mask}
        if self.augmentations is not None:
            sample = self.augmentations(**sample)

        if self.preprocessing is not None:
            sample = self.preprocessing(**sample)
            sample["mask"] = sample["mask"].bool().permute(2, 0, 1)

        return sample['image'].float(), sample['mask']

    def __len__(self):
        return len(self.ids)
