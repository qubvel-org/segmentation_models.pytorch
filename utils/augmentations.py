import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2


def apply_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=15, shift_limit=0.05),
        A.PadIfNeeded(320, 320, always_apply=True),
        A.Resize(320, 320, always_apply=True),
        # A.ElasticTransform(p=0.5),
        A.OneOf([
            A.HueSaturationValue(p=1),
            A.IAAAdditiveGaussianNoise(p=1),
        ], p=0.6),
        A.OneOf([
            A.RandomBrightnessContrast(p=1, brightness_limit=0.2, contrast_limit=0.2),
            A.RandomGamma(p=1),
        ],
                p=0.6),
        A.OneOf(
            [
                A.IAASharpen(p=1, alpha=(0.1, 0.3)),
                A.Blur(blur_limit=3, p=1),
            ],
            p=0.6,
        ),
    ]
    return A.Compose(train_transform)


def apply_light_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=15, shift_limit=0.05, p=0.5),
        A.PadIfNeeded(320, 320, always_apply=True),
        A.Resize(320, 320, always_apply=True),
    ]
    return A.Compose(train_transform)


def apply_preprocessing():
    return A.Compose([A.Normalize(), ToTensorV2()])


def get_preprocessing(preprocessing_fn):
    return A.Compose([A.Lambda(image=preprocessing_fn), ToTensorV2()])


def apply_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(320, 320, always_apply=True),
        A.Resize(320, 320, always_apply=True),
    ]
    return A.Compose(test_transform)


def apply_test_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [A.PadIfNeeded(320, 320, always_apply=True)]
    return A.Compose(test_transform)
