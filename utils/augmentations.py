import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


# define heavy augmentations
def apply_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.RandomCrop(height=320, width=320, always_apply=True),
        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def apply_light_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        # A.RandomResizedCrop(320, 320, p=0.5),
        A.Resize(320, 320, always_apply=True),
    ]
    return A.Compose(train_transform)


def apply_preprocessing():
    return A.Compose([A.Normalize(), ToTensorV2()])


def get_preprocessing(preprocessing_fn):
    return A.Compose([A.Lambda(image=preprocessing_fn), ToTensorV2()])


def apply_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [A.PadIfNeeded(384, 480)]
    return A.Compose(test_transform)


def apply_test_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [A.PadIfNeeded(384, 480)]
    return A.Compose(test_transform)
