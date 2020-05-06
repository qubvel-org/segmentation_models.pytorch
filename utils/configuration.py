from typing import Iterable
from dataclasses import dataclass


@dataclass
class System:
    seed: int = 42


@dataclass
class DataSet:
    root_dir: str = "./data"
    img_dir: str = "train"
    mask_dir: str = "trainannot"
    img_val_dir: str = "val"
    mask_val_dir: str = "valannot"
    number_of_classes: int = 13
    classes: tuple = (
        'sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian',
        'bicyclist', 'unlabelled'
    )


@dataclass
class DataLoader:
    batch_size: int = 16
    num_workers: int = 4


@dataclass
class Optimizer:
    learning_rate: float = 0.0005
    momentum: float = 0.9
    weight_decay: float = 4e-5
    lr_step_milestones: Iterable = (70, 90)
    lr_gamma: float = 0.1


@dataclass
class Trainer:
    device: str = "cuda:1"
    epoch_num: int = 100
    save_interval: int = 5


@dataclass
class Model:
    encoder = 'se_resnet50'
    encoder_weights = 'imagenet'
    activation = 'softmax2d'
    model_name = 'fpn'
