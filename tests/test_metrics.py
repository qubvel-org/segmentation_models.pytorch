import pytest
import torch
import segmentation_models_pytorch as smp

from segmentation_models_pytorch.metrics import get_stats


gpu_device = torch.device('cuda:0')
cpu_device = torch.device('cpu')


def test_get_stats_multiclass_input_gpu_output_no_device():
    size = (10, 32, 32)
    output = torch.randint(0, 5, size, device=gpu_device)
    target = torch.randint(0, 5, size, device=gpu_device)

    stats = get_stats(output, target, mode='multiclass', num_classes=5)

    for s in stats:
        # default to cpu
        s.device == cpu_device


def test_get_stats_multiclass_input_gpu_output_cpu():
    size = (10, 32, 32)
    output = torch.randint(0, 5, size, device=gpu_device)
    target = torch.randint(0, 5, size, device=gpu_device)

    stats = get_stats(output, target, mode='multiclass',
                      num_classes=5, output_device=cpu_device)

    for s in stats:
        s.device == cpu_device


def test_get_stats_multiclass_input_gpu_output_gpu():
    size = (10, 32, 32)
    output = torch.randint(0, 5, size, device=gpu_device)
    target = torch.randint(0, 5, size, device=gpu_device)

    stats = get_stats(output, target, mode='multiclass',
                      num_classes=5, output_device=gpu_device)

    for s in stats:
        s.device == gpu_device


def test_get_stats_multiclass_input_cpu_output_no_device():
    size = (10, 32, 32)
    output = torch.randint(0, 5, size, device=cpu_device)
    target = torch.randint(0, 5, size, device=cpu_device)

    stats = get_stats(output, target, mode='multiclass', num_classes=5)

    for s in stats:
        # default to cpu
        s.device == cpu_device


def test_get_stats_multiclass_input_cpu_output_cpu():
    size = (10, 32, 32)
    output = torch.randint(0, 5, size, device=cpu_device)
    target = torch.randint(0, 5, size, device=cpu_device)

    stats = get_stats(output, target, mode='multiclass',
                      num_classes=5, output_device=cpu_device)

    for s in stats:
        s.device == cpu_device


def test_get_stats_multiclass_input_cpu_output_gpu():
    size = (10, 32, 32)
    output = torch.randint(0, 5, size, device=cpu_device)
    target = torch.randint(0, 5, size, device=cpu_device)

    stats = get_stats(output, target, mode='multiclass',
                      num_classes=5, output_device=gpu_device)

    for s in stats:
        s.device == gpu_device


def test_get_stats_multilabel_input_gpu_ouput_no_device():
    size = (10, 32, 32)
    output = torch.randint(0, 5, size, device=gpu_device)
    target = torch.randint(0, 5, size, device=gpu_device)

    stats = get_stats(output, target, mode='multilabel',
                      num_classes=5)

    for s in stats:
        s.device == gpu_device


def test_get_stats_multilabel_input_gpu_ouput_gpu():
    size = (10, 32, 32)
    output = torch.randint(0, 5, size, device=gpu_device)
    target = torch.randint(0, 5, size, device=gpu_device)

    stats = get_stats(output, target, mode='multilabel',
                      num_classes=5, output_device=gpu_device)

    for s in stats:
        s.device == gpu_device


def test_get_stats_multilabel_input_gpu_ouput_cpu():
    size = (10, 32, 32)
    output = torch.randint(0, 5, size, device=gpu_device)
    target = torch.randint(0, 5, size, device=gpu_device)

    stats = get_stats(output, target, mode='multilabel',
                      num_classes=5, output_device=cpu_device)

    for s in stats:
        s.device == cpu_device
