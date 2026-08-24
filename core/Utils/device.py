"""Single place that decides which device the benchmark runs on."""

import os

import torch


def get_device():
    """
        Honours OSR_DEVICE when set (e.g. "cpu", "cuda:1"), otherwise picks CUDA
        when it is available and falls back to CPU. Falling back matters for the
        artifact: the whole benchmark stays runnable on a machine without a GPU.
    """
    requested = os.environ.get("OSR_DEVICE")
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = get_device()
