"""
    Names and fixed transforms used across the benchmark.

    This is the TinyImageNet-only subset of the original enum: the entries for
    MNIST+Omniglot, Panicum, Eucalyptus and CIFAR were dropped, and the absolute
    paths that used to live here now resolve through `Utils.paths`.
"""

from enum import Enum

import torchvision.transforms as transforms

from .paths import FEATURES_ROOT, TINY_IMAGENET_DIR


class NAMES(Enum):

    # ── Model names ───────────────────────────────────────────────────────────
    RESNET18 = "ResNet18"

    # ── Feature extraction (OpenGan reads pre-extracted features from here) ───
    FEATS_DIR = FEATURES_ROOT

    # ── Tiny ImageNet ─────────────────────────────────────────────────────────
    TINY_IMAGE_NET = TINY_IMAGENET_DIR

    TINY_IMAGE_NET_RESNET18_VAL_TEST_TRANSFORMS = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
