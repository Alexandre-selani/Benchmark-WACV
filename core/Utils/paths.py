"""
    Central path configuration for the whole benchmark.

    Every path used by the five methods resolves through this module, so a
    fresh clone runs without editing source files. Each entry falls back to a
    location inside the repository and can be overridden with an environment
    variable, which is what you want when the dataset or the checkpoints live
    on a different disk:

        export OSR_TINY_IMAGENET_DIR=/mnt/datasets/tiny-imagenet-200
        export OSR_CHECKPOINTS=/mnt/scratch/checkpoints
"""

import os
from pathlib import Path

# WACV/core/Utils/paths.py -> WACV/
REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(var, default):
    return Path(os.environ.get(var, default)).expanduser()


DATA_ROOT = _env_path("OSR_DATA_ROOT", REPO_ROOT / "data")

TINY_IMAGENET_DIR = _env_path("OSR_TINY_IMAGENET_DIR", DATA_ROOT / "tiny-imagenet-200")
# Committed to the repository: 24 KB that define the entire open-set
# protocol (which 20 of the 200 classes are known in each split).
SPLITS_DIR = _env_path("OSR_SPLITS_DIR", REPO_ROOT / "splits")

CHECKPOINTS_ROOT = _env_path("OSR_CHECKPOINTS", REPO_ROOT / "checkpoints")
RESULTS_ROOT = _env_path("OSR_RESULTS", REPO_ROOT / "results")
FEATURES_ROOT = _env_path("OSR_FEATURES", REPO_ROOT / "features")


def classifier_ckpt(split, model="ResNet18"):
    """Closed-set backbone for one split — shared by OpenMax, COSTARR and OpenGan."""
    return CHECKPOINTS_ROOT / model / f"Split_{split}" / f"{model}_TinyImageNet_split_{split}.pt"


def cac_ckpt(split, model="ResNet18"):
    """CAC-trained backbone for one split."""
    return CHECKPOINTS_ROOT / model / f"Split_{split}" / f"{model}_TinyImageNet_cac_split_{split}.pt"


def results_dir(method, model="ResNet18"):
    d = RESULTS_ROOT / method / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def features_dir(split, model="ResNet18"):
    return FEATURES_ROOT / model / f"Split_{split}"


def gfror_generator_ckpt(split):
    """GFROR autoencoder for one split (a pickled module, not a state_dict)."""
    return CHECKPOINTS_ROOT / "gfror" / "ae_tinyimgnet" / f"split_{split}.pth"


def gfror_classifier_ckpt(split, model="ResNet18"):
    """GFROR open-set classifier for one split (a pickled module, not a state_dict)."""
    return CHECKPOINTS_ROOT / "gfror" / "openset_ae_tinyimgnet" / model / f"Split_{split}" / "ckpt.pth"


def opengan_discriminator_ckpt(split, model="ResNet18"):
    """OpenGan discriminator selected on the validation split."""
    return CHECKPOINTS_ROOT / "opengan" / model / f"Fold+{split}" / "best_epoch.DNet"
