"""
    Shared plumbing for the training scripts.

    Same idea as examples/_common.py: only what would otherwise be repeated in
    every script lives here — argument parsing and the strong augmentation
    pipeline the two backbones share.
"""

import argparse

import torchvision.transforms as transforms

IMAGE_SIZE = 64
NUM_CLASSES = 20   # known classes per split, standard TinyImageNet OSR protocol
N_SPLITS = 5


def base_parser(description, default_epochs, default_lr, default_batch_size):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--splits", type=int, nargs="+", default=list(range(N_SPLITS)),
                        metavar="N", help="splits to train (default: all five)")
    parser.add_argument("--epochs", type=int, default=default_epochs,
                        help=f"epochs per split (default: {default_epochs})")
    parser.add_argument("--lr", type=float, default=default_lr,
                        help=f"learning rate (default: {default_lr})")
    parser.add_argument("--batch-size", type=int, default=default_batch_size,
                        help=f"batch size (default: {default_batch_size})")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="dataloader workers (default: 4)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite checkpoints that already exist")
    return parser


def guard_output(path, force):
    """
        Refuses to overwrite an existing checkpoint unless --force was passed.

        Worth the friction twice over. Retraining a split costs hours, and a
        checkpoint directory assembled with symlinks — which is a convenient way
        to point at weights stored elsewhere — would let a write here destroy
        the original file outside the repository.
    """
    if path.exists() and not force:
        target = f"\n    (a symlink to {path.resolve()})" if path.is_symlink() else ""
        raise SystemExit(
            f"error: refusing to overwrite an existing checkpoint\n"
            f"    {path}{target}\n\n"
            f"Pass --force to overwrite it, or point OSR_CHECKPOINTS somewhere else."
        )
    return path


def strong_train_transform(mean, std, image_size=IMAGE_SIZE):
    """
        Heavier than the loader's default train transform, and used only for the
        two backbones: with 10k training images over 20 classes, the light crop
        (0.8-1.0) plus flip was not enough regularization.

        Reuses the split's mean/std so normalization stays consistent with
        validation.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25),
    ])


def build_scheduler(optimizer, epochs, warmup_epochs):
    """
        AdamW from scratch (freshly initialized stem, no pretraining) tends to
        take noisy, oversized steps in the first epochs; a short linear warmup
        before the cosine decay avoids that initial instability.
    """
    import torch.optim as optim
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
