"""Seeding and the CAC loss."""

import random

import numpy as np
import torch


def fix_random_seed(seed: int = 12345) -> None:
    """
    Set all random seeds.

    :param seed: seed to set
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # Stops CuDNN from benchmarking for the fastest algorithm, which would make
    # results depend on which algorithm it happened to pick.
    torch.backends.cudnn.benchmark = False


def CACLoss(distances, gt, num_classes, lbda):
    """Returns CAC loss, as well as the Anchor and Tuplet loss components separately for visualisation."""
    true = torch.gather(distances, 1, gt.view(-1, 1)).view(-1)
    non_gt = torch.Tensor([[i for i in range(num_classes) if gt[x] != i]
                           for x in range(len(distances))]).long().to(distances.device)
    others = torch.gather(distances, 1, non_gt)

    anchor = torch.mean(true)

    tuplet = torch.exp(-others + true.unsqueeze(1))
    tuplet = torch.mean(torch.log(1 + torch.sum(tuplet, dim=1)))

    total = lbda * anchor + tuplet

    return total, anchor, tuplet
