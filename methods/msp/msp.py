"""
    Maximum Softmax Probability, the standard open-set baseline.

    No novelty-detection machinery: a closed-set backbone is run as-is and the
    largest softmax probability is compared against a threshold.

        prediction = argmax(softmax(logits))  if  max(softmax(logits)) >= epsilon
                     unknown (-1)             otherwise

    It reuses the stage 1 backbone, so it needs no training of its own, and it
    is the reference every other method here should be expected to beat.
"""

import torch

from Utils import DEVICE


@torch.no_grad()
def collect_msp(model, dataloader):
    """
        One forward pass over the loader.

        Returns the largest softmax probability, the predicted class and the
        label of every sample. None of the three depends on the threshold, so a
        sweep re-labels these arrays instead of repeating the forward pass.
    """
    model.eval()
    msp, preds, labels = [], [], []

    for X, y in dataloader:
        logits = model(X.to(DEVICE))
        if isinstance(logits, (tuple, list)):   # backbones returning (logits, features)
            logits = logits[0]

        max_prob, pred = torch.max(torch.softmax(logits, dim=1), dim=1)

        msp.append(max_prob.cpu())
        preds.append(pred.cpu())
        labels.append(y.cpu())

    return torch.cat(msp), torch.cat(preds), torch.cat(labels)
