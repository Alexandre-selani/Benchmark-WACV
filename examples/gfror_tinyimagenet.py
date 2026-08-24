"""
    GFROR on TinyImageNet.

    Perera et al., "Generative-Discriminative Feature Representations for Open-Set
    Recognition", CVPR 2020.

    A class-conditioned autoencoder reconstructs the input; the reconstruction
    is concatenated to the original along the channel axis and fed to a
    six-channel classifier. Known inputs reconstruct well and produce a large
    peak activation, so a sample whose largest activation falls below `epsilon`
    is rejected.

    Note the autoencoder is trained on unnormalized [0, 1] inputs, so this
    example builds its own transform instead of using the split's normalized
    one.

    The default epsilon is illustrative, chosen near the median peak activation
    so the example produces a balanced result. The operating point reported in
    the paper comes from a sweep on the validation split.

    Run:
        python examples/gfror_tinyimagenet.py --split 0
"""

import torch
import torchvision.transforms as T
from tqdm import tqdm

from Datasets import TinyImageNet_loader
from Utils import (DEVICE, fix_random_seed, gfror_classifier_ckpt,
                   gfror_generator_ckpt, OSRMetrics)
# Imported for its side effect: registers the import aliases the pickled
# GFROR checkpoints need. See methods/gfror/_compat.py.
import methods.gfror  # noqa: F401

from _common import base_parser, report, require

IMAGE_SIZE = 64


def predict(dataloader, generator, classifier):
    generator.eval()
    classifier.eval()

    max_acts, indices, unknown_probs, targets = [], [], [], []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="scoring"):
            x = x.to(DEVICE)
            out = classifier(torch.cat((x, generator(x)), dim=1))[0]

            max_act, idx = torch.max(out, dim=-1)
            z = torch.exp(out).sum(dim=1)
            prob_known = z / (z + 1)

            max_acts.append(max_act.cpu())
            indices.append(idx.cpu())
            unknown_probs.append((1 - prob_known).cpu())
            targets.append(y.cpu())

    return (torch.cat(max_acts), torch.cat(indices),
            torch.cat(unknown_probs), torch.cat(targets))


def main():
    args = base_parser(__doc__, default_epsilon=6.0).parse_args()
    fix_random_seed(42)

    # No normalization: the autoencoder was trained on [0, 1] inputs.
    transform = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])
    data = TinyImageNet_loader(batch_size=args.batch_size, num_workers=args.num_workers,
                               image_size=IMAGE_SIZE)
    test_loader = data.get_test_loader(args.split, transform)

    # These checkpoints are pickled modules rather than state_dicts, which is
    # why they are loaded with weights_only=False.
    generator = torch.load(
        require(gfror_generator_ckpt(args.split), f"GFROR generator for split {args.split}"),
        weights_only=False, map_location=DEVICE).to(DEVICE)
    classifier = torch.load(
        require(gfror_classifier_ckpt(args.split), f"GFROR classifier for split {args.split}"),
        weights_only=False, map_location=DEVICE).to(DEVICE)

    max_act, preds, unknown_score, labels = predict(test_loader, generator, classifier)
    predicts = torch.where(max_act < args.epsilon, -1, preds)

    metrics = OSRMetrics(
        predict=predicts.numpy(), label=labels.numpy(),
        outlier_scores=-unknown_score.numpy(), convention="opengan",
    ).compute()

    report("GFROR", args.split, args.epsilon, metrics)


if __name__ == "__main__":
    main()
