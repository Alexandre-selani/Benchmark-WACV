"""
    OpenMax on TinyImageNet.

    Bendale & Boult, "Towards Open Set Deep Networks", CVPR 2016.

    OpenMax fits a Weibull tail to the distance between each training sample and
    its class mean activation vector, then redistributes logit mass to a
    synthetic unknown class at inference. `tailsize` is how many extreme
    samples per class the Weibull is fitted on and `alpha` how many top classes
    get revised; a sample whose largest revised activation falls below
    `epsilon` is rejected.

    The detector comes from the vendored pytorch_ood under third_party/ — see
    third_party/NOTICE for why it is not the PyPI release.

    Run:
        python examples/openmax_tinyimagenet.py --split 0
"""

import numpy as np
import torch
from osr_pytorch_ood.detector import OpenMax

from Datasets import TinyImageNet_loader
from Models import ResNet18_tinyimgnet
from Utils import DEVICE, classifier_ckpt, fix_random_seed, OSRMetrics

from _common import base_parser, report, require

NUM_CLASSES = 20


def main():
    parser = base_parser(__doc__, default_epsilon=0.1)
    parser.add_argument("--tailsize", type=int, default=100,
                        help="Weibull tail size per class (default: 100)")
    parser.add_argument("--alpha", type=int, default=10,
                        help="how many top classes are revised (default: 10)")
    args = parser.parse_args()
    fix_random_seed(42)

    data = TinyImageNet_loader(batch_size=args.batch_size, num_workers=args.num_workers)
    # Weibull tails are fitted on unaugmented training data.
    fit_loader = data.get_train_loader(args.split, data.eval_transforms[args.split])
    test_loader = data.get_test_loader(args.split, data.eval_transforms[args.split])

    model = ResNet18_tinyimgnet(num_classes=NUM_CLASSES, weights=None)
    ckpt = require(classifier_ckpt(args.split), f"closed-set checkpoint for split {args.split}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.to(DEVICE).eval()

    detector = OpenMax(model, tailsize=args.tailsize, alpha=args.alpha,
                       euclid_weight=1, epsilon=args.epsilon)
    detector.fit(fit_loader, device=str(DEVICE))

    predicts, labels, outlier_scores = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            score = detector(X.to(DEVICE))
            max_values, predicted = torch.max(score, dim=1)
            # Column 0 of the OpenMax output is the synthetic unknown class, so
            # rejected samples are mapped to 0 rather than -1 here.
            predicts.append(
                torch.where(max_values >= detector.epsilon, predicted,
                            torch.zeros_like(predicted)).cpu())
            outlier_scores.append(score[:, 0].cpu())
            labels.append(y.cpu())

    metrics = OSRMetrics(
        predict=torch.cat(predicts).numpy(),
        label=torch.cat(labels).numpy(),
        outlier_scores=torch.cat(outlier_scores).numpy(),
        convention="openmax",
    ).compute()

    report("OpenMax", args.split, args.epsilon, metrics)


if __name__ == "__main__":
    main()
