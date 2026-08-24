"""
    MSP (Maximum Softmax Probability) on TinyImageNet.

    Hendrycks & Gimpel, "A Baseline for Detecting Misclassified and
    Out-of-Distribution Examples in Neural Networks", ICLR 2017.

    The open-set baseline: the closed-set backbone is used unchanged, and a
    sample whose largest softmax probability falls below `epsilon` is rejected
    as unknown. Reuses the stage 1 checkpoint, so it trains nothing of its own.

    Run:
        python examples/msp_tinyimagenet.py --split 0
"""

import torch

from Datasets import TinyImageNet_loader
from Models import ResNet18_tinyimgnet
from Utils import DEVICE, OSRMetrics, classifier_ckpt, fix_random_seed
from methods.msp import collect_msp

from _common import base_parser, report, require

NUM_CLASSES = 20


def main():
    args = base_parser(__doc__, default_epsilon=0.5).parse_args()
    fix_random_seed(42)

    data = TinyImageNet_loader(batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = data.get_test_loader(args.split, data.eval_transforms[args.split])

    model = ResNet18_tinyimgnet(num_classes=NUM_CLASSES, weights=None)
    ckpt = require(classifier_ckpt(args.split), f"closed-set checkpoint for split {args.split}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.to(DEVICE).eval()

    msp, preds, labels = collect_msp(model, test_loader)
    predicts = torch.where(msp >= args.epsilon, preds, -1)

    # The MSP is already oriented so that higher means "more likely known",
    # which is what the "opengan" convention expects.
    metrics = OSRMetrics(predict=predicts.numpy(), label=labels.numpy(),
                         outlier_scores=msp.numpy(), convention="opengan").compute()

    report("MSP", args.split, args.epsilon, metrics)


if __name__ == "__main__":
    main()
