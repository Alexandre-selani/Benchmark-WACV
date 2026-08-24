"""
    CAC (Class Anchor Clustering) on TinyImageNet.

    Miller et al., "Class Anchor Clustering: A Loss for Distance-based Open Set
    Recognition", WACV 2021.

    CAC classifies by distance to a set of per-class anchors rather than by
    logit magnitude. The anchors are the mean logit vector of the correctly
    classified training samples of each class, so they are recomputed from the
    training split before evaluating. A sample whose smallest anchor distance
    exceeds `epsilon` is rejected as unknown.

    Run:
        python examples/cac_tinyimagenet.py --split 0
"""

import torch

from Datasets import TinyImageNet_loader
from Models import ResNet18_tinyimgnet_cac
from Utils import DEVICE, cac_ckpt, fix_random_seed, OSRMetrics
from methods.cac import find_anchor_means, gather_outputs

from _common import base_parser, report, require

NUM_CLASSES = 20


def main():
    args = base_parser(__doc__, default_epsilon=0.38).parse_args()
    fix_random_seed(42)

    data = TinyImageNet_loader(batch_size=args.batch_size, num_workers=args.num_workers)
    # The anchors describe the training distribution, so they are estimated
    # with the deterministic transform rather than the augmented one.
    train_loader = data.get_train_loader(args.split, data.eval_transforms[args.split])
    test_loader = data.get_test_loader(args.split, data.eval_transforms[args.split])

    model = ResNet18_tinyimgnet_cac(NUM_CLASSES).to(DEVICE)
    ckpt = require(cac_ckpt(args.split), f"CAC checkpoint for split {args.split}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    model.set_anchors(find_anchor_means(model, train_loader, DEVICE, NUM_CLASSES))

    _, distances, labels = gather_outputs(model, test_loader, DEVICE)
    predicts, min_scores, _ = model.predict_by_distance(args.epsilon, distances)

    # Smaller distance means more confidently known, so the outlier score is
    # the negated distance.
    #
    # `convention` selects two conventions at once: the index that marks unknown
    # (-1 here, 0 for "openmax") and whether AUROC uses the score as given or
    # flipped to 1 - score. "opengan" is the branch that takes the score as
    # given, which is what an already-oriented score like this one needs — it
    # is not a statement about which method is running.
    metrics = OSRMetrics(
        predict=predicts.numpy(),
        label=labels.numpy(),
        outlier_scores=-min_scores.numpy(),
        convention="opengan",
    ).compute()

    report("CAC", args.split, args.epsilon, metrics)


if __name__ == "__main__":
    main()
