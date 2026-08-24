"""
    COSTARR on TinyImageNet.

    COSTARR scores a sample by combining its activation with the ratio between
    its feature vector and the per-class mean feature vector learned from the
    training split. The fit stage measures those class statistics once and
    caches them; the predict stage turns them into a single score per sample,
    which `epsilon` thresholds into known/unknown.

    Run:
        python examples/costarr_tinyimagenet.py --split 0
"""

import torch

from Datasets import TinyImageNet_loader
from Models import ResNet18_tinyimgnet_featurizer
from Utils import DEVICE, RESULTS_ROOT, classifier_ckpt, fix_random_seed, OSRMetrics
from methods.costarr import costarrFit, costarrPredict, threshold_predictions

from _common import base_parser, report, require

NUM_CLASSES = 20


def main():
    args = base_parser(__doc__, default_epsilon=0.5).parse_args()
    fix_random_seed(42)

    data = TinyImageNet_loader(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = data.get_train_loader(args.split, data.eval_transforms[args.split])
    test_loader = data.get_test_loader(args.split, data.eval_transforms[args.split])

    model = ResNet18_tinyimgnet_featurizer(num_classes=NUM_CLASSES).to(DEVICE)
    ckpt = require(classifier_ckpt(args.split), f"closed-set checkpoint for split {args.split}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    # Fit caches the per-class statistics to disk, mirroring how the full
    # experiment reuses one fit across many epsilon values.
    calcs_dir = RESULTS_ROOT / "costarr" / "calcs"
    calcs_dir.mkdir(parents=True, exist_ok=True)
    calcs_path = calcs_dir / f"tinyimgnet_split_{args.split}_costarr.pt"
    costarrFit(model, train_loader, str(calcs_path))

    calcs = torch.load(calcs_path, weights_only=False)
    scores, _, max_logits_idx, labels = costarrPredict(model, test_loader, calcs)
    predicts = threshold_predictions(scores, max_logits_idx, args.epsilon)

    metrics = OSRMetrics(
        predict=predicts, label=labels, outlier_scores=scores, convention="opengan",
    ).compute()

    report("COSTARR", args.split, args.epsilon, metrics)


if __name__ == "__main__":
    main()
