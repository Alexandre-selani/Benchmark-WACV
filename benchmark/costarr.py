"""
    COSTARR sweep on TinyImageNet: five splits x an epsilon sweep.

    The fit stage runs once per split and its statistics are cached to disk, so
    re-running the sweep skips straight to scoring. Scores are then computed
    once and re-thresholded at every epsilon.

    Writes results/costarr/<model>/{Val,Test}/{Val,Test}.csv.

    Run:
        python benchmark/costarr.py
"""

import gc

import torch

from Datasets import TinyImageNet_loader
from Models import ResNet18_tinyimgnet_featurizer
from Utils import (DEVICE, MetricLogger, OSRMetrics, RESULTS_ROOT,
                   classifier_ckpt, fix_random_seed)
from methods.costarr import costarrFit, costarrPredict, threshold_predictions

from _common import (NUM_CLASSES, announce, base_parser, epsilons_from,
                     loader_for, mc_column_names, output_dir,
                     write_best_hyperparameters)

METHOD = "costarr"


def fit_statistics(model, data, split, args):
    """Fits once per split and caches, since the fit does not depend on epsilon."""
    calcs_dir = RESULTS_ROOT / METHOD / "calcs"
    calcs_dir.mkdir(parents=True, exist_ok=True)
    calcs_path = calcs_dir / f"tinyimgnet_split_{split}_costarr.pt"

    if not calcs_path.exists():
        train_loader = data.get_train_loader(split, data.eval_transforms[split])
        costarrFit(model, train_loader, str(calcs_path))
        del train_loader

    return torch.load(calcs_path, weights_only=False)


def main():
    args = base_parser(__doc__, default_start=0.0, default_stop=1.0, default_step=0.01).parse_args()
    fix_random_seed(42)

    epsilons = epsilons_from(args)
    data = TinyImageNet_loader(batch_size=args.batch_size, num_workers=args.num_workers)

    for subset in args.subsets:
        announce("COSTARR", subset, args.splits, epsilons)
        directory = output_dir(METHOD, args.model, subset)
        logger = MetricLogger(epsilons, len(args.splits), str(directory),
                              flag_mc=args.confusion_matrices,
                              mc_column_names=mc_column_names(),
                              mc_title=f"COSTARR - TinyImageNet ({subset})")

        for fold, split in enumerate(args.splits):
            gc.collect()
            torch.cuda.empty_cache()

            model = ResNet18_tinyimgnet_featurizer(num_classes=NUM_CLASSES).to(DEVICE)
            model.load_state_dict(torch.load(classifier_ckpt(split, args.model), map_location=DEVICE))
            model.eval()

            calcs = fit_statistics(model, data, split, args)

            eval_loader = loader_for(data, subset, split, data.eval_transforms[split])
            scores, _, max_logits_idx, labels = costarrPredict(model, eval_loader, calcs)
            labels_np = labels.numpy()
            scores_np = scores.numpy()

            for epsilon in epsilons:
                predicts = threshold_predictions(scores, max_logits_idx, epsilon)
                metrics = OSRMetrics(predict=predicts.numpy(), label=labels_np,
                                     outlier_scores=scores_np,
                                     convention="opengan").compute()
                logger.update(metrics, fold, epsilon)
                if args.confusion_matrices:
                    logger.update_mc(epsilon, predicts.numpy(), labels_np, labels_np)

            print(f"  split {split} done")
            del model, eval_loader, scores

        frame = logger.aggregate(f"{subset.capitalize()}.csv")
        if subset == "val":
            write_best_hyperparameters([({}, frame)], directory)


if __name__ == "__main__":
    main()
