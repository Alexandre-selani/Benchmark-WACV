"""
    OpenMax sweep on TinyImageNet: a tailsize x alpha grid, each point swept
    over epsilon, across five splits.

    Unlike the other four methods, OpenMax cannot reuse one scoring pass for
    the whole sweep: tailsize and alpha change the Weibull fit itself, so the
    detector is refit at every grid point. Only the epsilon axis is free, and
    the scores of one fit are re-thresholded across it.

    Cost is therefore len(tailsizes) x len(alphas) x len(splits) fits, each
    walking the training set once. The defaults come to 125 fits.

    Writes one CSV per grid point under
        results/openmax/<model>/{Val,Test}/tail_<t>_alpha_<a>/
    plus grid_summary.csv, which names the best epsilon of each grid point by
    mean macro F1 so the grid can be read at a glance.

    Run:
        python benchmark/openmax.py
        python benchmark/openmax.py --tailsizes 100 --alphas 10   # one grid point
"""

import gc

import numpy as np
import pandas as pd
import torch
from pytorch_ood.detector import OpenMax
from tqdm import tqdm

from Datasets import TinyImageNet_loader
from Models import ResNet18_tinyimgnet
from Utils import (DEVICE, MetricLogger, OSRMetrics, classifier_ckpt,
                   fix_random_seed)

from _common import (NUM_CLASSES, announce, base_parser, epsilons_from,
                     loader_for, mc_column_names, output_dir)

METHOD = "openmax"


def score(detector, dataloader):
    """OpenMax activations and labels; epsilon is applied afterwards."""
    scores, targets = [], []
    detector.model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="    scoring", leave=False):
            scores.append(detector(X.to(DEVICE)).cpu())
            targets.append(y.cpu())
    return torch.cat(scores), torch.cat(targets)


def main():
    parser = base_parser(__doc__, default_start=0.0, default_stop=1.0, default_step=0.2)
    parser.add_argument("--tailsizes", type=int, nargs="+", default=[0, 50, 100, 150, 200],
                        metavar="N", help="Weibull tail sizes to try (default: 0 50 100 150 200)")
    parser.add_argument("--alphas", type=int, nargs="+", default=[1, 5, 10, 15, 20],
                        metavar="N", help="how many top classes get revised (default: 1 5 10 15 20)")
    args = parser.parse_args()
    fix_random_seed(42)

    epsilons = epsilons_from(args)
    data = TinyImageNet_loader(batch_size=args.batch_size, num_workers=args.num_workers)

    for subset in args.subsets:
        announce("OpenMax", subset, args.splits, epsilons)
        print(f"  grid: {len(args.tailsizes)} tailsizes x {len(args.alphas)} alphas "
              f"x {len(args.splits)} splits = "
              f"{len(args.tailsizes) * len(args.alphas) * len(args.splits)} fits")

        subset_root = output_dir(METHOD, args.model, subset)
        summary_rows = []

        for tail in args.tailsizes:
            for alpha in args.alphas:
                point_dir = subset_root / f"tail_{tail}_alpha_{alpha}"
                point_dir.mkdir(parents=True, exist_ok=True)

                logger = MetricLogger(epsilons, len(args.splits), str(point_dir),
                                      flag_mc=args.confusion_matrices,
                                      mc_column_names=mc_column_names(),
                                      mc_title=f"OpenMax t={tail} a={alpha} ({subset})")

                for fold, split in enumerate(args.splits):
                    gc.collect()
                    torch.cuda.empty_cache()

                    model = ResNet18_tinyimgnet(num_classes=NUM_CLASSES, weights=None)
                    model.load_state_dict(torch.load(classifier_ckpt(split, args.model),
                                                     map_location=DEVICE))
                    model.to(DEVICE).eval()

                    # Weibull tails are fitted on unaugmented training data.
                    fit_loader = data.get_train_loader(split, data.eval_transforms[split])
                    detector = OpenMax(model, tailsize=tail, alpha=alpha,
                                       euclid_weight=1, epsilon=epsilons[0])
                    detector.fit(fit_loader, device=str(DEVICE))

                    eval_loader = loader_for(data, subset, split, data.eval_transforms[split])
                    activations, labels = score(detector, eval_loader)
                    labels_np = labels.numpy()
                    # Column 0 is the synthetic unknown class.
                    outlier_np = activations[:, 0].numpy()
                    max_values, predicted = torch.max(activations, dim=1)

                    for epsilon in epsilons:
                        # Rejected samples map to 0, matching the openmax convention.
                        predicts = torch.where(max_values >= epsilon, predicted,
                                               torch.zeros_like(predicted))
                        metrics = OSRMetrics(predict=predicts.numpy(), label=labels_np,
                                             outlier_scores=outlier_np,
                                             convention="openmax").compute()
                        logger.update(metrics, fold, epsilon)
                        if args.confusion_matrices:
                            logger.update_mc(epsilon, predicts.numpy(), labels_np, labels_np)

                    del model, detector, fit_loader, eval_loader, activations

                logger.aggregate(f"{subset.capitalize()}.csv")

                # Best epsilon of this grid point, for the summary.
                df = pd.read_csv(point_dir / f"{subset.capitalize()}.csv")
                best = df.loc[df["F1 macro_mean"].idxmax()]
                summary_rows.append({
                    "tailsize": tail, "alpha": alpha, "epsilon": best["epsilon"],
                    "F1 macro_mean": best["F1 macro_mean"], "F1 macro_std": best["F1 macro_std"],
                    "accuracy_mean": best["accuracy_mean"], "auroc_mean": best["auroc_mean"],
                })
                print(f"  tail {tail:>3} alpha {alpha:>2} -> best F1 "
                      f"{best['F1 macro_mean']:.3f} at epsilon {best['epsilon']}")

        summary = pd.DataFrame(summary_rows).sort_values("F1 macro_mean", ascending=False)
        summary_path = subset_root / "grid_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
