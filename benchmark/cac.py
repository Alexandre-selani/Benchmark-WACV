"""
    CAC sweep on TinyImageNet: five splits x an epsilon sweep.

    Anchor distances are gathered once per split, then re-thresholded at every
    epsilon, so the sweep costs about one evaluation per split.

    Writes results/cac/<model>/{Val,Test}/{Val,Test}.csv with mean and standard
    deviation per epsilon across the splits, plus per-fold CSVs under Folds/.

    Run:
        python benchmark/cac.py
        python benchmark/cac.py --splits 0 --subsets test --epsilon-step 0.01
"""

import gc

import torch

from Datasets import TinyImageNet_loader
from Models import ResNet18_tinyimgnet_cac
from Utils import DEVICE, MetricLogger, OSRMetrics, cac_ckpt, fix_random_seed
from methods.cac import find_anchor_means, gather_outputs

from _common import (NUM_CLASSES, announce, base_parser, epsilons_from,
                     loader_for, mc_column_names, output_dir,
                     write_best_hyperparameters)

METHOD = "cac"


def main():
    args = base_parser(__doc__, default_start=0.0, default_stop=1.0, default_step=0.001).parse_args()
    fix_random_seed(42)

    epsilons = epsilons_from(args)
    data = TinyImageNet_loader(batch_size=args.batch_size, num_workers=args.num_workers)

    for subset in args.subsets:
        announce("CAC", subset, args.splits, epsilons)
        directory = output_dir(METHOD, args.model, subset)
        logger = MetricLogger(epsilons, len(args.splits), str(directory),
                              flag_mc=args.confusion_matrices,
                              mc_column_names=mc_column_names(),
                              mc_title=f"CAC - TinyImageNet ({subset})")

        for fold, split in enumerate(args.splits):
            gc.collect()
            torch.cuda.empty_cache()

            train_loader = data.get_train_loader(split, data.eval_transforms[split])
            eval_loader = loader_for(data, subset, split, data.eval_transforms[split])

            model = ResNet18_tinyimgnet_cac(NUM_CLASSES).to(DEVICE)
            model.load_state_dict(torch.load(cac_ckpt(split, args.model), map_location=DEVICE))
            model.set_anchors(find_anchor_means(model, train_loader, DEVICE, NUM_CLASSES))

            # Scored once; the epsilon loop below only re-thresholds.
            _, distances, labels = gather_outputs(model, eval_loader, DEVICE)
            labels_np = labels.numpy()

            for epsilon in epsilons:
                predicts, min_scores, _ = model.predict_by_distance(epsilon, distances)
                metrics = OSRMetrics(predict=predicts.numpy(), label=labels_np,
                                     outlier_scores=-min_scores.numpy(),
                                     convention="opengan").compute()
                logger.update(metrics, fold, epsilon)
                if args.confusion_matrices:
                    logger.update_mc(epsilon, predicts.numpy(), labels_np, labels_np)

            print(f"  split {split} done")
            del model, train_loader, eval_loader, distances

        frame = logger.aggregate(f"{subset.capitalize()}.csv")
        if subset == "val":
            write_best_hyperparameters([({}, frame)], directory)


if __name__ == "__main__":
    main()
