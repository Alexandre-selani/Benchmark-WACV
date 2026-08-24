"""
    MSP sweep on TinyImageNet: five splits x an epsilon sweep.

    Softmax probabilities are collected once per split and re-thresholded at
    every epsilon.

    Writes results/msp/<model>/{Val,Test}/{Val,Test}.csv.

    Run:
        python benchmark/msp.py
"""

import gc

import torch

from Datasets import TinyImageNet_loader
from Models import ResNet18_tinyimgnet
from Utils import (DEVICE, MetricLogger, OSRMetrics, classifier_ckpt,
                   fix_random_seed)
from methods.msp import collect_msp

from _common import (NUM_CLASSES, announce, base_parser, epsilons_from,
                     loader_for, mc_column_names, output_dir)

METHOD = "msp"


def main():
    args = base_parser(__doc__, default_start=0.0, default_stop=1.0, default_step=0.01).parse_args()
    fix_random_seed(42)

    epsilons = epsilons_from(args)
    data = TinyImageNet_loader(batch_size=args.batch_size, num_workers=args.num_workers)

    for subset in args.subsets:
        announce("MSP", subset, args.splits, epsilons)
        logger = MetricLogger(epsilons, len(args.splits), str(output_dir(METHOD, args.model, subset)),
                              flag_mc=args.confusion_matrices,
                              mc_column_names=mc_column_names(),
                              mc_title=f"MSP - TinyImageNet ({subset})")

        for fold, split in enumerate(args.splits):
            gc.collect()
            torch.cuda.empty_cache()

            model = ResNet18_tinyimgnet(num_classes=NUM_CLASSES, weights=None)
            model.load_state_dict(torch.load(classifier_ckpt(split, args.model), map_location=DEVICE))
            model.to(DEVICE).eval()

            eval_loader = loader_for(data, subset, split, data.eval_transforms[split])
            msp, preds, labels = collect_msp(model, eval_loader)
            labels_np, msp_np = labels.numpy(), msp.numpy()

            for epsilon in epsilons:
                predicts = torch.where(msp >= epsilon, preds, -1)
                metrics = OSRMetrics(predict=predicts.numpy(), label=labels_np,
                                     outlier_scores=msp_np, convention="opengan").compute()
                logger.update(metrics, fold, epsilon)
                if args.confusion_matrices:
                    logger.update_mc(epsilon, predicts.numpy(), labels_np, labels_np)

            print(f"  split {split} done")
            del model, eval_loader, msp

        logger.aggregate(f"{subset.capitalize()}.csv")


if __name__ == "__main__":
    main()
