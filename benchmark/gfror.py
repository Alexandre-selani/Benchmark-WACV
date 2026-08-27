"""
    GFROR sweep on TinyImageNet: five splits x an epsilon sweep.

    Peak activations are gathered once per split, then re-thresholded at every
    epsilon. The threshold acts on the raw activation rather than a probability,
    so the sweep runs over a wider range than the other methods.

    Writes results/gfror/<model>/{Val,Test}/{Val,Test}.csv.

    Run:
        python benchmark/gfror.py
"""

import gc

import torch
import torchvision.transforms as T
from tqdm import tqdm

from Datasets import TinyImageNet_loader
from Utils import (DEVICE, MetricLogger, OSRMetrics, fix_random_seed,
                   gfror_classifier_ckpt, gfror_generator_ckpt)
import methods.gfror  # noqa: F401  — registers the aliases the pickled checkpoints need

from _common import (announce, base_parser, epsilons_from, loader_for,
                     mc_column_names, output_dir,
                     write_best_hyperparameters)

METHOD = "gfror"
IMAGE_SIZE = 64


def score(dataloader, generator, classifier):
    generator.eval()
    classifier.eval()

    max_acts, indices, unknown_probs, targets = [], [], [], []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="  scoring", leave=False):
            x = x.to(DEVICE)
            out = classifier(torch.cat((x, generator(x)), dim=1))[0]

            max_act, idx = torch.max(out, dim=-1)
            z = torch.exp(out).sum(dim=1)

            max_acts.append(max_act.cpu())
            indices.append(idx.cpu())
            unknown_probs.append((1 - z / (z + 1)).cpu())
            targets.append(y.cpu())

    return (torch.cat(max_acts), torch.cat(indices),
            torch.cat(unknown_probs), torch.cat(targets))


def main():
    args = base_parser(__doc__, default_start=0.0, default_stop=30.0, default_step=0.2).parse_args()
    fix_random_seed(42)

    epsilons = epsilons_from(args)
    # No normalization: the autoencoder was trained on [0, 1] inputs.
    transform = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])
    data = TinyImageNet_loader(batch_size=args.batch_size, num_workers=args.num_workers,
                               image_size=IMAGE_SIZE)

    for subset in args.subsets:
        announce("GFROR", subset, args.splits, epsilons)
        directory = output_dir(METHOD, args.model, subset)
        logger = MetricLogger(epsilons, len(args.splits), str(directory),
                              flag_mc=args.confusion_matrices,
                              mc_column_names=mc_column_names(),
                              mc_title=f"GFROR - TinyImageNet ({subset})")

        for fold, split in enumerate(args.splits):
            gc.collect()
            torch.cuda.empty_cache()

            generator = torch.load(gfror_generator_ckpt(split), weights_only=False,
                                   map_location=DEVICE).to(DEVICE)
            classifier = torch.load(gfror_classifier_ckpt(split, args.model), weights_only=False,
                                    map_location=DEVICE).to(DEVICE)

            eval_loader = loader_for(data, subset, split, transform)
            max_act, preds, unknown_score, labels = score(eval_loader, generator, classifier)
            labels_np = labels.numpy()
            outlier_np = -unknown_score.numpy()

            for epsilon in epsilons:
                predicts = torch.where(max_act < epsilon, -1, preds)
                metrics = OSRMetrics(predict=predicts.numpy(), label=labels_np,
                                     outlier_scores=outlier_np,
                                     convention="opengan").compute()
                logger.update(metrics, fold, epsilon)
                if args.confusion_matrices:
                    logger.update_mc(epsilon, predicts.numpy(), labels_np, labels_np)

            print(f"  split {split} done")
            del generator, classifier, eval_loader, max_act

        frame = logger.aggregate(f"{subset.capitalize()}.csv")
        if subset == "val":
            write_best_hyperparameters([({}, frame)], directory)


if __name__ == "__main__":
    main()
