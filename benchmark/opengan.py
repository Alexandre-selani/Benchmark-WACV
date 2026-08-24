"""
    OpenGan sweep on TinyImageNet: five splits x an epsilon sweep.

    Both quantities the decision rests on — the discriminator's confidence and
    the closed-set prediction — are independent of epsilon, so each split is
    scored once and epsilon only moves the accept/reject boundary.

    Features are extracted on the fly from the TinyImageNet loader, so this does
    not depend on the cached features that training stage 5 produces.

    Writes results/opengan/<model>/{Val,Test}/{Val,Test}.csv.

    Run:
        python benchmark/opengan.py
"""

import gc

import torch
import torch.nn.functional as F
from tqdm import tqdm

from Datasets import TinyImageNet_loader
from Utils import (DEVICE, MetricLogger, OSRMetrics, classifier_ckpt,
                   fix_random_seed, opengan_discriminator_ckpt)
from methods.opengan.classifier import Discriminator
from methods.opengan.Feat_extraction import ResNet18_64x64_feature_extraction

from _common import (NUM_CLASSES, announce, base_parser, epsilons_from,
                     loader_for, mc_column_names, output_dir)

METHOD = "opengan"
FEATURE_CHANNELS = 512    # ResNet18 penultimate width — the discriminator's input
DISCRIMINATOR_WIDTH = 100


def score(dataloader, classifier, discriminator):
    """Returns the discriminator confidence, the closed-set prediction and the labels."""
    likelihoods, predictions, targets = [], [], []

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="  scoring", leave=False):
            # The discriminator is convolutional, so features enter as 1x1 maps.
            features = classifier.extract_features(X.to(DEVICE)).unsqueeze(-1).unsqueeze(-1)
            likelihoods.append(discriminator(features).view(-1).cpu())

            flat = features.view(features.size(0), -1)
            logits = classifier.classify_features(flat)
            predictions.append(torch.argmax(F.softmax(logits, dim=1), dim=1).cpu())
            targets.append(y.cpu())

    return torch.cat(likelihoods), torch.cat(predictions), torch.cat(targets)


def main():
    args = base_parser(__doc__, default_start=0.0, default_stop=1.0, default_step=0.01).parse_args()
    fix_random_seed(42)

    epsilons = epsilons_from(args)
    data = TinyImageNet_loader(batch_size=args.batch_size, num_workers=args.num_workers)

    for subset in args.subsets:
        announce("OpenGan", subset, args.splits, epsilons)
        logger = MetricLogger(epsilons, len(args.splits), str(output_dir(METHOD, args.model, subset)),
                              flag_mc=args.confusion_matrices,
                              mc_column_names=mc_column_names(),
                              mc_title=f"OpenGan - TinyImageNet ({subset})")

        for fold, split in enumerate(args.splits):
            gc.collect()
            torch.cuda.empty_cache()

            classifier = ResNet18_64x64_feature_extraction(NUM_CLASSES)
            classifier.load_model(torch.load(classifier_ckpt(split, args.model), map_location=DEVICE))
            classifier.model.eval()

            discriminator = Discriminator(nc=FEATURE_CHANNELS, ndf=DISCRIMINATOR_WIDTH).to(DEVICE)
            discriminator.load_state_dict(
                torch.load(opengan_discriminator_ckpt(split, args.model), map_location=DEVICE))
            discriminator.eval()

            eval_loader = loader_for(data, subset, split, data.eval_transforms[split])
            likelihood, predicted, labels = score(eval_loader, classifier, discriminator)
            labels_np = labels.numpy()
            likelihood_np = likelihood.numpy()

            for epsilon in epsilons:
                predicts = torch.where(likelihood >= epsilon, predicted, -1)
                metrics = OSRMetrics(predict=predicts.numpy(), label=labels_np,
                                     outlier_scores=likelihood_np,
                                     convention="opengan").compute()
                logger.update(metrics, fold, epsilon)
                if args.confusion_matrices:
                    logger.update_mc(epsilon, predicts.numpy(), labels_np, labels_np)

            print(f"  split {split} done")
            del classifier, discriminator, eval_loader, likelihood

        logger.aggregate(f"{subset.capitalize()}.csv")


if __name__ == "__main__":
    main()
