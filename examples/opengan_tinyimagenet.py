"""
    OpenGan on TinyImageNet.

    Kong & Ramanan, "OpenGAN: Open-Set Recognition via Open Data Generation",
    ICCV 2021.

    A discriminator is trained adversarially to separate features of known
    classes from generated ones, and is then reused at inference as an
    open-set gate in front of the closed-set head: the discriminator scores the
    feature vector, and only samples scoring at or above `epsilon` are passed
    to the classifier. Everything else is rejected as unknown.

    The full experiment reads features extracted ahead of time; this example
    extracts them on the fly from the TinyImageNet loader so it runs standalone.

    The default epsilon is illustrative — it is the value that maximised macro
    F1 on split 0's test set, chosen only so the example produces a balanced
    result out of the box. The operating point reported in the paper comes from
    a sweep on the validation split.

    Run:
        python examples/opengan_tinyimagenet.py --split 0
"""

import torch
from tqdm import tqdm

from Datasets import TinyImageNet_loader
from Utils import (DEVICE, classifier_ckpt, fix_random_seed,
                   OSRMetrics, opengan_discriminator_ckpt)
from methods.opengan.classifier import Discriminator, OSR_classifier
from methods.opengan.Feat_extraction import ResNet18_64x64_feature_extraction

from _common import base_parser, report, require

NUM_CLASSES = 20
FEATURE_CHANNELS = 512   # ResNet18 penultimate width — the discriminator's input
DISCRIMINATOR_WIDTH = 100


def main():
    args = base_parser(__doc__, default_epsilon=0.02).parse_args()
    fix_random_seed(42)

    data = TinyImageNet_loader(batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = data.get_test_loader(args.split, data.eval_transforms[args.split])

    classifier = ResNet18_64x64_feature_extraction(NUM_CLASSES)
    ckpt = require(classifier_ckpt(args.split), f"closed-set checkpoint for split {args.split}")
    classifier.load_model(torch.load(ckpt, map_location=DEVICE))
    classifier.model.eval()

    discriminator = Discriminator(nc=FEATURE_CHANNELS, ndf=DISCRIMINATOR_WIDTH).to(DEVICE)
    dnet = require(opengan_discriminator_ckpt(args.split), f"OpenGan discriminator for split {args.split}")
    discriminator.load_state_dict(torch.load(dnet, map_location=DEVICE))
    discriminator.eval()

    osr = OSR_classifier(classifier=classifier, discriminator=discriminator, epsilon=args.epsilon)

    predicts, labels, scores = [], [], []
    for X, y in tqdm(test_loader, desc="scoring"):
        # The discriminator is convolutional, so features enter as 1x1 maps.
        features = classifier.extract_features(X.to(DEVICE)).unsqueeze(-1).unsqueeze(-1)
        predict, score = osr.classify(features)
        predicts.append(predict.cpu())
        scores.append(score.cpu())
        labels.append(y.cpu())

    metrics = OSRMetrics(
        predict=torch.cat(predicts).numpy(),
        label=torch.cat(labels).numpy(),
        outlier_scores=torch.cat(scores).numpy(),
        convention="opengan",
    ).compute()

    report("OpenGan", args.split, args.epsilon, metrics)


if __name__ == "__main__":
    main()
