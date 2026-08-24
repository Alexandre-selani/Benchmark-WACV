"""
    Stage 5 — cache OpenGan features. Requires stage 1.

    OpenGan trains its GAN on penultimate-layer features rather than images, so
    they are extracted once with the stage 1 backbone and cached. Stage 6 needs
    three of these sets: `train` (real features the discriminator learns to
    accept), `kkc_val` and `uuc_val` (known and unknown validation features,
    used to select the best epoch by AUROC).

    Produces:
        $OSR_FEATURES/ResNet18/Split_{i}/{train,kkc_val,uuc_val,...}_features.pt

    Run:
        python training/5_opengan_features.py
"""

import gc

import torch

from Datasets import TinyImageNet_loader
from Utils import DEVICE, classifier_ckpt, features_dir, fix_random_seed
from methods.opengan.Feat_extraction import ResNet18_64x64_feature_extraction

from _common import NUM_CLASSES, base_parser

# Every subset stage 6 or later analysis might want.
SUBSETS = ["train", "val", "test", "kkc_val", "kkc_test", "uuc_val", "uuc_test"]


def main():
    parser = base_parser(__doc__, default_epochs=0, default_lr=0.0, default_batch_size=256)
    # Extraction has neither epochs nor a learning rate; mark the inherited
    # flags so --help does not advertise options that do nothing here.
    for action in parser._actions:
        if action.dest in ("epochs", "lr"):
            action.help = "(unused for feature extraction)"
    args = parser.parse_args()
    fix_random_seed(42)

    data = TinyImageNet_loader(batch_size=args.batch_size, num_workers=args.num_workers)

    for split in args.splits:
        gc.collect()
        torch.cuda.empty_cache()

        model = ResNet18_64x64_feature_extraction(num_classes=NUM_CLASSES)
        ckpt = classifier_ckpt(split)
        if not ckpt.exists():
            raise SystemExit(f"error: stage 1 output missing at {ckpt}\n"
                             f"Run training/1_closedset_backbone.py first.")
        model.load_model(torch.load(ckpt, map_location=DEVICE))

        out_dir = features_dir(split)
        out_dir.mkdir(parents=True, exist_ok=True)

        loaders = {
            "train":    data.get_train_loader(split, data.eval_transforms[split]),
            "val":      data.get_val_loader(split, data.eval_transforms[split]),
            "test":     data.get_test_loader(split, data.eval_transforms[split]),
            "kkc_val":  data.get_val_known_loader(split, data.eval_transforms[split]),
            "kkc_test": data.get_test_known_loader(split, data.eval_transforms[split]),
            "uuc_val":  data.get_val_unknown_loader(split, data.eval_transforms[split]),
            "uuc_test": data.get_test_unknown_loader(split, data.eval_transforms[split]),
        }
        for name in SUBSETS:
            model.save_features(loaders[name], str(out_dir), name)

        del model, loaders


if __name__ == "__main__":
    main()
