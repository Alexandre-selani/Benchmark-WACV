"""
    Stage 1 — closed-set ResNet18 backbone.

    Trains one backbone per split on the known classes only. This checkpoint is
    the starting point for three of the five methods: OpenMax fits its Weibull
    tails on it, COSTARR measures its class statistics from it, and OpenGan
    extracts features with it.

    Produces:
        $OSR_CHECKPOINTS/ResNet18/Split_{i}/ResNet18_TinyImageNet_split_{i}.pt

    Run:
        python training/1_closedset_backbone.py
        python training/1_closedset_backbone.py --splits 0 --epochs 10   # quick check
"""

import copy
import gc

import torch
import torch.nn as nn
import torch.optim as optim

from Datasets import TinyImageNet_loader
from Models import ResNet18_tinyimgnet
from Utils import (DEVICE, TrainingCurves, CHECKPOINTS_ROOT, eval,
                   fix_random_seed, train)

from _common import (NUM_CLASSES, base_parser, build_scheduler, guard_output,
                     strong_train_transform)

WARMUP_EPOCHS = 5


def main():
    args = base_parser(__doc__, default_epochs=100, default_lr=8e-4,
                       default_batch_size=256).parse_args()
    fix_random_seed(42)

    save_root = CHECKPOINTS_ROOT / "ResNet18"
    save_root.mkdir(parents=True, exist_ok=True)

    data = TinyImageNet_loader(batch_size=args.batch_size, num_workers=args.num_workers)

    for split in args.splits:
        gc.collect()
        torch.cuda.empty_cache()

        split_dir = save_root / f"Split_{split}"
        split_dir.mkdir(parents=True, exist_ok=True)
        out = guard_output(split_dir / f"ResNet18_TinyImageNet_split_{split}.pt", args.force)
        curves = TrainingCurves(f"ResNet18_Split_{split}", "TinyImageNet", dir=str(split_dir))

        model = ResNet18_tinyimgnet(num_classes=NUM_CLASSES, weights=None).to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        scheduler = build_scheduler(optimizer, args.epochs, WARMUP_EPOCHS)

        mean, std = data.norm_stats[split]
        train_loader = data.get_train_loader(split, strong_train_transform(mean, std))
        # Model selection uses known-class validation only: the unknowns must
        # not influence training in any way.
        val_loader = data.get_val_known_loader(split, data.eval_transforms[split])

        best_val_acc, best_state = 0.0, None
        for epoch in range(args.epochs):
            train_loss, train_acc = train(train_loader, model, criterion, optimizer)
            val_loss, val_acc = eval(val_loader, model, criterion)
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc, best_state = val_acc, copy.deepcopy(model.state_dict())

            lr_now = optimizer.param_groups[0]["lr"]
            print(f"split {split} epoch {epoch+1}/{args.epochs} | lr {lr_now:.6f} | "
                  f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f}")
            curves.add_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

        curves.save_plot()

        torch.save(best_state, out)
        print(f"split {split} saved to {out} (best val acc {best_val_acc:.4f})")

        del model, train_loader, val_loader, optimizer, curves, criterion, scheduler


if __name__ == "__main__":
    main()
