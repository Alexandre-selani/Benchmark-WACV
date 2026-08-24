"""
    Stage 2 — CAC backbone.

    Trains one CAC backbone per split. Anchors are fixed at the start as
    alpha * I, so each known class is pulled toward its own axis in logit
    space, and the CAC loss combines the anchor term (distance to the true
    class anchor) with a tuplet term (distance to the others).

    Independent of stage 1 — CAC trains its own backbone from scratch.

    Produces:
        $OSR_CHECKPOINTS/ResNet18/Split_{i}/ResNet18_TinyImageNet_cac_split_{i}.pt

    Run:
        python training/2_cac_backbone.py
"""

import copy
import gc

import torch
import torch.optim as optim

from Datasets import TinyImageNet_loader
from Models import ResNet18_tinyimgnet_cac
from Utils import (DEVICE, TrainingCurves, CACLoss, CHECKPOINTS_ROOT,
                   eval_cac, fix_random_seed, train_cac)

from _common import (NUM_CLASSES, base_parser, build_scheduler, guard_output,
                     strong_train_transform)

WARMUP_EPOCHS = 5
ALPHA = 10      # anchor magnitude
LAMBDA = 0.15   # weight of the anchor term in the CAC loss


def main():
    args = base_parser(__doc__, default_epochs=200, default_lr=1e-3,
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
        out = guard_output(split_dir / f"ResNet18_TinyImageNet_cac_split_{split}.pt", args.force)
        curves = TrainingCurves(f"ResNet18_Split_{split}", "TinyImageNet_cac", dir=str(split_dir))

        model = ResNet18_tinyimgnet_cac(num_classes=NUM_CLASSES, weights=None)
        model.set_anchors(torch.diag(torch.Tensor([ALPHA] * NUM_CLASSES)))
        model = model.to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        scheduler = build_scheduler(optimizer, args.epochs, WARMUP_EPOCHS)

        mean, std = data.norm_stats[split]
        train_loader = data.get_train_loader(split, strong_train_transform(mean, std))
        val_loader = data.get_val_known_loader(split, data.eval_transforms[split])

        best_val_acc, best_state = 0.0, None
        for epoch in range(args.epochs):
            train_loss, train_acc = train_cac(train_loader, model, CACLoss, optimizer=optimizer,
                                              num_classes=NUM_CLASSES, lbda=LAMBDA)
            val_loss, val_acc = eval_cac(val_loader, model, CACLoss,
                                         num_classes=NUM_CLASSES, lbda=LAMBDA)
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

        del model, train_loader, val_loader, optimizer, curves, scheduler


if __name__ == "__main__":
    main()
