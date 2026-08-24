"""
    Stage 4 — GFROR open-set classifier. Requires stage 3.

    The autoencoder from stage 3 is frozen and used only to produce x_hat. The
    classifier sees the six-channel concatenation (x, x_hat) and trains two
    heads at once: classification over the known classes, and a self-supervised
    head predicting which of eight rotation/flip transformations was applied.
    The same transformation is applied to x and x_hat so the pair stays aligned.

    Produces:
        $OSR_CHECKPOINTS/gfror/openset_ae_tinyimgnet/ResNet18/Split_{i}/ckpt.pth

    Run:
        python training/4_gfror_classifier.py
"""

import copy
import gc

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm

from Datasets import TinyImageNet_loader
from Models import ResNet18_tinyimgnet_GFROR
from Utils import CHECKPOINTS_ROOT, DEVICE, fix_random_seed, gfror_generator_ckpt
import methods.gfror  # noqa: F401  — registers the aliases the pickled AE needs

from _common import IMAGE_SIZE, NUM_CLASSES, base_parser, guard_output

CE_WEIGHT, SS_WEIGHT = 0.8, 0.2

# The eight transformations of the self-supervised task.
TRANSFORMATIONS = np.array([
    T.RandomRotation(degrees=[90, 90]),      # deterministic rotation
    T.RandomRotation(degrees=[180, 180]),
    T.RandomRotation(degrees=[270, 270]),
    T.RandomRotation(degrees=[360, 360]),    # original input
    T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[90, 90])]),
    T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[180, 180])]),
    T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[270, 270])]),
    T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[360, 360])]),
])


def train_epoch(generator, classifier, dataloader, optimizer, loss_fn):
    classifier.train()
    generator.eval()
    ce_losses, ss_losses, totals = [], [], []

    for x, y in tqdm(dataloader, desc="train", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            x_hat = generator(x)

        ce_loss = loss_fn(classifier(torch.cat((x, x_hat), dim=1))[0], y)

        trans_ind = torch.randint(len(TRANSFORMATIONS), (x.size(0),))
        rand_trans = TRANSFORMATIONS[trans_ind]
        t_x = torch.stack([t(x[i]) for i, t in enumerate(rand_trans)], dim=0)
        t_x_hat = torch.stack([t(x_hat[i]) for i, t in enumerate(rand_trans)], dim=0)
        ss_loss = loss_fn(classifier(torch.cat((t_x, t_x_hat), dim=1))[1], trans_ind.to(DEVICE))

        loss = CE_WEIGHT * ce_loss + SS_WEIGHT * ss_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ce_losses.append(ce_loss.item())
        ss_losses.append(ss_loss.item())
        totals.append(loss.item())

    return (sum(ce_losses) / len(ce_losses), sum(ss_losses) / len(ss_losses),
            sum(totals) / len(totals))


@torch.no_grad()
def evaluate_closedset(generator, classifier, dataloader, loss_fn):
    """Known classes only, classification head only."""
    classifier.eval()
    generator.eval()
    losses, correct, total = [], 0, 0
    for x, y in tqdm(dataloader, desc="val", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = classifier(torch.cat((x, generator(x)), dim=1))[0]
        losses.append(loss_fn(logits, y).item())
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return sum(losses) / len(losses), correct / total


def main():
    args = base_parser(__doc__, default_epochs=100, default_lr=1e-3,
                       default_batch_size=128).parse_args()
    fix_random_seed(42)

    save_root = CHECKPOINTS_ROOT / "gfror" / "openset_ae_tinyimgnet" / "ResNet18"
    save_root.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])
    data = TinyImageNet_loader(batch_size=args.batch_size, num_workers=args.num_workers,
                               image_size=IMAGE_SIZE)

    for split in args.splits:
        gc.collect()
        torch.cuda.empty_cache()

        split_dir = save_root / f"Split_{split}"
        split_dir.mkdir(parents=True, exist_ok=True)
        out = guard_output(split_dir / "ckpt.pth", args.force)

        classifier = ResNet18_tinyimgnet_GFROR(
            num_classes=NUM_CLASSES, num_transforms=len(TRANSFORMATIONS), weights=None,
        ).to(DEVICE)

        ae_path = gfror_generator_ckpt(split)
        if not ae_path.exists():
            raise SystemExit(f"error: stage 3 output missing at {ae_path}\n"
                             f"Run training/3_gfror_generator.py first.")
        generator = torch.load(ae_path, weights_only=False, map_location=DEVICE).to(DEVICE)
        generator.eval()
        for param in generator.parameters():
            param.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",
                                                               patience=5, factor=0.8)

        train_loader = data.get_train_loader(split, transform)
        val_loader = data.get_val_known_loader(split, transform)

        best_val_acc, best_state = 0.0, None
        for epoch in range(args.epochs):
            ce, ss, total = train_epoch(generator, classifier, train_loader, optimizer, criterion)
            val_loss, val_acc = evaluate_closedset(generator, classifier, val_loader, criterion)
            scheduler.step(val_loss)

            print(f"split {split} epoch {epoch+1}/{args.epochs} | "
                  f"lr {optimizer.param_groups[0]['lr']:.6f} | ce {ce:.4f} ss {ss:.4f} "
                  f"train {total:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

            # Selected on closed-set accuracy over the known classes — the only
            # legitimate criterion here, since the unknowns cannot inform training.
            if val_acc > best_val_acc:
                best_val_acc, best_state = val_acc, copy.deepcopy(classifier.state_dict())

        if best_state is not None:
            classifier.load_state_dict(best_state)
        # Pickled module, matching the released checkpoints.
        torch.save(classifier, out)
        print(f"split {split} saved (best val acc {best_val_acc:.4f})")

        del classifier, generator, train_loader, val_loader


if __name__ == "__main__":
    main()
