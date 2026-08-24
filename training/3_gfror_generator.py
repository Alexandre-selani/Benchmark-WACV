"""
    Stage 3 — GFROR autoencoder.

    Trains one autoencoder per split on the known classes only, with plain MSE
    reconstruction loss. Because it never sees the unknowns, its reconstruction
    degrades on them at evaluation time, which is what stage 4's classifier
    turns into an open-set signal.

    No normalization anywhere in this pipeline: the decoder ends in a Sigmoid,
    so inputs and outputs both live in [0, 1].

    Produces:
        $OSR_CHECKPOINTS/gfror/ae_tinyimgnet/split_{i}.pth

    Run:
        python training/3_gfror_generator.py
"""

import copy
import gc

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import save_image
from tqdm import tqdm

from Datasets import TinyImageNet_loader
from Utils import CHECKPOINTS_ROOT, DEVICE, RESULTS_ROOT, fix_random_seed
from methods.gfror.model import VanillaAE64

from _common import IMAGE_SIZE, base_parser, guard_output

LATENT_SIZE = 200
BETAS = (0.5, 0.999)


def train_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    losses = []
    for batch, _ in tqdm(dataloader, desc="train", leave=False):
        batch = batch.float().to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(batch), batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate(model, dataloader, loss_fn):
    model.eval()
    losses = [loss_fn(model(b.float().to(DEVICE)), b.float().to(DEVICE)).item()
              for b, _ in tqdm(dataloader, desc="val", leave=False)]
    return sum(losses) / len(losses)


@torch.no_grad()
def save_recons(model, dataloader, path, n=20):
    """Grid alternating originals and reconstructions, for visual sanity checks."""
    model.eval()
    imgs = next(iter(dataloader))[0][:n].to(DEVICE)
    out = model(imgs)
    merged = torch.stack((imgs.cpu().clamp(0, 1), out.cpu().clamp(0, 1)), dim=1)
    save_image(merged.view(-1, 3, IMAGE_SIZE, IMAGE_SIZE), path, nrow=8)


def main():
    args = base_parser(__doc__, default_epochs=40, default_lr=1e-4,
                       default_batch_size=256).parse_args()
    fix_random_seed(42)

    ckpt_dir = CHECKPOINTS_ROOT / "gfror" / "ae_tinyimgnet"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # No augmentation and no normalization: same pipeline for train and val.
    transform = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])
    data = TinyImageNet_loader(batch_size=args.batch_size, num_workers=args.num_workers,
                               image_size=IMAGE_SIZE)

    for split in args.splits:
        gc.collect()
        torch.cuda.empty_cache()

        out = guard_output(ckpt_dir / f"split_{split}.pth", args.force)
        out_dir = RESULTS_ROOT / "gfror" / "recons" / f"split_{split}"
        out_dir.mkdir(parents=True, exist_ok=True)

        train_loader = data.get_train_loader(split, transform)
        val_loader = data.get_val_known_loader(split, transform)

        model = VanillaAE64(LATENT_SIZE).to(DEVICE)
        loss_fn = nn.MSELoss(reduction="mean").to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=BETAS,
                                     weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=3,
                                      threshold_mode="abs")

        best_loss, best_state = float("inf"), None
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
            val_loss = evaluate(model, val_loader, loss_fn)
            scheduler.step(val_loss)

            print(f"split {split} epoch {epoch+1}/{args.epochs} | "
                  f"lr {optimizer.param_groups[0]['lr']:.6f} | "
                  f"train loss {train_loss:.4f} | val loss {val_loss:.4f}")

            if (epoch + 1) % 10 == 0:
                save_recons(model, val_loader, out_dir / f"recons_epoch{epoch+1}.png")

            if val_loss < best_loss:
                best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())

        if best_state is not None:
            model.load_state_dict(best_state)
            # Saved as a pickled module, matching the released checkpoints.
            # See the note in the repository README.
            torch.save(model, out)
            print(f"split {split} saved (best val loss {best_loss:.4f})")

        del train_loader, val_loader, model, optimizer, loss_fn, scheduler


if __name__ == "__main__":
    main()
