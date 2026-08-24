"""
    Stage 6 — OpenGan discriminator. Requires stages 1 and 5.

    Kong & Ramanan, "OpenGAN: Open-Set Recognition via Open Data Generation",
    ICCV 2021.

    A standard GAN is trained in *feature* space: the generator synthesizes fake
    512-d feature vectors, and the discriminator learns to tell them from the
    real features cached in stage 5. Only the discriminator is kept — at
    inference it acts as the open-set gate in front of the closed-set head.

    Epoch selection is what makes this work. A GAN has no meaningful
    convergence point for this purpose, so after every epoch the discriminator
    is scored by AUROC between known and unknown validation features, and the
    best-scoring epoch is the one saved.

    Produces:
        $OSR_CHECKPOINTS/opengan/ResNet18/Fold+{i}/best_epoch.DNet

    Run:
        python training/6_opengan_discriminator.py
"""

import copy
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from Utils import (CHECKPOINTS_ROOT, DEVICE, features_dir, fix_random_seed,
                   RESULTS_ROOT)
from methods.opengan.classifier import Discriminator, Generator, weights_init
from methods.opengan.utils import FeatDataset

from _common import base_parser, guard_output

NC = 512     # feature channels — ResNet18 penultimate width
NZ = 100     # generator latent size
NGF = 100    # generator feature maps
NDF = 100    # discriminator feature maps
BETA1 = 0.5
REAL_LABEL, FAKE_LABEL = 1, 0


def load_features(split, name, batch_size, num_workers):
    path = features_dir(split) / f"{name}_features.pt"
    if not path.exists():
        raise SystemExit(f"error: stage 5 output missing at {path}\n"
                         f"Run training/5_opengan_features.py first.")
    dataset = FeatDataset(torch.load(path, weights_only=False))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


@torch.no_grad()
def discriminator_auroc(netD, known_loader, unknown_loader):
    """
        AUROC of the discriminator's confidence at separating known validation
        features from unknown ones. Higher confidence should mean "known", so
        the scores are negated to match the convention that the positive class
        is the unknown one.
    """
    netD.eval()

    def confidences(loader):
        out = []
        for X, _ in loader:
            feats = X.to(DEVICE, dtype=torch.float32).unsqueeze_(-1).unsqueeze_(-1)
            out.append(netD(feats).view(-1).detach().cpu())
        return torch.cat(out).numpy()

    known, unknown = confidences(known_loader), confidences(unknown_loader)
    y_true = np.array([0] * len(known) + [1] * len(unknown))
    return roc_auc_score(y_true, np.concatenate([-known, -unknown]))


def main():
    args = base_parser(__doc__, default_epochs=200, default_lr=5e-4,
                       default_batch_size=256).parse_args()
    fix_random_seed(42)

    save_root = CHECKPOINTS_ROOT / "opengan" / "ResNet18"
    curves_dir = RESULTS_ROOT / "opengan"
    curves_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        gc.collect()
        torch.cuda.empty_cache()

        split_dir = save_root / f"Fold+{split}"
        split_dir.mkdir(parents=True, exist_ok=True)
        out = guard_output(split_dir / "best_epoch.DNet", args.force)

        netG = Generator(ngpu=1, nz=NZ, ngf=NGF, nc=NC).to(DEVICE)
        netD = Discriminator(ngpu=1, nc=NC, ndf=NDF).to(DEVICE)
        netG.apply(weights_init)
        netD.apply(weights_init)

        train_loader = load_features(split, "train", args.batch_size, args.num_workers)
        kkc_val_loader = load_features(split, "kkc_val", args.batch_size, args.num_workers)
        uuc_val_loader = load_features(split, "uuc_val", args.batch_size, args.num_workers)

        criterion = nn.BCELoss()
        # D is given a slightly lower learning rate than G to keep the two from
        # diverging early, when D can trivially win.
        optimizerD = optim.Adam(netD.parameters(), lr=args.lr / 1.5, betas=(BETA1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(BETA1, 0.999))

        g_losses, d_losses = [], []
        best_auroc, best_state = -1.0, None

        for epoch in range(args.epochs):
            for data, _ in train_loader:
                if data.size(0) <= 1:      # BatchNorm needs more than one sample
                    continue

                # (1) update D: maximize log(D(x)) + log(1 - D(G(z)))
                netD.train()
                netD.zero_grad()
                real = data.view(data.size(0), NC, 1, 1).to(DEVICE, dtype=torch.float32)
                b_size = real.size(0)
                label = torch.full((b_size,), REAL_LABEL, device=DEVICE, dtype=torch.float32)

                errD_real = criterion(netD(real).view(-1), label)
                errD_real.backward()

                noise = torch.randn(b_size, NZ, 1, 1, device=DEVICE)
                fake = netG(noise)
                label.fill_(FAKE_LABEL)
                errD_fake = criterion(netD(fake.detach()).view(-1), label)
                errD_fake.backward()
                optimizerD.step()

                # (2) update G: maximize log(D(G(z)))
                netG.zero_grad()
                label.fill_(REAL_LABEL)    # the generator wants D to call these real
                errG = criterion(netD(fake).view(-1), label)
                errG.backward()
                optimizerG.step()

                g_losses.append(errG.item())
                d_losses.append((errD_real + errD_fake).item())

            auroc = discriminator_auroc(netD, kkc_val_loader, uuc_val_loader)
            print(f"split {split} epoch {epoch+1}/{args.epochs} | "
                  f"loss_D {d_losses[-1]:.4f} loss_G {g_losses[-1]:.4f} | val auroc {auroc:.4f}")

            if auroc > best_auroc:
                best_auroc, best_state = auroc, copy.deepcopy(netD.state_dict())
                torch.save(best_state, out)

        print(f"split {split} saved (best val auroc {best_auroc:.4f})")
        save_loss_curve(g_losses, d_losses, curves_dir / f"learningCurves_{split}.png")

        del criterion, optimizerD, optimizerG, netD, netG


def save_loss_curve(g_losses, d_losses, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.title("Generator and discriminator loss during training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(path, bbox_inches="tight", transparent=True)
    plt.close()


if __name__ == "__main__":
    main()
