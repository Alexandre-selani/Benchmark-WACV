# INSERT TITLE

Authors: Removed before camera-ready

Abstract:

---

Six open-set recognition methods — **CAC**, **GFROR**, **COSTARR**, **OpenMax**,
**OpenGan** and **MSP** — evaluated on the same TinyImageNet splits, through the
same data loader and the same metric implementation. MSP is the softmax-threshold
baseline: it reuses the closed-set backbone unchanged and serves as the reference
the other five are measured against.

Each method ships with one runnable example that scores a single class split
end to end.

---

## Install

```bash
conda env create -f environment.yml
conda activate osr-tinyimagenet
pip install -e .
```

That single `pip install -e .` is the whole dependency story. It makes five
top-level packages importable from anywhere:

| Package | Lives in | What it holds |
|---|---|---|
| `Datasets` | `core/Datasets/` | the TinyImageNet open-set loader |
| `Models` | `core/Models/` | the ResNet18 backbones each method adapts |
| `Utils` | `core/Utils/` | paths, device, metrics, metric logger |
| `methods` | `methods/` | one subpackage per method |
| `osr_pytorch_ood` | `third_party/osr_pytorch_ood/` | vendored, for OpenMax only |

Because everything is installed, **no script depends on the working
directory** — the examples run from anywhere, and there are no `sys.path`
manipulations in this repository.

> Install into a **fresh** environment. If an environment already provides
> `Datasets`, `Models` or `Utils` from somewhere else, the two installs fight
> over those names and which one wins is not predictable.

## Data

The class splits **are** in this repository, under `splits/` — five JSON files,
24 KB in total, naming the 20 known and 180 unknown classes of each split. They
define the entire open-set protocol.

TinyImageNet itself is not, so fetch and prepare it first:

```bash
python training/0_prepare_data.py           # downloads and restructures
python training/0_prepare_data.py --check   # verifies an existing copy
```

That step is not optional even if you already have TinyImageNet. The archive
published at cs231n is not laid out the way `ImageFolder` expects — `train`
has an extra `images` level and all of `val` sits in one folder with a
`val_annotations.txt` beside it. Used as-is, `val` would read as a single class
called "images". The script removes the extra level and moves each validation
image into its class folder; `--check` confirms 100000 train and 10000 val
images over 200 classes.

## Paths

Every path resolves through `Utils.paths`, so nothing in the source needs
editing. Override any of them by environment variable:

```bash
export OSR_TINY_IMAGENET_DIR=/path/to/tiny-imagenet-200   # default: data/tiny-imagenet-200
export OSR_SPLITS_DIR=/path/to/class_splits               # default: splits/
export OSR_CHECKPOINTS=/path/to/checkpoints               # default: checkpoints/
export OSR_FEATURES=/path/to/features                     # default: features/
export OSR_RESULTS=/path/to/results                       # default: results/
```

`OSR_CHECKPOINTS` is expected to hold:

```
checkpoints/
├── ResNet18/Split_{0..4}/
│   ├── ResNet18_TinyImageNet_split_{i}.pt         # closed-set backbone
│   └── ResNet18_TinyImageNet_cac_split_{i}.pt     # CAC-trained backbone
├── gfror/
│   ├── ae_tinyimgnet/split_{i}.pth                # autoencoder
│   └── openset_ae_tinyimgnet/ResNet18/Split_{i}/ckpt.pth
└── opengan/ResNet18/Fold+{i}/best_epoch.DNet      # discriminator
```

Set `OSR_DEVICE` (`cpu`, `cuda:1`, …) to override device selection; by default
CUDA is used when available and CPU otherwise.

## Training

The weights are not distributed with this repository, and every checkpoint the
examples need can be regenerated from these six stages. They are numbered in
dependency order:

```bash
python training/0_prepare_data.py            # dataset (see Data above)
python training/1_closedset_backbone.py      # backbone for OpenMax, COSTARR, OpenGan
python training/2_cac_backbone.py            # CAC backbone (independent of stage 1)
python training/3_gfror_generator.py         # GFROR autoencoder
python training/4_gfror_classifier.py        # GFROR classifier      (needs 3)
python training/5_opengan_features.py        # cache OpenGan features (needs 1)
python training/6_opengan_discriminator.py   # OpenGan discriminator  (needs 5)
```

Each stage trains all five splits by default and writes into `OSR_CHECKPOINTS`
under the layout above. Common flags: `--splits 0 1`, `--epochs`, `--lr`,
`--batch-size`, `--num-workers`. A short `--splits 0 --epochs 5` run is a good
way to confirm the setup before committing to a full one.

Stages will **refuse to overwrite an existing checkpoint** unless you pass
`--force`. This matters if you assembled `checkpoints/` out of symlinks to
weights stored elsewhere: without the guard, a training run would follow the
symlink and destroy the original.

## Reproducing the tables

The examples score one split at one threshold. The paper tables are the sweeps
in `benchmark/`: five splits x an epsilon sweep, aggregated into mean and
standard deviation per epsilon.

```bash
python benchmark/cac.py
python benchmark/costarr.py
python benchmark/gfror.py
python benchmark/opengan.py
python benchmark/openmax.py
python benchmark/msp.py

python benchmark/summarize.py          # one comparison table across all six
```

Each writes, under `OSR_RESULTS`:

```
results/<method>/<model>/
├── Val/
│   ├── Val.csv                     mean and std of each metric, per epsilon
│   ├── Folds/Results_Fold_<i>.csv  the per-split values behind them
│   └── best_hiperparameters.csv    the winning configuration and its metrics
└── Test/
    ├── Test.csv
    └── Folds/Results_Fold_<i>.csv
```

`best_hiperparameters.csv` is written for the validation subset only. It holds
a single row: the hyperparameter combination with the highest mean macro F1
across the splits, followed by every metric at that point. Selection compares
at full precision, since the CSVs are rounded to three decimals and a fine
sweep ties hundreds of rows there. There is deliberately no test-side
equivalent — choosing the operating point on test reports an upper bound
rather than a result.

`summarize.py` picks each method's operating point on the **validation** sweep
and reports what it scores on **test**, which is the protocol the tables should
use. `--select test` reads the best point straight off the test sweep instead;
that number is an upper bound, not a result.

Flags: `--splits`, `--subsets val test`, `--epsilon-start/-stop/-step`,
`--batch-size`, `--num-workers`, `--confusion-matrices`. OpenMax additionally
takes `--tailsizes` and `--alphas`.

These are fast because a sample's score does not depend on epsilon: each split
is scored once and the scores are re-thresholded across the sweep. The CAC
sweep — 1000 thresholds over five splits — takes well under a minute on one
GPU. OpenMax is the exception: tailsize and alpha change the Weibull fit, so it
refits at every grid point, and the default 5x5 grid over five splits is 125
fits. It also writes `grid_summary.csv`, ranking the grid points by their best
macro F1; its `best_hiperparameters.csv` accordingly names a tailsize, an alpha
and an epsilon rather than an epsilon alone.

Default sweeps, matching the original experiments:

| Method | Sweep | Points |
|---|---|---|
| CAC | 0 to 1, step 0.001 | 1000 |
| COSTARR | 0 to 1, step 0.01 | 100 |
| OpenGan | 0 to 1, step 0.01 | 100 |
| GFROR | 0 to 30, step 0.2 | 150 |
| OpenMax | tailsize x alpha x epsilon | 5x5x5 |
| MSP | 0 to 1, step 0.01 | 100 |

## Running one method once

```bash
python examples/cac_tinyimagenet.py      --split 0
python examples/gfror_tinyimagenet.py    --split 0
python examples/costarr_tinyimagenet.py  --split 0
python examples/openmax_tinyimagenet.py  --split 0
python examples/opengan_tinyimagenet.py  --split 0
python examples/msp_tinyimagenet.py      --split 0
```

Every example takes `--split {0..4}`, `--epsilon`, `--batch-size` and
`--num-workers`; OpenMax additionally takes `--tailsize` and `--alpha`.

**The `--epsilon` defaults are illustrative.** The reported operating points come
from the validation sweep in `benchmark/`, not from these defaults.

## The splits

`TinyImageNet_loader` builds each split from a `{split}.json` listing known and
unknown classes:

- **train** — the 500 `train` images of each known class.
- **val / test** — drawn from the `val` folder and disjoint: each known class
  contributes 25 images to each, each unknown class 3 to each.
- Unknown samples are labelled `-1`; known labels are remapped to `0…19`.

Normalization statistics are computed per split from that split's known-class
training images, so a split never sees statistics derived from its unknowns.

## Layout

```
core/           shared loader, backbones, metrics, path and device config
methods/        cac · costarr · gfror · msp · openmax · opengan
splits/         the five class splits — the open-set protocol itself
training/       numbered stages: dataset preparation, then every checkpoint
examples/       one runnable TinyImageNet example per method
benchmark/      the five-split epsilon sweeps behind the reported tables
docs/           adding-a-method.md — how to plug a sixth method in
third_party/    vendored pytorch-ood (Apache 2.0) — see third_party/NOTICE
```

## Implementation sources

Official implementations were used wherever they existed. Where none did, the
code comes from a peer-reviewed framework; where neither did, from an unofficial
implementation validated against the numbers reported in the original paper.

| Method | Source | Provenance |
|---|---|---|
| CAC | [dimitymiller/cac-openset](https://github.com/dimitymiller/cac-openset) | official |
| OpenGan | [aimerykong/OpenGAN](https://github.com/aimerykong/OpenGAN) | official |
| OpenMax | [kkirchheim/pytorch-ood](https://github.com/kkirchheim/pytorch-ood) | peer-reviewed framework, adapted |
| COSTARR | [Vastlab/COSTARR](https://github.com/Vastlab/COSTARR) | reimplemented, later validated against the official release |
| GFROR | [misraya/unsupervised_open_set](https://github.com/misraya/unsupervised_open_set) | unofficial, validated against the paper |
| MSP | — | no external source; 40 lines over the stage 1 backbone |

Three of them needed more than a copy:

- **OpenMax.** pytorch-ood targets out-of-distribution detection, where only the
  known/unknown decision matters, so its detector returned a novelty score alone
  and discarded the class prediction. It was modified to return the full score
  vector, leaving the core logic untouched. It is also vendored and renamed —
  see the note at the end of this file for why.
- **COSTARR.** The official repository was unavailable when these experiments
  began, so the scoring function was reimplemented from the paper. Once the
  official code was released, the two were compared on synthetic data and on
  MNIST/Omniglot and agreed exactly.
- **GFROR.** No official release exists. The unofficial implementation was
  validated by reproducing the paper's CIFAR-10 benchmark: it reports
  AUROC 80.9 ± 3, and this copy measured 78.4 ± 3.

## Adding a method

`docs/adding-a-method.md` walks through adding a method, using MSP as a worked
example — it was added by following that guide, so its files can be read
alongside it. Covers the package layout, where checkpoint paths live, why
scoring is kept separate from thresholding, and how to pick the metric
`convention`.

## Notes on the code

A few decisions are worth knowing before reading:

- **`convention=` in `OSRMetrics` names a convention, not the method being
  evaluated.** It selects two things at once: which index marks unknown (`0`
  for `"openmax"`, `-1` otherwise) and whether AUROC uses the outlier score as
  given (`"opengan"`) or flipped to `1 - score`. Four of the five examples pass
  `"opengan"` because their scores are already oriented that way.
- **GFROR checkpoints are pickled modules**, not state dicts, so they carry the
  import paths of the classes inside them. `methods/gfror/_compat.py` registers
  the pre-refactor `model.*` names on import so the released checkpoints load
  unchanged. Stages 3 and 4 keep saving them this way, so freshly trained
  weights stay interchangeable with the released ones.
- **pytorch-ood is vendored rather than installed from PyPI, and imported as
  `osr_pytorch_ood`**. Vendored because the OpenMax detector used here takes an
  extra `epsilon` argument upstream does not have; renamed because a vendored
  copy under the upstream name is still shadowed by any `pytorch_ood` on
  sys.path, silently. See `third_party/NOTICE`.
