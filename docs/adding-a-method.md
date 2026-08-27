# Adding a method

This walks through adding an open-set method to the benchmark, using **MSP**
(Maximum Softmax Probability) as the worked example.

MSP was added by following exactly these steps, and it is in the repository, so
every file quoted below can be opened alongside this page:
`methods/msp/`, `examples/msp_tinyimagenet.py`, `benchmark/msp.py`. The numbers
at the end are what that run produced.

MSP is a good example precisely because it is minimal: it reuses the closed-set
backbone, so it needs no training stage and no new checkpoint path. Where a
method needs more than that, the extra step is flagged.

---

## The five contracts

A method fits into the benchmark by satisfying these. Nothing else is required,
and skipping any of them tends to fail late rather than loudly.

1. **It is a package under `methods/`**, importing only through relative or
   installed imports — never from the working directory.
2. **Paths come from `Utils.paths`**, never from a literal.
3. **The device comes from `Utils.DEVICE`**, never `"cuda"` or `.cuda()`.
4. **Metrics come from `OSRMetrics`**, with the `convention` matching how your
   score is oriented.
5. **Scoring is separated from thresholding.** This is what keeps a
   thousand-point sweep as cheap as a single evaluation.

---

## Step 1 — the method package

```
methods/msp/
├── __init__.py
└── msp.py
```

`methods/msp/msp.py` holds the part that is specific to the method. For MSP
that is one function:

```python
import torch

from Utils import DEVICE


@torch.no_grad()
def collect_msp(model, dataloader):
    model.eval()
    msp, preds, labels = [], [], []

    for X, y in dataloader:
        logits = model(X.to(DEVICE))
        if isinstance(logits, (tuple, list)):   # backbones returning (logits, features)
            logits = logits[0]

        max_prob, pred = torch.max(torch.softmax(logits, dim=1), dim=1)

        msp.append(max_prob.cpu())
        preds.append(pred.cpu())
        labels.append(y.cpu())

    return torch.cat(msp), torch.cat(preds), torch.cat(labels)
```

Note what it returns: the score, the closed-set prediction, and the labels —
and **not** a decision. No epsilon appears anywhere. That is contract 5, and
everything downstream depends on it.

`methods/msp/__init__.py` exports the public surface:

```python
"""MSP — Maximum Softmax Probability (Hendrycks & Gimpel, ICLR 2017)."""
from .msp import collect_msp
```

If your method has internal modules that import each other, use relative
imports (`from .encoder import Encoder`). An absolute `from encoder import ...`
works only when the working directory happens to be right, which is the failure
this layout exists to prevent.

## Step 2 — checkpoint paths

MSP reuses the stage 1 backbone, so `Utils.classifier_ckpt` already covers it
and there is nothing to add.

A method with weights of its own adds one helper to `core/Utils/paths.py`:

```python
def mymethod_ckpt(split, model="ResNet18"):
    """One-line description of what this checkpoint holds."""
    return CHECKPOINTS_ROOT / "mymethod" / model / f"Split_{split}" / "ckpt.pt"
```

and exports it from `core/Utils/__init__.py`. Keep the literal path inside
`paths.py` — that is the only file allowed to know a directory layout.

## Step 3 — the example

`examples/msp_tinyimagenet.py` runs one split at one threshold. It is the
smallest complete demonstration of the method, and the first thing a reader
opens.

```python
def main():
    args = base_parser(__doc__, default_epsilon=0.5).parse_args()
    fix_random_seed(42)

    data = TinyImageNet_loader(batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = data.get_test_loader(args.split, data.eval_transforms[args.split])

    model = ResNet18_tinyimgnet(num_classes=NUM_CLASSES, weights=None)
    ckpt = require(classifier_ckpt(args.split), f"closed-set checkpoint for split {args.split}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.to(DEVICE).eval()

    msp, preds, labels = collect_msp(model, test_loader)
    predicts = torch.where(msp >= args.epsilon, preds, -1)

    metrics = OSRMetrics(predict=predicts.numpy(), label=labels.numpy(),
                         outlier_scores=msp.numpy(), convention="opengan").compute()

    report("MSP", args.split, args.epsilon, metrics)
```

`base_parser`, `report` and `require` come from `examples/_common.py`. Use
`require()` for every checkpoint: it turns a missing file into a message that
says which environment variable to set, instead of a traceback from inside
`torch.load`.

### Choosing the `convention`

`convention` does not name the method being evaluated. It selects two
conventions at once — which index marks unknown, and how AUROC reads the
score:

| `convention` | unknown index | AUROC uses |
|---|---|---|
| `"openmax"` | `0` (labels shifted by +1) | `1 - outlier_scores` |
| `"opengan"` | `-1` | `outlier_scores` as given |
| anything else | `-1` | `1 - outlier_scores` |

MSP passes `"opengan"` with the raw softmax probability, because a **higher**
MSP already means "more likely known". Passing `1 - msp` under any other
string is equivalent. Getting this backwards does not crash — it silently
produces an AUROC below 0.5, so check that number first when a new method looks
wrong.

## Step 4 — the sweep

`benchmark/msp.py` produces the table: five splits, an epsilon sweep, and mean
plus standard deviation per epsilon, plus `best_hiperparameters.csv` naming the
validation-selected operating point. Its shape is the same for every method:

```python
for subset in args.subsets:
    directory = output_dir(METHOD, args.model, subset)
    logger = MetricLogger(epsilons, len(args.splits), str(directory),
                          flag_mc=args.confusion_matrices,
                          mc_column_names=mc_column_names(),
                          mc_title=f"MSP - TinyImageNet ({subset})")

    for fold, split in enumerate(args.splits):
        model = ...                                   # load once
        eval_loader = loader_for(data, subset, split, data.eval_transforms[split])
        msp, preds, labels = collect_msp(model, eval_loader)      # SCORE ONCE

        for epsilon in epsilons:                                  # THRESHOLD MANY
            predicts = torch.where(msp >= epsilon, preds, -1)
            metrics = OSRMetrics(..., convention="opengan").compute()
            logger.update(metrics, fold, epsilon)

    frame = logger.aggregate(f"{subset.capitalize()}.csv")
    if subset == "val":
        write_best_hyperparameters([({}, frame)], directory)
```

The scoring call sits **outside** the epsilon loop. That is the whole reason
the CAC sweep evaluates 1000 thresholds across five splits in well under a
minute. If your method's score genuinely depends on a hyperparameter — as
OpenMax's does on `tailsize` and `alpha` — put that parameter in an outer loop
and refit there, keeping epsilon innermost and free. `benchmark/openmax.py` is
the example to copy.

Three details worth matching:

- `fold` is the position in `args.splits`, not the split number. This keeps
  `--splits 2 3` working, where `MetricLogger` expects folds `0..n-1`.
- Pass `epsilon_decimals` to `MetricLogger` if your step is finer than 1e-6.
  Rounding exists only to absorb `np.arange` float noise; if it is coarser than
  the step, distinct epsilons collapse into one row and get averaged together.
- `write_best_hyperparameters` runs under `if subset == "val"` and nowhere
  else. It takes `(hyperparameters, frame)` pairs, so a method sweeping only
  epsilon passes one pair with an empty dict, while a grid passes one pair per
  combination and gets the winner across all of them. Feed it the frame
  `aggregate` returns rather than re-reading the CSV: the file on disk is
  rounded to three decimals and would tie.

## Step 5 — packaging

Nothing to do. `pyproject.toml` declares `include = ["methods*", ...]`, so
`methods.msp` is picked up automatically, and the editable install resolves new
submodules without reinstalling. This was verified: the example and the sweep
below both ran against an install made before `methods/msp/` existed.

The exception is the set of top-level packages in `package-dir`. Those names
are baked into a finder module at install time, so renaming or moving one —
`Models`, `Utils`, `Datasets`, `methods`, `osr_pytorch_ood` — needs
`pip install -e .` again. Adding a subpackage beneath an existing one does not.

## Step 6 — training, if the method needs it

MSP does not. A method that trains something of its own adds a numbered stage
under `training/`, continuing the existing sequence:

- read hyperparameters through `base_parser` from `training/_common.py`;
- wrap every checkpoint path in `guard_output(path, args.force)`, which refuses
  to overwrite existing weights — this matters most when `checkpoints/` is
  assembled from symlinks, where a write would otherwise destroy the original
  file outside the repository;
- if a stage depends on an earlier one, check for its output and name the stage
  to run, rather than failing on a missing file.

---

## Verifying

```bash
python examples/msp_tinyimagenet.py --split 0
python benchmark/msp.py
```

What the steps above actually produced:

```
  MSP  ·  TinyImageNet split 0  ·  epsilon 0.5  ·  cuda
  accuracy         0.6346
  F1 macro         0.6223
  auroc            0.7688
```

and the sweep, 100 epsilons across five splits in 9 seconds, best macro F1 at
epsilon 0.67:

| Metric | Value |
|---|---|
| F1 macro | 0.616 ± 0.034 |
| accuracy | 0.669 ± 0.021 |
| auroc | 0.743 |

Three checks before considering a method done:

1. **AUROC above 0.5.** Below it almost always means the `convention` is
   inverted rather than that the method failed.
2. **The sweep CSV has as many rows as the sweep has points.** Fewer means
   `epsilon_decimals` is coarser than the step.
3. **It runs from another directory.** `cd /tmp && python
   ~/path/to/benchmark/msp.py` catches any leftover dependence on the working
   directory.
