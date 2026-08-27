"""
    Shared plumbing for the benchmark sweeps.

    Every sweep follows the same shape, and it is the shape that keeps the cost
    reasonable: a sample's score does not depend on epsilon, so each split is
    scored ONCE and the resulting scores are re-thresholded at every epsilon.
    A 1000-point sweep therefore costs about the same as a single evaluation.

    OpenMax is the exception — tailsize and alpha change the Weibull fit, so it
    refits per grid point and only the epsilon axis is free.
"""

import argparse

import numpy as np
import pandas as pd

from Utils import NAMES, RESULTS_ROOT

NUM_CLASSES = 20
N_SPLITS = 5

BEST_FILENAME = "best_hiperparameters.csv"
SELECTION_METRIC = "F1 macro_mean"


def base_parser(description, default_start, default_stop, default_step):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--splits", type=int, nargs="+", default=list(range(N_SPLITS)),
                        metavar="N", help="splits to evaluate (default: all five)")
    parser.add_argument("--subsets", nargs="+", default=["val", "test"],
                        choices=["val", "test"],
                        help="which subsets to sweep (default: both)")
    parser.add_argument("--epsilon-start", type=float, default=default_start,
                        help=f"first epsilon (default: {default_start})")
    parser.add_argument("--epsilon-stop", type=float, default=default_stop,
                        help=f"one past the last epsilon (default: {default_stop})")
    parser.add_argument("--epsilon-step", type=float, default=default_step,
                        help=f"epsilon step (default: {default_step})")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="evaluation batch size (default: 128)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="dataloader workers (default: 4)")
    parser.add_argument("--model", default=NAMES.RESNET18.value,
                        help=f"backbone name used in the output paths (default: {NAMES.RESNET18.value})")
    parser.add_argument("--confusion-matrices", action="store_true",
                        help="also render one confusion matrix per epsilon; off by "
                             "default because a fine sweep would produce thousands")
    return parser


def epsilons_from(args):
    """The sweep as a plain list, rounded to kill np.arange float noise."""
    values = np.arange(args.epsilon_start, args.epsilon_stop, args.epsilon_step)
    return [round(float(e), 6) for e in values]


def output_dir(method, model, subset):
    """results/<method>/<model>/<Val|Test>/"""
    directory = RESULTS_ROOT / method / model / subset.capitalize()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def mc_column_names(num_classes=NUM_CLASSES):
    return ["Unknown"] + [str(i) for i in range(num_classes)]


def loader_for(data, subset, split, transform):
    """The val or test loader of one split, with unknowns included."""
    if subset == "val":
        return data.get_val_loader(split, transform)
    return data.get_test_loader(split, transform)


def announce(method, subset, splits, epsilons):
    print(f"\n{method} · {subset} · splits {list(splits)} · "
          f"{len(epsilons)} epsilons from {epsilons[0]} to {epsilons[-1]}")


def write_best_hyperparameters(sweeps, directory):
    """
        Names the single best hyperparameter combination and writes it out.

        `sweeps` is a list of (hyperparameters, frame) pairs -- one entry per
        combination of the hyperparameters that are not epsilon, holding the
        full-precision frame aggregate() returned for it. A method whose only
        hyperparameter is epsilon passes one entry with an empty dict.

        Selection is by mean macro F1 across the splits, compared at full
        precision because the CSVs on disk are rounded to three decimals and
        would tie. An exact tie resolves to the first row, meaning the lowest
        epsilon of the earliest combination.

        Call this for the validation subset only: picking the operating point
        on test would report an optimistic upper bound rather than a choice
        that has to generalize.
    """
    frames = []
    for hyperparameters, frame in sweeps:
        tagged = frame.copy()
        for column, value in reversed(list(hyperparameters.items())):
            tagged.insert(0, column, value)
        frames.append(tagged)

    pooled = pd.concat(frames, ignore_index=True)
    best = pooled.loc[[pooled[SELECTION_METRIC].idxmax()]].copy()
    best["selected_by"] = SELECTION_METRIC

    metric_columns = [c for c in best.columns
                      if c.endswith(("_mean", "_std")) and c != "selected_by"]
    path = directory / BEST_FILENAME
    best.round({c: 3 for c in metric_columns}).to_csv(path, index=False)

    row = best.iloc[0]
    chosen = ", ".join(f"{c}={row[c]}" for c in
                       [*(sweeps[0][0].keys()), "epsilon"])
    print(f"best by {SELECTION_METRIC}: {chosen} -> "
          f"{row[SELECTION_METRIC]:.3f}  ({path})")
    return best
