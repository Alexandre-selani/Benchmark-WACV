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

from Utils import NAMES, RESULTS_ROOT

NUM_CLASSES = 20
N_SPLITS = 5


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
