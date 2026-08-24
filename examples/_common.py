"""
    Shared plumbing for the five TinyImageNet examples.

    Each example is a single readable file that runs one method on one class
    split. What lives here is only what would otherwise be copy-pasted five
    times: argument parsing, checkpoint lookup with a useful error, and result
    printing.
"""

import argparse
import sys

from Utils import DEVICE


def base_parser(description, default_epsilon):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split", type=int, default=0, choices=range(5), metavar="{0..4}",
                        help="which of the five class splits to evaluate (default: 0)")
    parser.add_argument("--epsilon", type=float, default=default_epsilon,
                        help=f"open-set rejection threshold (default: {default_epsilon})")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="evaluation batch size (default: 128)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="dataloader workers (default: 4)")
    return parser


def require(path, what):
    """
        Resolves a checkpoint path, failing with an actionable message instead
        of a bare FileNotFoundError deep inside torch.load. Checkpoints are not
        shipped in the repository — see the README on where to point
        OSR_CHECKPOINTS.
    """
    if not path.exists():
        sys.exit(
            f"error: {what} not found at\n"
            f"    {path}\n\n"
            f"Checkpoints are not distributed with this repository. Point\n"
            f"OSR_CHECKPOINTS at the directory that holds them, e.g.\n"
            f"    export OSR_CHECKPOINTS=/path/to/checkpoints"
        )
    return path


def report(method, split, epsilon, metrics):
    print()
    print(f"  {method}  ·  TinyImageNet split {split}  ·  epsilon {epsilon}  ·  {DEVICE}")
    print("  " + "-" * 52)
    for name, value in metrics.items():
        print(f"  {name:<16} {value:.4f}" if isinstance(value, float) else f"  {name:<16} {value}")
    print()
