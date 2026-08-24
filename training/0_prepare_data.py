"""
    Stage 0 — download and restructure TinyImageNet.

    The archive published at cs231n does not have the layout torchvision's
    ImageFolder expects, and the loader in this repository reads both `train`
    and `val` with ImageFolder. Two changes are needed:

        train/nXXXX/images/*.JPEG  ->  train/nXXXX/*.JPEG
            (the extra `images` level is removed; the per-class *_boxes.txt
             files are dropped)

        val/images/*.JPEG          ->  val/nXXXX/*.JPEG
        val/val_annotations.txt         (each image is moved into the folder of
                                         the class the annotations file assigns
                                         to it)

    Without this, ImageFolder would read `val` as a single class called
    "images" and every split would be meaningless.

    Run:
        python training/0_prepare_data.py                 # download and prepare
        python training/0_prepare_data.py --archive t.zip # use a local archive
        python training/0_prepare_data.py --check         # only verify the layout
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from Utils import TINY_IMAGENET_DIR

URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
N_CLASSES = 200
N_TRAIN_PER_CLASS = 500
N_VAL_TOTAL = 10000


def download(dest):
    print(f"downloading {URL}\n  -> {dest}  (about 240 MB)")

    def progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(100.0, count * block_size * 100.0 / total_size)
            print(f"\r  {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(URL, dest, reporthook=progress)
    print()
    return dest


def flatten_train(root):
    """train/nXXXX/images/*.JPEG -> train/nXXXX/*.JPEG"""
    moved = 0
    for class_dir in sorted((root / "train").iterdir()):
        if not class_dir.is_dir():
            continue
        images = class_dir / "images"
        if images.is_dir():
            for img in images.iterdir():
                img.rename(class_dir / img.name)
                moved += 1
            images.rmdir()
        for boxes in class_dir.glob("*_boxes.txt"):
            boxes.unlink()
    return moved


def regroup_val(root):
    """val/images/*.JPEG -> val/<class>/*.JPEG, using val_annotations.txt"""
    val = root / "val"
    images, annotations = val / "images", val / "val_annotations.txt"
    if not images.is_dir():
        return 0
    if not annotations.is_file():
        sys.exit(f"error: {annotations} is missing, cannot regroup val/")

    moved = 0
    for line in annotations.read_text().splitlines():
        if not line.strip():
            continue
        filename, class_id = line.split("\t")[:2]
        source = images / filename
        if not source.is_file():
            continue
        target_dir = val / class_id
        target_dir.mkdir(exist_ok=True)
        source.rename(target_dir / filename)
        moved += 1

    if not any(images.iterdir()):
        images.rmdir()
    return moved


def check(root):
    """Verifies the prepared layout, returning True when it is usable."""
    problems = []

    for name in ("train", "val"):
        directory = root / name
        if not directory.is_dir():
            problems.append(f"{directory} does not exist")
            continue

        classes = sorted(d for d in directory.iterdir() if d.is_dir())
        if len(classes) != N_CLASSES:
            problems.append(f"{directory}: found {len(classes)} class folders, expected {N_CLASSES}")
        if (directory / "images").is_dir():
            problems.append(f"{directory}/images still exists — the layout was not converted")
        if classes and (classes[0] / "images").is_dir():
            problems.append(f"{classes[0]}/images still exists — train was not flattened")

    train_images = sum(1 for _ in (root / "train").glob("*/*.JPEG")) if (root / "train").is_dir() else 0
    val_images = sum(1 for _ in (root / "val").glob("*/*.JPEG")) if (root / "val").is_dir() else 0
    expected_train = N_CLASSES * N_TRAIN_PER_CLASS
    if train_images != expected_train:
        problems.append(f"train: {train_images} images, expected {expected_train}")
    if val_images != N_VAL_TOTAL:
        problems.append(f"val: {val_images} images, expected {N_VAL_TOTAL}")

    if problems:
        print("layout NOT usable:")
        for problem in problems:
            print(f"  - {problem}")
        return False

    print(f"layout OK — {root}")
    print(f"  train: {train_images} images over {N_CLASSES} classes")
    print(f"  val:   {val_images} images over {N_CLASSES} classes")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=TINY_IMAGENET_DIR,
                        help=f"where the dataset should end up (default: {TINY_IMAGENET_DIR})")
    parser.add_argument("--archive", type=Path,
                        help="use an already downloaded zip instead of fetching it")
    parser.add_argument("--check", action="store_true",
                        help="only verify the layout at --dest, change nothing")
    args = parser.parse_args()

    if args.check:
        sys.exit(0 if check(args.dest) else 1)

    dest = args.dest
    if dest.is_dir() and (dest / "train").is_dir():
        print(f"{dest} already exists; verifying it instead of downloading")
        sys.exit(0 if check(dest) else 1)

    dest.parent.mkdir(parents=True, exist_ok=True)
    archive = args.archive or download(dest.parent / "tiny-imagenet-200.zip")

    print(f"extracting {archive}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest.parent)

    extracted = dest.parent / "tiny-imagenet-200"
    if extracted != dest:
        shutil.move(str(extracted), str(dest))

    print("flattening train/ ...", flush=True)
    print(f"  moved {flatten_train(dest)} images")
    print("regrouping val/ ...", flush=True)
    print(f"  moved {regroup_val(dest)} images")

    sys.exit(0 if check(dest) else 1)


if __name__ == "__main__":
    main()
