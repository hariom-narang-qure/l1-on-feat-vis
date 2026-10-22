"""Imagenette / Imagewoof as fixed 224 px uint8 arrays.

Source: the fastai 320 px releases (github.com/fastai/imagenette), 10 ImageNet classes each.
  imagenette: 10 easy, distinct classes (tench, church, parachute, ...)
  imagewoof:  10 dog breeds (fine-grained)
Their own train / val split is kept: train for every fit and selection, val scored once.

Preprocessing, the usual linear-probe eval transform with no augmentation (features are frozen):
  1. RGB, resize the shorter side to 224 (bilinear, antialiased)
  2. centre crop 224 x 224
Everything is decoded once into images.npy (n, 224, 224, 3) uint8 plus labels.npy and is_val.npy, so
feature extraction reads one array instead of re-decoding JPEGs for every net.

Usage: python -m l1_on_feat_vis.imagenet_probe.data
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/realdata/hariom.narang/imagenet_subsets")
RAW = {"nette": ROOT / "raw/imagenette2-320", "woof": ROOT / "raw/imagewoof2-320"}
SIZE = 224


def dataset_dir(name):
    return ROOT / name


def centre_crop_224(path):
    image = Image.open(path).convert("RGB")
    scale = SIZE / min(image.size)
    w, h = max(SIZE, round(image.width * scale)), max(SIZE, round(image.height * scale))
    image = image.resize((w, h), Image.BILINEAR, reducing_gap=None)
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    return np.asarray(image.crop((left, top, left + SIZE, top + SIZE)))


def list_images(raw):
    classes = sorted(p.name for p in (raw / "train").iterdir())
    items = []
    for split in ("train", "val"):
        for label, wnid in enumerate(classes):
            items += [(path, label, split == "val") for path in sorted((raw / split / wnid).glob("*.JPEG"))]
    return classes, items


def build(name):
    out = dataset_dir(name)
    if (out / "images.npy").exists():
        return
    classes, items = list_images(RAW[name])
    with Pool(16) as pool:
        images = pool.map(centre_crop_224, [path for path, _, _ in items], chunksize=64)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "images.npy", np.stack(images))
    np.save(out / "labels.npy", np.array([label for _, label, _ in items], dtype=np.int64))
    np.save(out / "is_val.npy", np.array([is_val for _, _, is_val in items]))
    (out / "classes.txt").write_text("\n".join(classes))
    print(f"{name}: {len(items)} images, {sum(v for _, _, v in items)} val", flush=True)


class Dataset:
    def __init__(self, name):
        root = dataset_dir(name)
        self.name = name
        self.images = np.load(root / "images.npy", mmap_mode="r")
        self.labels = np.load(root / "labels.npy")
        self.is_val = np.load(root / "is_val.npy")


if __name__ == "__main__":
    for name in RAW:
        build(name)
