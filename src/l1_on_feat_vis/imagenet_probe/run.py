"""The controlled grid: random-net depth, number of nets and width, against pretrained references.

One factor moves at a time:
  nets   width 64, depth 6 and 20, every block, N = 1, 2, 5, 10, 20, 40 nets
  depth  width 64, N = 10, depth 1, 2, 3, 4, 6, 10, 20, 40, with every block or only the last block
         ("last" keeps 128 features per net at every depth, so it separates depth from feature count)
  width  width 16, depth 6 and 20, N = 10 and 40 (N = 40 at width 16 has the feature count of N = 10 at 64)
  res    the depth ladder at depth 10, 20, 40 with residual wiring on the same conv weights (features.py);
         depth <= 6 is identical to plain, since the skips start after the downsampling blocks
  refs   pixels, DINOv2-S/B/L, RAD-DINO, ResNet-50 (IN-1k supervised: a ceiling)
Each row runs at 10, 25 and 100 training images per class, 3 replicates.
Replicate r uses nets seeds 40r .. 40r + N - 1 (so nets nest within a replicate as N grows: N = 5 is
N = 2 plus 3 nets) and its own draw of training images from the 200-per-class pool. Seed noise and
subset noise are therefore averaged together over replicates.

Readouts per row: linear (all features), select64 / select256 (screened units + linear), mlp.
Results: one JSON line per (row, shots, replicate) in results_<worker>.jsonl; done rows are skipped.
Usage: python -m l1_on_feat_vis.imagenet_probe.run --worker I --workers W
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from l1_on_feat_vis.imagenet_probe.data import Dataset
from l1_on_feat_vis.imagenet_probe.features import DATASETS, image_index, random_path, ref_path
from l1_on_feat_vis.imagenet_probe.probes import linear_probe, mlp_probe, selected_units, stratified_folds

OUT = Path("/home/hariom.narang/l1-on-feat-vis/outputs/imagenet_probe")
SHOTS = (10, 25, 100)
REPLICATES = 3
SELECT_K = (64, 256)
REFS = ("pixels", "dinov2_s", "dinov2_b", "dinov2_l", "rad_dino", "resnet50")


def random_row(width, depth, nets, blocks="all", arch="plain"):
    row = {"source": "random", "width": width, "depth": depth, "nets": nets, "blocks": blocks}
    return row if arch == "plain" else {**row, "arch": arch}


def grid_rows():
    rows = [random_row(64, depth, nets) for depth in (6, 20) for nets in (1, 2, 5, 10, 20, 40)]
    rows += [random_row(64, depth, 10, blocks) for depth in (1, 2, 3, 4, 6, 10, 20, 40) for blocks in ("all", "last")]
    rows += [random_row(16, depth, nets) for depth in (6, 20) for nets in (10, 40)]
    rows += [random_row(64, depth, 10, blocks, "res") for depth in (10, 20, 40) for blocks in ("all", "last")]
    rows += [{"source": ref} for ref in REFS]
    unique = {json.dumps(r, sort_keys=True): r for r in rows}
    return list(unique.values())


def row_key(row):
    return json.dumps(row, sort_keys=True)


def block_range(row):
    return range(row["depth"]) if row["blocks"] == "all" else range(row["depth"] - 1, row["depth"])


def random_features(dataset, row, rep):
    """(images, features) float32 and the block of every feature."""
    blocks = list(block_range(row))
    parts = [np.load(random_path(dataset, row["width"], 40 * rep + i, row.get("arch", "plain")), mmap_mode="r")[:, blocks] for i in range(row["nets"])]
    x = np.concatenate([p.reshape(len(p), -1) for p in parts], 1).astype(np.float32)
    block_of = np.tile(np.repeat(blocks, row["width"] * 2), row["nets"])
    return x, block_of


def features(dataset, row, rep):
    if row["source"] == "random":
        return random_features(dataset, row, rep)
    return np.load(ref_path(dataset, row["source"])).astype(np.float32), None


def labels_and_split(dataset):
    data, index = Dataset(dataset), image_index(dataset)
    return data.labels[index], data.is_val[index]


def training_subset(y, is_val, shots, rep):
    rng = np.random.default_rng(1000 * rep + shots)
    return np.concatenate([rng.choice(np.flatnonzero((y == c) & ~is_val), shots, replace=False) for c in np.unique(y)])


def block_histogram(block_of, units):
    if block_of is None:
        return None
    blocks, counts = np.unique(block_of[units], return_counts=True)
    return {int(b): int(c) for b, c in zip(blocks, counts)}


def evaluate(x, block_of, y, is_val, shots, rep, device):
    classes = int(y.max()) + 1
    subset = training_subset(y, is_val, shots, rep)
    fold = stratified_folds(y[subset], seed=rep)
    train, val = torch.from_numpy(x[subset]).to(device), torch.from_numpy(x[is_val]).to(device)
    y_train, y_val = torch.from_numpy(y[subset]).to(device), torch.from_numpy(y[is_val]).to(device)
    out = {"features": x.shape[1], "linear": linear_probe(train, y_train, val, y_val, classes, fold)}
    for k in SELECT_K:
        if k < x.shape[1]:
            units = selected_units(train, y_train, classes, fold, k)
            out[f"select{k}"] = linear_probe(train[:, units], y_train, val[:, units], y_val, classes, fold)
            out[f"select{k}"]["blocks"] = block_histogram(block_of, units.cpu().numpy())
    out["mlp"] = mlp_probe(train, y_train, val, y_val, classes, fold)
    return out


def done_keys():
    keys = set()
    for path in OUT.glob("results_*.jsonl"):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            keys.add((r["dataset"], row_key(r["row"]), r["shots"], r["rep"]))
    return keys


def jobs():
    """One job per (dataset, row, replicate): features load once, then every shot count runs on them."""
    return [(dataset, row, rep) for dataset in DATASETS for row in grid_rows() for rep in range(REPLICATES)]


def run(worker, workers, device):
    done = done_keys()
    out = open(OUT / f"results_{worker}.jsonl", "a")
    for i, (dataset, row, rep) in enumerate(jobs()):
        todo = [s for s in SHOTS if (dataset, row_key(row), s, rep) not in done]
        if i % workers != worker or not todo:
            continue
        x, block_of = features(dataset, row, rep)
        y, is_val = labels_and_split(dataset)
        for shots in todo:
            result = evaluate(x, block_of, y, is_val, shots, rep, device)
            out.write(json.dumps({"dataset": dataset, "row": row, "shots": shots, "rep": rep, **result}) + "\n")
            out.flush()
            print(dataset, row, shots, rep, {k: v["val_acc"] for k, v in result.items() if isinstance(v, dict)}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.worker, args.workers, torch.device("cuda"))


if __name__ == "__main__":
    main()
