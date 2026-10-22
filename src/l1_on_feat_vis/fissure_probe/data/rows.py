"""The shared row table: which token cells of which slices the activation store holds.

Every column of the store (one unit of one layer of one model) is read at the
same rows, so units of different models can be compared and pooled.

Token grid: 74 x 74 over the 512 px canvas (views.py), 6.9 px per token; it is
DINOv2's patch grid at 1036 px input, and every other network is resampled onto it.
Per token, label fractions are pooled from the canvas labels:
  - line:   any line pixel in the token (max pool)
  - lung:   lung fraction >= 0.5, or a line token
  - ring:   ring fraction >= 0.5, not line
  - pleura: pleura fraction >= 0.25, not line
  - vessel: vessel fraction >= 0.25, not line, not pleura
Rows per slice (train scans only; test scans are never used for selection):
  - every line token
  - ring tokens, up to 3 x the line tokens
  - up to 40 pleura and 40 vessel tokens (hard negatives)
  - 80 uniform lung non-line tokens; `weight` = lung non-line tokens / 80, so
    line rows (weight 1) + uniform rows (weight) estimate the whole lung at its
    natural frequency (for AP and probe loss over "everything else").
A token picked by several strata is stored once, with every flag set.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from l1_on_feat_vis.fissure_probe.sheet_probing.probes import split_scans
from l1_on_feat_vis.fissure_probe.data.views import BITS, STORE

GRID = 74
RING_PER_LINE, PLEURA_ROWS, VESSEL_ROWS, UNIFORM_ROWS = 3, 40, 40, 80
FOLDS = 3


def pooled(labels, name, mode):
    mask = torch.from_numpy(((labels & BITS[name]) > 0).astype(np.float32))[:, None]
    pool = F.adaptive_max_pool2d if mode == "max" else F.adaptive_avg_pool2d
    return pool(mask, GRID)[:, 0].flatten(1).numpy()


def token_labels(labels):
    """(slices, GRID*GRID) boolean strata from canvas label bitmaps (slices, 512, 512)."""
    line = pooled(labels, "line", "max") > 0
    lung = (pooled(labels, "lung", "avg") >= 0.5) | line
    ring = (pooled(labels, "ring", "avg") >= 0.5) & ~line
    pleura = (pooled(labels, "pleura", "avg") >= 0.25) & ~line
    vessel = (pooled(labels, "vessel", "avg") >= 0.25) & ~line & ~pleura
    return {"line": line, "lung": lung, "ring": ring & lung, "pleura": pleura & lung, "vessel": vessel & lung}


def pick(rng, candidates, count):
    return candidates if len(candidates) <= count else rng.choice(candidates, count, replace=False)


def slice_rows(strata, s, rng):
    line = np.flatnonzero(strata["line"][s])
    negatives = strata["lung"][s] & ~strata["line"][s]
    uniform_pool = np.flatnonzero(negatives)
    chosen = {
        "line": line,
        "ring": pick(rng, np.flatnonzero(strata["ring"][s]), RING_PER_LINE * len(line)),
        "pleura": pick(rng, np.flatnonzero(strata["pleura"][s]), PLEURA_ROWS),
        "vessel": pick(rng, np.flatnonzero(strata["vessel"][s]), VESSEL_ROWS),
        "uniform": pick(rng, uniform_pool, UNIFORM_ROWS),
    }
    tokens = np.unique(np.concatenate(list(chosen.values())))
    flags = {f"in_{name}": np.isin(tokens, picked) for name, picked in chosen.items()}
    weight = len(uniform_pool) / max(len(chosen["uniform"]), 1)
    return tokens, flags, weight


def fold_of_scans(train_scans, seed=0):
    """3 folds of 4 train scans, 2 review_v1 + 2 severe each."""
    from l1_on_feat_vis.fissure_probe.data.data import list_scans
    cohort = {r.short: r.cohort for r in list_scans()}
    rng = np.random.default_rng(seed)
    folds = {}
    for name in ("review_v1", "severe"):
        members = rng.permutation(sorted(s for s in train_scans if cohort[s] == name))
        for i, scan in enumerate(members):
            folds[str(scan)] = i % FOLDS
    return folds


def build(out_path=STORE / "rows.npz", seed=0):
    train, _, _ = split_scans()
    folds = fold_of_scans(train)
    rng = np.random.default_rng(seed)
    columns = {key: [] for key in ("scan", "fold", "slice", "view", "k", "token", "weight",
                                   "line", "ring", "pleura", "vessel",
                                   "in_line", "in_ring", "in_pleura", "in_vessel", "in_uniform")}
    for scan_index, scan in enumerate(train):
        cache = np.load(STORE / "slices" / f"{scan}.npz")
        strata = token_labels(cache["labels"])
        for s in range(len(cache["k"])):
            tokens, flags, weight = slice_rows(strata, s, rng)
            n = len(tokens)
            columns["scan"].append(np.full(n, scan_index, np.int8))
            columns["fold"].append(np.full(n, folds[scan], np.int8))
            columns["slice"].append(np.full(n, s, np.int16))
            columns["view"].append(np.full(n, cache["view"][s], np.int8))
            columns["k"].append(np.full(n, cache["k"][s], np.int16))
            columns["token"].append(tokens.astype(np.int16))
            columns["weight"].append(np.full(n, weight, np.float32))
            for name in ("line", "ring", "pleura", "vessel"):
                columns[name].append(strata[name][s][tokens])
            for name, flag in flags.items():
                columns[name].append(flag)
    arrays = {key: np.concatenate(values) for key, values in columns.items()}
    np.savez(out_path, scans=np.array(train), **arrays)
    return arrays


class Rows:
    """The row table, loaded."""

    def __init__(self, path=STORE / "rows.npz"):
        data = np.load(path)
        self.scans = list(data["scans"])
        for key in data.files:
            if key != "scans":
                setattr(self, key, data[key])
        self.n = len(self.token)

    def of_scan(self, scan_index):
        return np.flatnonzero(self.scan == scan_index)


if __name__ == "__main__":
    a = build()
    n = len(a["token"])
    print(f"{n} rows; line {a['line'].sum()} ring {a['in_ring'].sum()} pleura {a['in_pleura'].sum()} "
          f"vessel {a['in_vessel'].sum()} uniform {a['in_uniform'].sum()}")
    for f in range(FOLDS):
        print(f"fold {f}: scans {sorted(set(a['scan'][a['fold'] == f]))} rows {(a['fold'] == f).sum()} line {(a['line'] & (a['fold'] == f)).sum()}")
