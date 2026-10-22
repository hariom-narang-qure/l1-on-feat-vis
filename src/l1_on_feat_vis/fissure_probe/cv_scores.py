"""Phase 2: score every unit of the store per fold and per view, and flag overfitters.

Rows (rows.py) come from the 12 train scans, split into 3 folds of 4 scans.
Per unit, per fold, per view (all, axial, coronal, sagittal):
  - auc_ring / auc_vessel / auc_pleura: line tokens vs tokens of that stratum
    (AUROC from ranks, as in unit_scan.py). Ring asks "the fissure itself, not
    where fissures run"; vessel and pleura are the bright structures a
    fissure detector confuses it with.
  - ap_lung: AP of line tokens against all lung at its natural frequency, from
    line rows (weight 1) and uniform rows (weight = lung tokens / sampled):
    precision at a rank = weighted TP / weighted (TP + FP).
The sign of a unit (a unit may mark fissure by switching off) is the sign of
its mean fold auc_ring - 0.5; every number is reported with that sign. The
sign is one bit chosen on all train rows, a negligible leak.

Generalisation, per fold with view "all" (thresholds set from the observed
distribution: no unit reaches a min-fold ring AUROC of 0.70 on the per-view
labels, best 0.698; ring tokens are the line token's direct neighbours):
  - generaliser: min-fold auc_ring >= 0.60, or min-fold AP lift (AP / prior) >= 3
  - overfitter:  strong in one fold only: max auc_ring >= 0.65 with min < 0.55,
    or max lift >= 3 with min < 1.5. Overfitters are never candidates.
Output: scores/<model>__<layer>.npz with metrics (units, folds, 4 views, 4 metrics), sign, prior.
"""

import argparse
import time

import numpy as np
import torch

from l1_on_feat_vis.fissure_probe.rows import FOLDS, Rows
from l1_on_feat_vis.fissure_probe.store import ACTS, load_layer, layer_list
from l1_on_feat_vis.fissure_probe.unit_scan import auroc_columns
from l1_on_feat_vis.fissure_probe.views import STORE

SCORES = STORE / "scores"
METRICS = ("auc_ring", "auc_vessel", "auc_pleura", "ap_lung")
VIEW_SETS = ("all", "axial", "coronal", "sagittal")
RING_MIN, LIFT_MIN = 0.60, 3.0
OVERFIT_RING, OVERFIT_LIFT = (0.65, 0.55), (3.0, 1.5)
CHUNK = 1024


def weighted_ap(x, positive, weight):
    """AP per column with row weights; x (n, C), positive (n,), weight (n,)."""
    order = torch.argsort(x, dim=0, descending=True)
    hits = positive[order].float()
    w = weight[order]
    tp = (hits * w).cumsum(0)
    seen = w.cumsum(0)
    return (hits * w * tp / seen).sum(0) / (hits * w).sum(0)


class RowMasks:
    """Boolean row masks per (fold, view) and per contrast, on the GPU."""

    def __init__(self, rows, device):
        g = lambda a: torch.from_numpy(np.asarray(a)).to(device)
        self.line, self.weight = g(rows.line), g(rows.weight)
        self.natural = g(rows.line | rows.in_uniform)
        self.natural_weight = torch.where(self.line, torch.ones_like(self.weight), self.weight)
        self.contrast = {"auc_ring": g(rows.ring), "auc_vessel": g(rows.vessel), "auc_pleura": g(rows.pleura)}
        self.subsets = {}
        for f in range(FOLDS):
            for v, name in enumerate(VIEW_SETS):
                mask = rows.fold == f
                if name != "all":
                    mask = mask & (rows.view == v - 1)
                self.subsets[(f, v)] = g(mask)

    def prior(self, f, v):
        m = self.subsets[(f, v)] & self.natural
        w = self.natural_weight[m]
        return float((w * self.line[m]).sum() / w.sum())


def score_block(x, masks):
    """(C, folds, views, metrics) raw (unsigned) metrics, plus AP for the negated unit."""
    C = x.shape[1]
    out = torch.zeros(C, FOLDS, len(VIEW_SETS), len(METRICS), device=x.device)
    ap_negated = torch.zeros(C, FOLDS, len(VIEW_SETS), device=x.device)
    for (f, v), subset in masks.subsets.items():
        for m, metric in enumerate(METRICS[:3]):
            rows = subset & (masks.line | masks.contrast[metric])
            out[:, f, v, m] = auroc_columns(x[rows], masks.line[rows])
        rows = subset & masks.natural
        xs, ys, ws = x[rows], masks.line[rows], masks.natural_weight[rows]
        out[:, f, v, 3] = weighted_ap(xs, ys, ws)
        ap_negated[:, f, v] = weighted_ap(-xs, ys, ws)
    return out, ap_negated


def score_layer(layer, rows, masks, device):
    x_all = load_layer(layer, rows.scans)
    C = x_all.shape[1]
    raw = np.zeros((C, FOLDS, len(VIEW_SETS), len(METRICS)), np.float32)
    ap_neg = np.zeros((C, FOLDS, len(VIEW_SETS)), np.float32)
    for start in range(0, C, CHUNK):
        x = torch.from_numpy(x_all[:, start:start + CHUNK]).to(device)
        a, b = score_block(x, masks)
        raw[start:start + CHUNK], ap_neg[start:start + CHUNK] = a.cpu().numpy(), b.cpu().numpy()
    sign = np.where(raw[:, :, 0, 0].mean(1) < 0.5, -1, 1).astype(np.int8)
    signed = raw.copy()
    flip = sign < 0
    signed[flip, :, :, :3] = 1 - raw[flip, :, :, :3]
    signed[flip, :, :, 3] = ap_neg[flip]
    return signed, sign


def classify(signed, priors=None):
    priors = np.load(SCORES / "priors.npy")[:, 0] if priors is None else priors
    ring = signed[:, :, 0, 0]  # (units, folds), view "all"
    lift = signed[:, :, 0, 3] / priors
    overfitter = ((ring.max(1) >= OVERFIT_RING[0]) & (ring.min(1) < OVERFIT_RING[1])) | \
                 ((lift.max(1) >= OVERFIT_LIFT[0]) & (lift.min(1) < OVERFIT_LIFT[1]))
    generaliser = ((ring.min(1) >= RING_MIN) | (lift.min(1) >= LIFT_MIN)) & ~overfitter
    return generaliser, overfitter


def load_scores(layer):
    data = np.load(SCORES / f"{layer.replace('/', '__')}.npz")
    return data["metrics"], data["sign"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="*")
    args = parser.parse_args()

    device = torch.device("cuda")
    rows = Rows()
    masks = RowMasks(rows, device)
    SCORES.mkdir(parents=True, exist_ok=True)
    priors = np.array([[masks.prior(f, v) for v in range(len(VIEW_SETS))] for f in range(FOLDS)])
    np.save(SCORES / "priors.npy", priors)
    for layer in args.layers or layer_list():
        path = SCORES / f"{layer.replace('/', '__')}.npz"
        if path.exists():
            continue
        start = time.time()
        signed, sign = score_layer(layer, rows, masks, device)
        np.savez(path, metrics=signed, sign=sign)
        generaliser, overfitter = classify(signed, priors[:, 0])
        best = signed[:, :, 0, 0].min(1).argmax()
        print(f"{layer:22s} {signed.shape[0]:5d} units | generalisers {generaliser.sum():5d} overfitters {overfitter.sum():4d} | "
              f"best min-fold ring AUC {signed[best, :, 0, 0].min():.3f} (unit {best}), its AP lift "
              f"{(signed[best, :, 0, 3] / priors[:, 0]).mean():.1f}x | {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
