"""Pick fissure units across many networks with one sparse probe over all of them.

The feature folders (slice_features.py for 2D nets, nnunet_features.py for
3D nnU-Nets) sample the same slices and tokens in the same order, so their rows
are the same cells and their columns can be concatenated (checked: labels must
match row for row).
  1. Candidates: per block, the 8 units with the highest train fissure-vs-ring
     AUROC (sign chosen on train), from pretrained networks only.
  2. Pool them, standardise on train, and order them by the L1 path on the
     train fissure-vs-ring rows (probes.l1_entry_order).
  3. Refit a logistic probe on the top k and score it on the test scans:
     fissure-vs-ring AUROC, fissure AP, lobe-boundary AP.
The chosen set is written to selected_units.json for the segmentation head.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from l1_on_feat_vis.fissure_probe.sheet_probing.probes import (BlockRows, Standardiser, evaluate_score, fit_probe, l1_entry_order,
                                                 split_scans)
from l1_on_feat_vis.fissure_probe.sheet_probing.unit_scan import auroc_columns

PER_BLOCK = 8
TOP_K = (4, 8, 16, 32, 64)


def blocks_in(feature_dir):
    sample = np.load(next(feature_dir.glob("*.npz")))
    return sorted({key.rsplit("/", 1)[0] for key in sample.files if key.endswith("/pretrained")})


def top_units(rows, device):
    either = rows.fissure | rows.ring
    x = torch.from_numpy(rows.x[either]).to(device)
    auc = auroc_columns(x, torch.from_numpy(rows.fissure[either]).to(device)).cpu().numpy()
    signs = np.where(auc < 0.5, -1.0, 1.0)
    score = np.maximum(auc, 1 - auc)
    order = np.argsort(-score)[:PER_BLOCK]
    return order, signs[order], score[order]


class Pool:
    def __init__(self, feature_dirs, device):
        train_scans, test_scans, _ = split_scans()
        self.columns, self.train_x, self.test_x = [], [], []
        self.train = self.test = None
        for feature_dir in feature_dirs:
            for block in blocks_in(feature_dir):
                train = BlockRows(feature_dir, block, train_scans, "pretrained", uniform_only=False)
                test = BlockRows(feature_dir, block, test_scans, "pretrained", uniform_only=True)
                self.check_alignment(train, test)
                units, signs, scores = top_units(train, device)
                self.train_x.append(train.x[:, units] * signs)
                self.test_x.append(test.x[:, units] * signs)
                self.columns += [{"dir": str(feature_dir), "block": block, "unit": int(u), "sign": int(s),
                                  "train_ring_auc": float(a)} for u, s, a in zip(units, signs, scores)]
        self.train_x, self.test_x = np.concatenate(self.train_x, 1), np.concatenate(self.test_x, 1)

    def check_alignment(self, train, test):
        if self.train is None:
            self.train, self.test = train, test
            return
        assert np.array_equal(train.fissure, self.train.fissure) and np.array_equal(test.ring, self.test.ring), \
            "feature folders do not sample the same rows"

    def select(self):
        standardise = Standardiser(self.train_x)
        keep = self.train.fissure | self.train.ring
        x_train, y_train = standardise(self.train_x)[keep], self.train.fissure[keep]
        x_test = standardise(self.test_x)
        rng = np.random.default_rng(0)
        order = l1_entry_order(x_train, y_train, rng)
        results = []
        for k in TOP_K:
            chosen = order[:k]
            probe = fit_probe(x_train, y_train, chosen, rng)
            score = probe.decision_function(x_test[:, chosen])
            results.append({"k": k, "units": [self.columns[i] for i in chosen],
                            **{name: float(v) for name, v in evaluate_score(score, self.test).items()}})
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/fissure_probe/pooled"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pool = Pool(args.features, torch.device(args.device))
    print(f"{len(pool.columns)} candidate units from {len({c['block'] for c in pool.columns})} blocks", flush=True)
    results = pool.select()
    for r in results:
        models = sorted({c["block"].split("/")[0] for c in r["units"]})
        print(f"k={r['k']:3d} ring-AUC {r['fissure_vs_ring_auc']:.3f} AP {r['fissure_ap']:.3f} lobeAP {r['lobe_boundary_ap']:.3f} | models {models}", flush=True)
    (args.out / "selected_units.json").write_text(json.dumps({"candidates": pool.columns, "probes": results}, indent=1))


if __name__ == "__main__":
    main()
