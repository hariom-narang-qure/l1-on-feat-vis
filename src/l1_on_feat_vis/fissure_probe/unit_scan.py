"""Score every unit of a block at once on the GPU, then probe the best 64.

probes.py fits sklearn models over all units and scores units one by one,
which takes hours for 3072 DINOv2 MLP neurons. Here:
  - AUROC per unit from ranks: (sum of positive ranks - P(P+1)/2) / (P * N).
  - AP per unit: sort each column, cumulative true positives, mean precision
    at the positives. Ties are broken arbitrarily, which is harmless for
    continuous activations.
Same protocol as probes.py: sign and ranking from fissure-vs-ring AUROC on the
train scans; every number reported is on the test scans (uniform rows).
The sparse probe runs on the 64 best train units only (L1 path entry order,
then a refit on the top k).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from l1_on_feat_vis.fissure_probe.probes import (BlockRows, Standardiser, evaluate_score, fit_probe,
                                                 l1_entry_order, split_scans)

PREFILTER = 64
TOP_K = (1, 2, 4, 8, 16, 32, 64)


def auroc_columns(x, positive):
    ranks = torch.argsort(torch.argsort(x, dim=0), dim=0).double() + 1
    p, n = positive.sum().double(), (~positive).sum().double()
    return ((ranks[positive].sum(0) - p * (p + 1) / 2) / (p * n)).float()


def ap_columns(x, positive):
    order = torch.argsort(x, dim=0, descending=True)
    hits = positive[order].float()
    precision_at = hits.cumsum(0) / torch.arange(1, len(x) + 1, device=x.device)[:, None]
    return (hits * precision_at).sum(0) / hits.sum(0)


class UnitScan:
    def __init__(self, feature_dir, block, network, device, seed=0):
        self.block, self.network, self.device = block, network, device
        train_scans, test_scans, _ = split_scans(seed)
        self.train = BlockRows(feature_dir, block, train_scans, network, uniform_only=False)
        self.test = BlockRows(feature_dir, block, test_scans, network, uniform_only=True)

    def gpu(self, array):
        return torch.from_numpy(np.ascontiguousarray(array)).to(self.device)

    def local_auroc(self, rows):
        either = rows.fissure | rows.ring
        return auroc_columns(self.gpu(rows.x[either]), self.gpu(rows.fissure[either]))

    def other_auroc(self, rows, other, signs):
        either = rows.fissure | other
        return auroc_columns(self.gpu(rows.x[either]) * signs, self.gpu(rows.fissure[either]))

    def unit_table(self):
        train_auc = self.local_auroc(self.train)
        signs = torch.where(train_auc < 0.5, -1.0, 1.0)
        train_score = torch.maximum(train_auc, 1 - train_auc)
        test_x = self.gpu(self.test.x) * signs
        test = {
            "fissure_vs_ring_auc": self.other_auroc(self.test, self.test.ring, signs),
            "fissure_vs_pleura_auc": self.other_auroc(self.test, self.test.pleura, signs),
            "fissure_ap": ap_columns(test_x, self.gpu(self.test.fissure)),
            "lobe_boundary_ap": ap_columns(test_x, self.gpu(self.test.lobe_boundary)),
        }
        order = torch.argsort(train_score, descending=True).cpu().numpy()
        table = [{"unit": int(u), "sign": int(signs[u]), "train_ring_auc": float(train_score[u]),
                  **{k: float(v[u]) for k, v in test.items()}} for u in order]
        return table, order

    def sparse_probe(self, order):
        candidates = order[:PREFILTER]
        standardise = Standardiser(self.train.x[:, candidates])
        keep = self.train.fissure | self.train.ring
        x_train = standardise(self.train.x[keep][:, candidates])
        y_train = self.train.fissure[keep]
        x_test = standardise(self.test.x[:, candidates])
        rng = np.random.default_rng(0)
        entry = l1_entry_order(x_train, y_train, rng)
        results = []
        for k in TOP_K:
            chosen = entry[:k]
            probe = fit_probe(x_train, y_train, chosen, rng)
            score = probe.decision_function(x_test[:, chosen])
            results.append({"k": k, "units": [int(u) for u in candidates[chosen]],
                            **{n: float(v) for n, v in evaluate_score(score, self.test).items()}})
        return results

    def summary(self):
        table, order = self.unit_table()
        return {
            "block": self.block, "network": self.network,
            "test_prior": float(self.test.fissure.mean()),
            "test_lobe_boundary_prior": float(self.test.lobe_boundary.mean()),
            "units": table[:30],
            "test_ring_auc_distribution": np.percentile([r["fissure_vs_ring_auc"] for r in table], [50, 90, 99, 100]).tolist(),
            "sparse_probe_local": self.sparse_probe(order),
            "baselines": {name: {k: float(v) for k, v in evaluate_score(getattr(self.test, name), self.test).items()}
                          for name in ("intensity", "sheetness")},
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--blocks", nargs="+", required=True)
    parser.add_argument("--networks", nargs="+", default=["pretrained", "random"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for network in args.networks:
        for block in args.blocks:
            s = UnitScan(args.features, block, network, torch.device(args.device)).summary()
            (args.out / f"{network}_{block.replace('/', '__')}.json").write_text(json.dumps(s, indent=1))
            top = s["units"][0]
            k8 = {r["k"]: r for r in s["sparse_probe_local"]}[8]
            print(f"{network:10s} {block:11s} prior {s['test_prior']:.3f} | top unit {top['unit']:4d}{top['sign']:+d} "
                  f"ring-AUC {top['fissure_vs_ring_auc']:.3f} AP {top['fissure_ap']:.3f} lobeAP {top['lobe_boundary_ap']:.3f} | "
                  f"k=8 probe ring-AUC {k8['fissure_vs_ring_auc']:.3f} AP {k8['fissure_ap']:.3f} | "
                  f"sheetness ring-AUC {s['baselines']['sheetness']['fissure_vs_ring_auc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
