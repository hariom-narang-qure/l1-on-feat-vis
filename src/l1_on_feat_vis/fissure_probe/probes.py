"""Sparse probes and single-channel scores for fissure detection, per Genesis block.

Split (by scan, stratified by cohort, fixed): 3 review_v1 + 2 severe test scans;
the rest train, except the two severe scans whose sheets are near-empty
(< 3000 voxels: disease hides the fissure), which are reported separately.

Per block and network (pretrained / random init):
  - Train rows: every sampled cell of the train scans (uniform + extra fissure).
    Test rows: only the uniform cells of the test scans, so fissure is at its
    natural frequency and average precision (AP) is honest. Chance AP = that
    frequency ("prior").
  - Single channels: a channel's sign is chosen on train (a channel may mark
    fissure by switching off), channels are ranked by train AP, and only then
    scored on test. Selecting from hundreds of channels on test would inflate it.
  - Sparse probe: L1 logistic regression over a grid of C. A channel's rank is
    the smallest C at which it becomes non-zero (the L1 path entry order,
    Gurnee et al. 2023 "Finding Neurons in a Haystack"). A logistic probe is
    then refit on the top-k channels for k = 1, 2, 4, 8, 16, 32, all.
  - Location control: a "ring" cell is lung 3-8 voxels from the sheet. Deep
    features know *where* in the lung a cell is, and fissures run through the
    interior while the pleura lines the edge, so plain AP rewards location
    (a random-init network scored as well as the pretrained one). Fissure-vs-
    ring AUROC compares cells with the same surroundings, so it only rewards
    responding to the fissure itself. Channels are also ranked by it on train,
    and a "local" sparse probe is trained on fissure-vs-ring rows only.
  - Also scored, with the fissure-trained rankings and probes (no refit):
    - AP against the TotalSegmentator lobe boundary: a fissure proxy the
      probe never saw.
    - Fissure-vs-pleura AUROC among cells that are one or the other: does it
      prefer fissures over the other thin bright sheet in the lung?
Baselines on the same test rows: intensity, Hessian sheetness.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from l1_on_feat_vis.fissure_probe.data import list_scans
from l1_on_feat_vis.fissure_probe.genesis import PROBED_BLOCKS

NEAR_EMPTY_SCANS = {"269479", "818602"}
TEST_SCANS = {"review_v1": 3, "severe": 2}
TOP_K = (1, 2, 4, 8, 16, 32)
L1_C_GRID = np.logspace(-4, 0, 25)
MAX_TRAIN_ROWS = 150_000


def split_scans(seed=0):
    rng = np.random.default_rng(seed)
    records = [r for r in list_scans() if r.short not in NEAR_EMPTY_SCANS]
    test = set()
    for cohort, count in TEST_SCANS.items():
        members = sorted(r.short for r in records if r.cohort == cohort)
        test |= {str(scan) for scan in rng.choice(members, count, replace=False)}
    train = {r.short for r in records} - test
    return sorted(train), sorted(test), sorted(NEAR_EMPTY_SCANS)


class BlockRows:
    """All sampled rows of one block, for a set of scans."""

    def __init__(self, feature_dir, block, scans, network, uniform_only):
        parts = [np.load(feature_dir / f"{scan}.npz") for scan in scans]
        keep = [p[f"{block}/uniform"] if uniform_only else np.ones(len(p[f"{block}/uniform"]), bool) for p in parts]
        take = lambda key: np.concatenate([p[f"{block}/{key}"][k] for p, k in zip(parts, keep)])
        self.x = take(network).astype(np.float32)
        self.fissure = take("fissure")
        self.lobe_boundary = take("lobe_boundary")
        self.pleura = take("pleura")
        self.ring = take("ring")
        self.intensity = take("intensity")
        self.sheetness = take("sheetness")


def channel_ap(x, y):
    return np.array([average_precision_score(y, x[:, c]) if x[:, c].std() > 0 else y.mean() for c in range(x.shape[1])])


def signed_channel_ranking(train):
    """Per channel: the sign that scores higher on train, and its train AP."""
    positive = channel_ap(train.x, train.fissure)
    negative = channel_ap(-train.x, train.fissure)
    signs = np.where(negative > positive, -1.0, 1.0)
    return signs, np.maximum(positive, negative)


def fissure_vs_other_auc(score, rows, other):
    either = rows.fissure | other
    if rows.fissure[either].all() or not rows.fissure[either].any():
        return float("nan")
    return roc_auc_score(rows.fissure[either], score[either])


def evaluate_score(score, rows):
    return {
        "fissure_ap": average_precision_score(rows.fissure, score),
        "lobe_boundary_ap": average_precision_score(rows.lobe_boundary, score),
        "fissure_vs_pleura_auc": fissure_vs_other_auc(score, rows, rows.pleura),
        "fissure_vs_ring_auc": fissure_vs_other_auc(score, rows, rows.ring),
    }


def local_channel_ranking(train):
    """Per channel: fissure-vs-ring AUROC on train, sign chosen so it is >= 0.5."""
    either = train.fissure | train.ring
    y = train.fissure[either]
    auc = np.array([roc_auc_score(y, train.x[either, c]) if train.x[either, c].std() > 0 else 0.5
                    for c in range(train.x.shape[1])])
    signs = np.where(auc < 0.5, -1.0, 1.0)
    return signs, np.maximum(auc, 1 - auc)


class Standardiser:
    def __init__(self, x):
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6

    def __call__(self, x):
        return (x - self.mean) / self.std


def l1_entry_order(x, y, rng):
    """Channels ordered by the smallest C at which L1 logistic regression uses them."""
    rows = rng.choice(len(y), min(len(y), MAX_TRAIN_ROWS // 3), replace=False)
    entry = np.full(x.shape[1], np.inf)
    for c in L1_C_GRID:
        model = LogisticRegression(penalty="l1", C=c, solver="liblinear", class_weight="balanced", max_iter=200)
        model.fit(x[rows], y[rows])
        newly_used = (np.abs(model.coef_[0]) > 0) & np.isinf(entry)
        entry[newly_used] = c
        if np.isfinite(entry).all():
            break
    # Never-entered channels go last, ordered by |correlation| with the label.
    correlation = np.abs(np.corrcoef(x[rows].T, y[rows])[-1, :-1])
    return np.lexsort((-np.nan_to_num(correlation), entry))


def fit_probe(x, y, channels, rng):
    rows = rng.choice(len(y), min(len(y), MAX_TRAIN_ROWS), replace=False)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500)
    model.fit(x[rows][:, channels], y[rows])
    return model


class BlockProbe:
    def __init__(self, feature_dir, block, network, train_scans, test_scans, near_empty_scans, seed=0):
        self.block, self.network = block, network
        self.rng = np.random.default_rng(seed)
        self.train = BlockRows(feature_dir, block, train_scans, network, uniform_only=False)
        self.test = BlockRows(feature_dir, block, test_scans, network, uniform_only=True)
        self.near_empty = BlockRows(feature_dir, block, near_empty_scans, network, uniform_only=True)

    def single_channels(self, ranking, top=10):
        signs, train_score = ranking(self.train)
        order = np.argsort(-train_score)[:top]
        return [
            {"channel": int(c), "sign": int(signs[c]), "train_score": float(train_score[c]),
             **{k: float(v) for k, v in evaluate_score(signs[c] * self.test.x[:, c], self.test).items()}}
            for c in order
        ]

    def sparse_probes(self, local=False):
        """local=True trains on fissure-vs-ring rows of the train scans only."""
        standardise = Standardiser(self.train.x)
        x_train, x_test = standardise(self.train.x), standardise(self.test.x)
        y_train = self.train.fissure
        if local:
            keep = self.train.fissure | self.train.ring
            x_train, y_train = x_train[keep], y_train[keep]
        order = l1_entry_order(x_train, y_train, self.rng)
        results = []
        for k in [k for k in TOP_K if k < x_train.shape[1]] + [x_train.shape[1]]:
            channels = order[:k]
            probe = fit_probe(x_train, y_train, channels, self.rng)
            score = probe.decision_function(x_test[:, channels])
            results.append({"k": k, "channels": channels[:32].tolist(),
                            "weights": probe.coef_[0][:32].round(4).tolist(),
                            **{name: float(v) for name, v in evaluate_score(score, self.test).items()}})
        return results

    def baselines(self):
        return {name: {k: float(v) for k, v in evaluate_score(getattr(self.test, name), self.test).items()}
                for name in ("intensity", "sheetness")}

    def summary(self):
        return {
            "block": self.block, "network": self.network,
            "test_prior": float(self.test.fissure.mean()),
            "test_lobe_boundary_prior": float(self.test.lobe_boundary.mean()),
            "test_rows": int(len(self.test.fissure)),
            "test_ring_rows": int(self.test.ring.sum()),
            "single_channels": self.single_channels(signed_channel_ranking),
            "single_channels_local": self.single_channels(local_channel_ranking),
            "sparse_probes": self.sparse_probes(),
            "sparse_probes_local": self.sparse_probes(local=True),
            "baselines": self.baselines(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("outputs/fissure_probe/features"))
    parser.add_argument("--out", type=Path, default=Path("outputs/fissure_probe/probes.json"))
    parser.add_argument("--blocks", nargs="+", default=PROBED_BLOCKS)
    parser.add_argument("--networks", nargs="+", default=["pretrained", "random"])
    args = parser.parse_args()

    train, test, near_empty = split_scans()
    print(f"train {train}\ntest {test}\nnear-empty {near_empty}", flush=True)
    results = {"split": {"train": train, "test": test, "near_empty": near_empty}, "blocks": []}
    for network in args.networks:
        for block in args.blocks:
            summary = BlockProbe(args.features, block, network, train, test, near_empty).summary()
            results["blocks"].append(summary)
            local_single = summary["single_channels_local"][0]
            local_probe = {r["k"]: r for r in summary["sparse_probes_local"]}
            print(f"{network:10s} {block:10s} | fissure-vs-ring AUROC: best channel {local_single['channel']:3d} "
                  f"{local_single['fissure_vs_ring_auc']:.3f}, local probe k=8 {local_probe.get(8, {}).get('fissure_vs_ring_auc', float('nan')):.3f} "
                  f"all {summary['sparse_probes_local'][-1]['fissure_vs_ring_auc']:.3f}, sheetness "
                  f"{summary['baselines']['sheetness']['fissure_vs_ring_auc']:.3f}, intensity "
                  f"{summary['baselines']['intensity']['fissure_vs_ring_auc']:.3f}", flush=True)
            args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
