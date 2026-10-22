"""Where the pooled head misses fissures, and which units see what it misses.

Everything here uses the 12 train scans only; the 5 test scans are never looked at.
  1. Out-of-fold misses. The train scans are split into 3 folds of 4. For
     each fold, a head (pooled64+ct, seg_head.py) is trained on the other 8
     scans' cached slices and run on this fold's scans, on the slices that
     the feature folders sampled (replayed with nnunet_features.SelectionReplay).
     A sampled fissure token is "missed" if no sheet pixel inside it reaches the
     head's train threshold.
  2. Misses by fissure type: the lobes either side of the token
     (lobes.nii.gz: 10/11 left oblique; 12/14 or 13/14 right oblique;
     12/13 right horizontal).
  3. Unit search over every unit of every block in the feature folders:
     AUROC of missed-fissure tokens vs their ring tokens (lung 3-8 voxels away,
     the same location control as before), computed per fold.
     A unit is kept only if
       - its AUROC is >= 0.6 in every fold (it was not picked on one fold's quirks), and
       - it adds to the current pool: a logistic probe on (pool score + unit)
         separates missed fissure from ring better than the pool score alone,
         by >= 0.01 AUROC on the held-out fold of a leave-one-fold-out fit.
     Survivors are ranked by their worst-fold AUROC and flagged "found_on_misses".
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from l1_on_feat_vis.fissure_probe.data.data import LoadedScan, list_scans
from l1_on_feat_vis.fissure_probe.sheet_probing.dino_features import GRID, to_grid
from l1_on_feat_vis.fissure_probe.sheet_probing.multi_cache import UnitMaps
from l1_on_feat_vis.fissure_probe.models.nnunet_features import SelectionReplay
from l1_on_feat_vis.fissure_probe.sheet_probing.pooled_selection import blocks_in
from l1_on_feat_vis.fissure_probe.sheet_probing.probes import BlockRows, split_scans
from l1_on_feat_vis.fissure_probe.segmentation.seg_head import HeadTrainer, SliceSet, best_threshold, ct_input
from l1_on_feat_vis.fissure_probe.sheet_probing.unit_scan import auroc_columns

FOLDS = 3
MIN_FOLD_AUC = 0.6
MIN_GAIN = 0.01
FISSURE_TYPES = {frozenset({10, 11}): "left_oblique", frozenset({12, 14}): "right_oblique",
                 frozenset({13, 14}): "right_oblique", frozenset({12, 13}): "right_horizontal"}


def replay_selections():
    """{scan: [(z, tokens, uniform)]} exactly as the feature folders sampled them."""
    replay, out = SelectionReplay(), {}
    for record in list_scans():
        scan = LoadedScan(record)
        slices = []
        for z in replay.pick_slices(scan):
            z = int(z)
            lung = to_grid(scan.lung[:, :, z], "avg") > 0.5
            fissure = to_grid(scan.fissure[:, :, z], "max") > 0
            if not lung.any():
                continue
            tokens, uniform = replay.pick_tokens(lung.ravel(), fissure.ravel())
            slices.append((z, tokens, uniform))
        out[record.short] = (record, slices)
        print(f"replay: {record.short} {len(slices)} slices", flush=True)
    return out


def token_lobe_pairs(lobes_slice):
    """Per token, the set of lobe labels inside it (to name the fissure type)."""
    size = lobes_slice.shape[0]
    edges = np.linspace(0, size, GRID + 1).astype(int)
    return [[frozenset(np.unique(lobes_slice[edges[i]:edges[i + 1], edges[j]:edges[j + 1]])) - {0}
             for j in range(GRID)] for i in range(GRID)]


def fissure_type(labels):
    for pair, name in FISSURE_TYPES.items():
        if pair <= labels:
            return name
    return "unknown"


class FoldMisses:
    """Out-of-fold head predictions on the replayed slices of the train scans."""

    def __init__(self, seg_cache, multi_cache, unit_list, device):
        self.seg_cache, self.multi_cache, self.device = seg_cache, multi_cache, device
        self.unit_maps = UnitMaps(unit_list, device)

    def run(self, train_scans, selections):
        folds = np.array_split(np.random.default_rng(0).permutation(train_scans), FOLDS)
        rows = {}
        for f, fold in enumerate(folds):
            others = [s for s in train_scans if s not in fold]
            train = SliceSet(self.seg_cache, others, "pooled64+ct", None, self.multi_cache)
            trainer = HeadTrainer(train, self.device)
            trainer.fit()
            threshold = best_threshold(trainer, train)
            for scan in fold:
                rows[scan] = self.scan_rows(trainer, threshold, *selections[scan], fold_index=f)
            print(f"fold {f}: {list(fold)} threshold {threshold:.2f}", flush=True)
        return rows, [list(map(str, fold)) for fold in folds]

    def scan_rows(self, trainer, threshold, record, slices, fold_index):
        scan = LoadedScan(record)
        maps = np.zeros((len(slices), len(self.unit_maps.units), GRID, GRID), np.float32)
        for i, (z, _, _) in enumerate(slices):
            self.unit_maps.slice_maps_2d(scan.hu[:, :, z], maps, i)
        self.unit_maps.scan_maps_3d(record, [z for z, _, _ in slices], maps)
        missed, kinds = [], []
        for i, (z, tokens, _) in enumerate(slices):
            tokens_input = (maps[i] - trainer.token_mean) / trainer.token_std
            to = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(self.device)
            with torch.no_grad():
                prob = torch.sigmoid(trainer.model(to(tokens_input)[None], to(ct_input(scan.hu[:, :, z]))[None, None],
                                                   scan.hu.shape[:2]))[0].cpu().numpy()
            sheet = scan.sheet[:, :, z] & scan.lung[:, :, z]
            detected_sheet = sheet & (prob >= threshold)
            sheet_tokens = to_grid(sheet, "max") > 0
            found_tokens = to_grid(detected_sheet, "max") > 0
            pairs = token_lobe_pairs(scan.lobes[:, :, z])
            ti, tj = np.unravel_index(tokens, (GRID, GRID))
            missed.append(sheet_tokens[ti, tj] & ~found_tokens[ti, tj])
            kinds.append([fissure_type(pairs[a][b]) if sheet_tokens[a, b] else "" for a, b in zip(ti, tj)])
        return {"missed": np.concatenate(missed), "kind": sum(kinds, []), "fold": fold_index,
                "sheet_token": np.concatenate([(to_grid(scan.sheet[:, :, z] & scan.lung[:, :, z], "max") > 0).ravel()[t] for z, t, _ in slices])}


def per_type_summary(rows):
    counts = {}
    for r in rows.values():
        for missed, kind, on_sheet in zip(r["missed"], r["kind"], r["sheet_token"]):
            if on_sheet:
                total, miss = counts.get(kind, (0, 0))
                counts[kind] = (total + 1, miss + int(missed))
    return {kind: {"sheet_tokens": t, "missed": m, "miss_rate": m / max(t, 1)} for kind, (t, m) in counts.items()}


class MissUnitSearch:
    def __init__(self, feature_dirs, train_scans, rows, folds, pool_units, device):
        self.feature_dirs, self.train_scans, self.rows, self.folds = feature_dirs, train_scans, rows, folds
        self.device = device
        self.pool_keys = {(u["block"], u["unit"]) for u in pool_units}

    def labelled_rows(self, feature_dir, block):
        """Rows of the train scans with missed / ring labels and the fold of each row."""
        parts = []
        for scan in self.train_scans:
            data = BlockRows(feature_dir, block, [scan], "pretrained", uniform_only=False)
            r = self.rows[scan]
            assert len(r["missed"]) == len(data.fissure), f"{scan}: replay rows do not match {block}"
            keep = r["missed"] | (data.ring & ~data.fissure)
            parts.append((data.x[keep], r["missed"][keep], np.full(keep.sum(), r["fold"])))
        x, y, fold = (np.concatenate(p) for p in zip(*parts))
        return x, y, fold

    def fold_aucs(self, x, y, fold):
        """(units, folds) AUROC of missed vs ring, sign fixed on all train rows."""
        tx, ty = torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)
        overall = auroc_columns(tx, ty).cpu().numpy()
        signs = np.where(overall < 0.5, -1.0, 1.0)
        per_fold = []
        for f in range(len(self.folds)):
            m = torch.from_numpy(fold == f).to(self.device)
            per_fold.append(auroc_columns(tx[m] * torch.from_numpy(signs).to(self.device), ty[m]).cpu().numpy())
        return signs, np.stack(per_fold, 1)

    def candidates(self):
        found = []
        for feature_dir in self.feature_dirs:
            for block in blocks_in(feature_dir):
                x, y, fold = self.labelled_rows(feature_dir, block)
                if y.sum() < 50:
                    continue
                signs, aucs = self.fold_aucs(x.astype(np.float32), y, fold)
                worst = aucs.min(1)
                for unit in np.flatnonzero(worst >= MIN_FOLD_AUC):
                    if (block, int(unit)) in self.pool_keys:
                        continue
                    found.append({"dir": str(feature_dir), "block": block, "unit": int(unit), "sign": int(signs[unit]),
                                  "fold_aucs": aucs[unit].round(3).tolist(), "worst_fold_auc": float(worst[unit]),
                                  "found_on_misses": True})
                print(f"search {block}: {int((worst >= MIN_FOLD_AUC).sum())} units pass every fold", flush=True)
        return sorted(found, key=lambda c: -c["worst_fold_auc"])

    def incremental_gain(self, candidate, pool_score_rows):
        """Leave-one-fold-out AUROC gain of (pool score + unit) over pool score alone."""
        x, y, fold = self.labelled_rows(Path(candidate["dir"]), candidate["block"])
        unit = candidate["sign"] * x[:, candidate["unit"]].astype(np.float32)
        base = pool_score_rows[candidate["block_rows_key"]] if "block_rows_key" in candidate else None
        features_with = np.stack([base, unit], 1)
        gains = []
        for f in np.unique(fold):
            train, test = fold != f, fold == f
            if y[test].all() or not y[test].any():
                continue
            with_unit = LogisticRegression(class_weight="balanced", max_iter=500).fit(features_with[train], y[train])
            gains.append(roc_auc_score(y[test], with_unit.decision_function(features_with[test])) - roc_auc_score(y[test], base[test]))
        return float(np.mean(gains)) if gains else float("nan")


def pool_scores(feature_dirs, train_scans, rows, pool_units):
    """The current pool's probe score on the same missed/ring rows (logistic on the 64 pool units)."""
    columns = []
    for u in pool_units:
        parts = []
        for scan in train_scans:
            data = BlockRows(Path(u["dir"]), u["block"], [scan], "pretrained", uniform_only=False)
            r = rows[scan]
            keep = r["missed"] | (data.ring & ~data.fissure)
            parts.append(u["sign"] * data.x[keep, u["unit"]])
        columns.append(np.concatenate(parts))
    x = np.stack(columns, 1).astype(np.float32)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    y = np.concatenate([rows[s]["missed"][(lambda d, r: r["missed"] | (d.ring & ~d.fissure))(
        BlockRows(Path(pool_units[0]["dir"]), pool_units[0]["block"], [s], "pretrained", uniform_only=False), rows[s])] for s in train_scans])
    fold = np.concatenate([np.full(len(c), 0) for c in [y]])
    score = np.zeros(len(y))
    fold_of_row = np.concatenate([np.full(int((rows[s]["missed"] | (BlockRows(Path(pool_units[0]["dir"]), pool_units[0]["block"], [s], "pretrained", uniform_only=False).ring
                                                                     & ~BlockRows(Path(pool_units[0]["dir"]), pool_units[0]["block"], [s], "pretrained", uniform_only=False).fissure)).sum()), rows[s]["fold"]) for s in train_scans])
    for f in np.unique(fold_of_row):
        train, test = fold_of_row != f, fold_of_row == f
        model = LogisticRegression(class_weight="balanced", max_iter=500).fit(x[train], y[train])
        score[test] = model.decision_function(x[test])
    return score, y, fold_of_row
