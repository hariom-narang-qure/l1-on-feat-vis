"""Phase 5: units that fix the head's misses (recall) and its false positives (precision).

Errors come from out-of-fold head predictions on the train scans
(head.out_of_fold: train on 8 scans, predict the other 4, rotate), mapped onto
the store rows (each row is one 6.9 px token cell):
  - missed row: a line row where most of the cell's line pixels have no
    predicted skeleton within 2 px;
  - false-positive row: a non-line row whose cell holds predicted skeleton
    pixels with no line within 2 px.
The store's rows already include every line token and many hard negatives
(ring, pleura, vessel, uniform), so no new forward passes are needed to score
candidates on these cells.

Candidate search over every unit of the store:
  - recall:    AUROC of missed rows vs ring rows, per fold; a candidate must be
               >= MIN_AUC (0.60, the Phase 2 bar) in every fold (fires on the fissures the head misses,
               not just near them);
  - precision: AUROC of false-positive rows vs found line rows, per fold, the
               unit's sign flipped so "high = false positive"; >= MIN_AUC in
               every fold. In the head it is negative evidence.
Gain test: a token-level stacked model, logit(line) = a * logit(head cell
score) + b, fitted per fold on the other folds; candidates are scored by the
held-out score statistic of forward_select (minimum over folds, sign-consistent),
so a candidate must explain head errors on scans the head was not trained on.
The best of each kind (decorrelated) are added to the pool, flagged
found_on_misses / found_on_false_positives, and confirmed by retraining the head.
"""

import json

import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import skeletonize

from l1_on_feat_vis.fissure_probe.forward_select import MAX_CORR, SELECTION, fit_logistic, predict, training_weights
from l1_on_feat_vis.fissure_probe.rows import FOLDS, GRID, Rows
from l1_on_feat_vis.fissure_probe.seg_metrics import drop_small_components
from l1_on_feat_vis.fissure_probe.store import layer_list, load_layer
from l1_on_feat_vis.fissure_probe.unit_scan import auroc_columns
from l1_on_feat_vis.fissure_probe.views import BITS, STORE

MIN_AUC = 0.60  # observed best min-fold AUROC: 0.646 (recall), 0.629 (precision); 0.70 passes nothing
TOL = 2
CELL = 512 / GRID


def cell_of_pixels(shape=(512, 512)):
    r, c = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return (np.minimum((r / CELL).astype(int), GRID - 1) * GRID + np.minimum((c / CELL).astype(int), GRID - 1))


CELL_OF = cell_of_pixels()


def slice_errors(probability, labels, threshold):
    """Per token cell: (missed line fraction, false skeleton pixels, head cell score); threshold = (t, min size)."""
    lung = (labels & BITS["lung"]) > 0
    line = (labels & BITS["line"]) > 0
    prediction = drop_small_components(probability > threshold[0], threshold[1]) & lung
    skeleton = skeletonize(prediction) if prediction.any() else prediction
    near_pred = ndimage.distance_transform_edt(~skeleton) <= TOL if skeleton.any() else np.zeros_like(line)
    near_line = ndimage.distance_transform_edt(~line) <= TOL if line.any() else np.zeros_like(line)
    count = lambda mask: np.bincount(CELL_OF[mask], minlength=GRID * GRID)
    line_px = count(line)
    missed_fraction = count(line & ~near_pred) / np.maximum(line_px, 1)
    false_px = count(skeleton & ~near_line)
    cell_score = ndimage.maximum(probability, CELL_OF, np.arange(GRID * GRID))
    return missed_fraction, false_px, cell_score


class RowErrors:
    """Missed / false-positive flags and the head's cell score for every store row, from OOF predictions."""

    def __init__(self, rows, oof):
        self.missed = np.zeros(rows.n, bool)
        self.false_positive = np.zeros(rows.n, bool)
        self.head_score = np.zeros(rows.n, np.float32)
        for f, part in oof.items():
            for scan, predictions in self.by_scan(part):
                scan_index = rows.scans.index(scan)
                cache = np.load(STORE / "slices" / f"{scan}.npz")
                index = rows.of_scan(scan_index)
                for s in np.unique(rows.slice[index]):
                    these = index[rows.slice[index] == s]
                    missed, false_px, score = slice_errors(predictions[s].astype(np.float32), cache["labels"][s], part["threshold"])
                    tokens = rows.token[these]
                    self.missed[these] = rows.line[these] & (missed[tokens] > 0.5)
                    self.false_positive[these] = ~rows.line[these] & (false_px[tokens] > 0)
                    self.head_score[these] = score[tokens]

    @staticmethod
    def by_scan(part):
        """OOF predictions are stacked in SliceData order (scans in split order, slices in cache order)."""
        offset = 0
        for scan in part["scans"]:
            n = len(np.load(STORE / "slices" / f"{scan}.npz")["k"])
            yield scan, part["predictions"][offset:offset + n]
            offset += n


def contrast_auc(x, positive, negative, fold, device):
    """(folds, units) AUROC of positive vs negative rows per fold."""
    out = []
    for f in range(FOLDS):
        m = (fold == f) & (positive | negative)
        xs = torch.from_numpy(x[m]).to(device)
        out.append(auroc_columns(xs, torch.from_numpy(positive[m]).to(device)).cpu().numpy())
    return np.stack(out)


def search(rows, errors, device):
    """Units passing the recall / precision filters in every fold, over all store layers."""
    found = {"recall": [], "precision": []}
    found_line = rows.line & ~errors.missed
    for layer in layer_list():
        x = load_layer(layer, rows.scans)
        recall = contrast_auc(x, errors.missed, rows.ring & ~rows.line, rows.fold, device)
        precision = contrast_auc(x, errors.false_positive, found_line, rows.fold, device)
        for kind, auc in (("recall", recall), ("precision", precision)):
            sign = np.where(auc.mean(0) < 0.5, -1, 1)
            signed = np.where(sign < 0, 1 - auc, auc)
            for u in np.flatnonzero(signed.min(0) >= MIN_AUC):
                found[kind].append({"layer": layer, "unit": int(u), "sign": int(sign[u]),
                                    "fold_auc": signed[:, u].round(4).tolist()})
    return found


def stacked_score_tests(rows, errors, candidates_x, device):
    """Held-out score statistic of each candidate against the stacked head model (min over folds)."""
    g = lambda a, dtype=None: torch.from_numpy(np.asarray(a)).to(device, dtype)
    y, w, fold = g(rows.line).double(), g(training_weights(rows), torch.float64), g(rows.fold)
    head = torch.logit(g(errors.head_score).double().clamp(1e-4, 1 - 1e-4))[:, None].float()
    logit = torch.zeros(rows.n, device=device, dtype=torch.float64)
    for f in range(FOLDS):
        train = fold != f
        beta = fit_logistic(head[train], y[train], w[train])
        logit[~train] = predict(head[~train], beta)
    p = torch.sigmoid(logit)
    x = g(candidates_x).float()
    stats = []
    for f in range(FOLDS):
        held = fold == f
        r, c = (w * (y - p))[held].float(), (w * p * (1 - p))[held].float()
        stats.append((x[held].T @ r) / ((x[held] ** 2).T @ c).clamp_min(1e-9).sqrt())
    stats = torch.stack(stats)
    sign = torch.sign(stats.sum(0))
    return (stats * sign).min(0).values.cpu().numpy(), sign.cpu().numpy()


def pick_decorrelated(x, scores, count, max_corr=MAX_CORR):
    order = np.argsort(-scores)
    chosen = []
    sample = np.random.default_rng(0).choice(len(x), min(20000, len(x)), replace=False)
    for c in order:
        if scores[c] <= 0 or len(chosen) == count:
            break
        if chosen and np.abs(np.corrcoef(x[sample][:, chosen + [c]].T)[-1, :-1]).max() > max_corr:
            continue
        chosen.append(c)
    return chosen


def finesse(oof, pool, per_kind, device, name):
    rows = Rows()
    errors = RowErrors(rows, oof)
    found = search(rows, errors, device)
    in_pool = {(p["layer"], p["unit"]) for p in pool}
    report = {"missed_rows": int(errors.missed.sum()), "line_rows": int(rows.line.sum()),
              "false_positive_rows": int(errors.false_positive.sum()),
              "passing": {k: len(v) for k, v in found.items()}}
    additions = []
    for kind, flag in (("recall", "found_on_misses"), ("precision", "found_on_false_positives")):
        members = [c for c in found[kind] if (c["layer"], c["unit"]) not in in_pool]
        if not members:
            continue
        x = np.concatenate([load_layer(c["layer"], rows.scans, [c["unit"]]) * c["sign"] for c in members], 1)
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        scores, _ = stacked_score_tests(rows, errors, x, device)
        chosen = pick_decorrelated(x, scores, per_kind)
        report[f"{kind}_scores"] = [float(scores[c]) for c in chosen]
        additions += [{**members[c], flag: True, "stacked_score": float(scores[c])} for c in chosen]
    (SELECTION / f"{name}.json").write_text(json.dumps({"report": report, "found": found, "additions": additions}, indent=1))
    return additions, report, errors


def load_oof(tag):
    """The OOF dict of head.run_oof outputs: {fold: {scans, threshold, predictions}}."""
    heads = STORE / "heads"
    summary = json.loads((heads / f"{tag}.json").read_text())
    predictions = np.load(heads / f"{tag}_predictions.npz")
    return {int(f): {"scans": part["scans"], "threshold": tuple(part["threshold"]), "predictions": predictions[f"fold{f}"]}
            for f, part in summary["folds"].items()}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True, help="head OOF tag, e.g. e1_step28_k0_oof")
    parser.add_argument("--pool", required=True, help="selection json with the head's pool")
    parser.add_argument("--per-kind", type=int, default=16)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    pool = json.loads(open(args.pool).read())["pool"]
    additions, report, _ = finesse(load_oof(args.oof), pool, args.per_kind, torch.device("cuda"), args.name)
    print(json.dumps(report, indent=1))
    for a in additions:
        kind = "recall" if a.get("found_on_misses") else "precision"
        print(f"{kind:9s} {a['layer']:20s} {a['unit']:5d} sign {a['sign']:+d} fold AUC {a['fold_auc']} stacked {a['stacked_score']:.2f}")
    (SELECTION / f"{args.name}_pool.json").write_text(json.dumps({"pool": pool + additions}, indent=1))


if __name__ == "__main__":
    main()
