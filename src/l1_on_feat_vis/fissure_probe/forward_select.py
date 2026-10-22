"""Phase 3: grow a unit pool in chunks by cross-validated gain, and prune units that do not help.

Probe: weighted logistic regression on standardised units (IRLS / Newton on the
GPU, small ridge), target = line token. Training weights per row:
  - line rows 1;
  - uniform lung rows their natural weight (lung tokens / sampled), so the
    probe sees "everything else" at its real frequency;
  - ring / pleura / vessel rows share a mass equal to HARD_FRACTION of the
    uniform mass, so hard negatives are not drowned out;
  - negatives rescaled so they weigh as much as the positives in total.
CV score of a pool: for each fold, fit on the other two folds (8 scans) and
score the held-out fold (4 scans): AP of line tokens against all lung at its
natural frequency (line + uniform rows), the token-level analogue of pixel AP.
The curve reports the mean over folds.

Forward step (a chunk of CHUNK units):
  1. Per fold, fit the current pool on the other folds and take the held-out
     residuals r = y - p.
  2. Score test of every candidate on each held-out fold:
        g = sum w r x / sqrt(sum w p (1 - p) x^2)
     ~ how much the held-out log-likelihood would improve if the unit were added.
  3. A candidate's score is the minimum over folds of its sign-consistent g:
     a unit that helps on one fold only scores ~0, so overfitters are never picked.
  4. Take the best candidates, skipping any with |correlation| > MAX_CORR to a
     unit already in the chunk (near-duplicates would split one gain).
Backward pruning (every PRUNE_EVERY chunks): drop-one CV score for every pool
unit (warm-started refits); units whose removal does not lower the CV AP are
removed, at most a quarter of the pool at a time.

Candidates: Phase 2 generalisers (cv_scores.py), capped to the MAX_CANDIDATES
with the best minimum-fold AP lift.
"""

import argparse
import json
import time

import numpy as np
import torch

from l1_on_feat_vis.fissure_probe.cv_scores import SCORES, classify, load_scores
from l1_on_feat_vis.fissure_probe.rows import FOLDS, Rows
from l1_on_feat_vis.fissure_probe.store import layer_list, load_layer
from l1_on_feat_vis.fissure_probe.views import STORE

HARD_FRACTION = 0.5
RIDGE = 1e-3
CHUNK, MAX_CORR, PRUNE_EVERY = 16, 0.9, 4
MAX_CANDIDATES = 8000
SELECTION = STORE / "selection"


def training_weights(rows):
    w = np.zeros(rows.n, np.float64)
    hard = (rows.in_ring | rows.in_pleura | rows.in_vessel) & ~rows.line
    uniform = rows.in_uniform & ~rows.line
    w[uniform] = rows.weight[uniform]
    hard_only = hard & ~uniform
    if hard_only.any():
        w[hard_only] = HARD_FRACTION * w[uniform].sum() / hard_only.sum()
    w[rows.line] = 1.0
    negatives = ~rows.line
    w[negatives] *= rows.line.sum() / w[negatives].sum()
    return w


def natural_weights(rows):
    """Line rows 1, uniform rows their natural weight, all else 0 (for AP)."""
    w = np.where(rows.in_uniform & ~rows.line, rows.weight, 0.0)
    return np.where(rows.line, 1.0, w)


def weighted_ap(score, positive, weight):
    keep = weight > 0
    score, positive, weight = score[keep], positive[keep], weight[keep]
    order = torch.argsort(score, descending=True)
    hits, w = positive[order].double(), weight[order]
    tp, seen = (hits * w).cumsum(0), w.cumsum(0)
    return float((hits * w * tp / seen).sum() / (hits * w).sum())


def auroc(score, positive):
    ranks = torch.argsort(torch.argsort(score)).double() + 1
    p, n = positive.sum().double(), (~positive).sum().double()
    return float((ranks[positive].sum() - p * (p + 1) / 2) / (p * n))


def fit_logistic(x, y, w, beta=None, iters=25):
    """Weighted ridge logistic regression by Newton steps; x (n, d) float32, returns (d + 1,) with bias last."""
    n, d = x.shape
    xb = torch.cat([x, torch.ones(n, 1, device=x.device)], 1).double()
    beta = torch.zeros(d + 1, device=x.device, dtype=torch.float64) if beta is None else beta.clone()
    penalty = torch.full((d + 1,), RIDGE * float(w.sum()), device=x.device, dtype=torch.float64)
    penalty[-1] = 0
    for _ in range(iters):
        p = torch.sigmoid(xb @ beta)
        gradient = xb.T @ (w * (p - y)) + penalty * beta
        hessian = (xb * (w * p * (1 - p))[:, None]).T @ xb + torch.diag(penalty)
        step = torch.linalg.solve(hessian, gradient)
        beta -= step
        if step.abs().max() < 1e-7:
            break
    return beta


def predict(x, beta):
    return x.double() @ beta[:-1] + beta[-1]


class Candidates:
    """Signed, standardised candidate units of the store, all rows, on the GPU (float16)."""

    def __init__(self, rows, device, layers=None, max_candidates=MAX_CANDIDATES, units=None):
        self.columns, parts = [], []
        chosen = units if units is not None else self.pick(layers or layer_list(), max_candidates)
        by_layer = {}
        for layer, unit, sign in chosen:
            by_layer.setdefault(layer, []).append((unit, sign))
        for layer, members in by_layer.items():
            units_here = [u for u, _ in members]
            signs = np.array([s for _, s in members], np.float32)
            x = load_layer(layer, rows.scans, units_here) * signs
            x = (x - x.mean(0)) / (x.std(0) + 1e-6)
            parts.append(torch.from_numpy(x.astype(np.float16)).to(device))
            self.columns += [{"layer": layer, "unit": int(u), "sign": int(s)} for u, s in members]
        self.x = torch.cat(parts, 1)

    @staticmethod
    def pick(layers, max_candidates):
        priors = np.load(SCORES / "priors.npy")[:, 0]
        pool = []
        for layer in layers:
            metrics, sign = load_scores(layer)
            generaliser, _ = classify(metrics)
            lift = (metrics[:, :, 0, 3] / priors).min(1)
            pool += [(lift[u], layer, int(u), int(sign[u])) for u in np.flatnonzero(generaliser)]
        pool.sort(key=lambda t: -t[0])
        return [(layer, u, s) for _, layer, u, s in pool[:max_candidates]]


class ForwardSelection:
    def __init__(self, rows, candidates, device):
        self.rows, self.cand, self.device = rows, candidates, device
        g = lambda a, dtype=None: torch.from_numpy(np.asarray(a)).to(device, dtype)
        self.y = g(rows.line).double()
        self.line = g(rows.line)
        self.w = g(training_weights(rows), torch.float64)
        self.natural = g(natural_weights(rows), torch.float64)
        self.fold = g(rows.fold)
        self.view = g(rows.view)
        self.strata = {"ring": g(rows.ring), "vessel": g(rows.vessel), "pleura": g(rows.pleura)}
        self.pool, self.curve, self.betas = [], [], {}

    def pool_x(self, pool):
        if not pool:
            return torch.zeros(self.rows.n, 0, device=self.device)
        return self.cand.x[:, pool].float()

    def fit_folds(self, pool, warm=None):
        x = self.pool_x(pool)
        betas, held_out_logit = {}, torch.zeros(self.rows.n, device=self.device, dtype=torch.float64)
        for f in range(FOLDS):
            train = self.fold != f
            start = None if warm is None else warm[f]
            betas[f] = fit_logistic(x[train], self.y[train], self.w[train], start, iters=25 if warm is None else 6)
            held_out_logit[~train] = predict(x[~train], betas[f])
        return betas, held_out_logit

    def cv_report(self, logit):
        report = {"ap": [], "ap_view": {v: [] for v in range(3)}, **{f"auc_{s}": [] for s in self.strata}}
        for f in range(FOLDS):
            held = self.fold == f
            report["ap"].append(weighted_ap(logit[held], self.line[held], self.natural[held]))
            for v in range(3):
                m = held & (self.view == v)
                report["ap_view"][v].append(weighted_ap(logit[m], self.line[m], self.natural[m]))
            for s, stratum in self.strata.items():
                m = held & (self.line | stratum)
                report[f"auc_{s}"].append(auroc(logit[m], self.line[m]))
        report["ap_mean"] = float(np.mean(report["ap"]))
        return report

    def score_tests(self, logit):
        """(candidates,) minimum over held-out folds of the sign-consistent score statistic."""
        p = torch.sigmoid(logit)
        stats = []
        for f in range(FOLDS):
            held = self.fold == f
            residual = (self.w * (self.y - p))[held].float()
            curvature = (self.w * p * (1 - p))[held].float()
            numerator = torch.zeros(self.cand.x.shape[1], device=self.device)
            denominator = torch.zeros_like(numerator)
            rows = torch.nonzero(held).flatten()
            for start in range(0, len(rows), 65536):
                block = rows[start:start + 65536]
                xb = self.cand.x[block].float()
                numerator += xb.T @ residual[start:start + 65536]
                denominator += (xb ** 2).T @ curvature[start:start + 65536]
            stats.append(numerator / denominator.clamp_min(1e-9).sqrt())
        stats = torch.stack(stats)
        sign = torch.sign(stats.sum(0))
        return (stats * sign).min(0).values

    def pick_chunk(self, scores, size=CHUNK):
        scores = scores.clone()
        scores[self.pool] = -np.inf
        order = torch.argsort(scores, descending=True)[:size * 20].tolist()
        sample = torch.randperm(self.rows.n, generator=torch.Generator().manual_seed(0))[:20000].to(self.device)
        chunk = []
        for c in order:
            if scores[c] <= 0 or len(chunk) == size:
                break
            if chunk:
                xs = self.cand.x[sample][:, chunk + [c]].float()
                corr = torch.corrcoef(xs.T)[-1, :-1].abs()
                if corr.max() > MAX_CORR:
                    continue
            chunk.append(c)
        return chunk

    def prune(self, betas):
        full = self.cv_report(self.fit_folds(self.pool, betas)[1])["ap_mean"]
        deltas = []
        for i in range(len(self.pool)):
            reduced = self.pool[:i] + self.pool[i + 1:]
            warm = {f: torch.cat([b[:i], b[i + 1:]]) for f, b in betas.items()}
            deltas.append(self.cv_report(self.fit_folds(reduced, warm)[1])["ap_mean"] - full)
        removable = [i for i in np.argsort(deltas)[::-1] if deltas[i] >= 0][: len(self.pool) // 4]
        removed = [self.pool[i] for i in removable]
        self.pool = [c for c in self.pool if c not in removed]
        return removed, full

    def record(self, step, logit, added, removed):
        report = self.cv_report(logit)
        entry = {"step": step, "size": len(self.pool), "added": [self.cand.columns[c] for c in added],
                 "removed": [self.cand.columns[c] for c in removed], **report}
        self.curve.append(entry)
        views = " ".join(f"{np.mean(report['ap_view'][v]):.3f}" for v in range(3))
        print(f"step {step:3d} pool {len(self.pool):3d} | CV AP {report['ap_mean']:.4f} (folds "
              f"{' '.join(f'{a:.3f}' for a in report['ap'])}) views ax/co/sa {views} | ring AUC "
              f"{np.mean(report['auc_ring']):.3f} vessel {np.mean(report['auc_vessel']):.3f} pleura "
              f"{np.mean(report['auc_pleura']):.3f} | -{len(removed)}", flush=True)

    def run(self, max_pool, out_path, log=print):
        betas, logit = self.fit_folds(self.pool)
        self.record(0, logit, [], [])
        step = 0
        while len(self.pool) < max_pool:
            step += 1
            chunk = self.pick_chunk(self.score_tests(logit))
            if not chunk:
                log("no candidate improves every held-out fold; stopping")
                break
            self.pool += chunk
            betas, logit = self.fit_folds(self.pool)
            removed = []
            if step % PRUNE_EVERY == 0:
                removed, _ = self.prune(betas)
                betas, logit = self.fit_folds(self.pool)
            self.record(step, logit, chunk, removed)
            self.save(out_path)
        return self.curve

    def save(self, out_path):
        out_path.write_text(json.dumps({"pool": [self.cand.columns[c] for c in self.pool], "curve": self.curve}, indent=1))


def cv_of_units(rows, units, device):
    """CV report of a fixed unit list [(layer, unit, sign)] (controls, old recipe)."""
    candidates = Candidates(rows, device, units=units)
    selection = ForwardSelection(rows, candidates, device)
    return selection.cv_report(selection.fit_folds(list(range(len(units))))[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pool", type=int, default=256)
    parser.add_argument("--layers", nargs="*", help="restrict candidates to these layers")
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--name", default="all_layers")
    args = parser.parse_args()

    device = torch.device("cuda")
    SELECTION.mkdir(parents=True, exist_ok=True)
    rows = Rows()
    start = time.time()
    candidates = Candidates(rows, device, args.layers, args.max_candidates)
    print(f"{candidates.x.shape[1]} candidates from {len({c['layer'] for c in candidates.columns})} layers "
          f"loaded in {time.time() - start:.0f}s", flush=True)
    ForwardSelection(rows, candidates, device).run(args.max_pool, SELECTION / f"{args.name}.json")


if __name__ == "__main__":
    main()


def pool_at(selection_json, step):
    """The pool after a given step, rebuilt from the curve's additions and removals (pool order = entry order)."""
    curve = json.loads(open(selection_json).read())["curve"]
    pool = []
    for entry in curve:
        if entry["step"] > step:
            break
        pool += entry["added"]
        removed = {(r["layer"], r["unit"]) for r in entry["removed"]}
        pool = [p for p in pool if (p["layer"], p["unit"]) not in removed]
    return pool
