"""Low-data readouts on frozen pooled features: how well do n images per class train a classifier?

Every readout sees only the training subset (shots images per class); val is scored once at the end.
Features are standardised with the subset's mean / std (label-free, so the CV folds below share them).

linear   ridge regression on one-hot class targets (a least-squares linear classifier), on every feature.
         Features are also divided by sqrt(n_features), so one ridge grid fits every feature count.
         With n << features it is solved in kernel form: K = X X^T over the n subset images, centred
         so the intercept is unpenalised, and A = (K + ridge I)^-1 Y; val logits = K_val A + mean(Y).
         One eigendecomposition per CV fold gives every ridge at once (an iterative logistic fit per
         fold and ridge was ~100x slower for the same job).
         The ridge is picked by 5-fold stratified CV accuracy on the subset (ties go to more ridge).
select   the fissure-style unit screen, then the same linear probe on the kept units:
         1. per feature, the ANOVA F (between-class / within-class variance) on each fold's training part
         2. score = min over folds, so a unit must separate classes in every fold (a stability screen)
         3. walk down the score, skipping units with |corr| > 0.9 to one already kept, up to k units
mlp      a trained head: features -> 512 ReLU -> dropout 0.5 -> classes, full-batch AdamW 300 steps,
         weight decay picked by the same 5-fold CV.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

FOLDS = 5
RIDGES = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
CORR_CUT = 0.9
SCREEN_CANDIDATES = 4000
MLP_DECAYS = (1e-3, 1e-1, 1.0)
MLP_STEPS = 300


def stratified_folds(y, seed=0):
    """Fold id per row: each class's rows are shuffled and dealt round-robin into FOLDS folds."""
    rng, fold = np.random.default_rng(seed), np.zeros(len(y), dtype=int)
    for c in np.unique(y):
        rows = rng.permutation(np.flatnonzero(y == c))
        fold[rows] = np.arange(len(rows)) % FOLDS
    return fold


def standardise(train, *others):
    mean, std = train.mean(0), train.std(0)
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    return [(x - mean) / std for x in (train, *others)]


def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def centred_kernels(K_fit, K_rows):
    """Centre the train kernel and the kernel of other rows against train, as if X were centred on train."""
    col_mean, total = K_fit.mean(0), K_fit.mean()
    K_fit_c = K_fit - col_mean[None, :] - col_mean[:, None] + total
    K_rows_c = K_rows - K_rows.mean(1, keepdim=True) - col_mean[None, :] + total
    return K_fit_c, K_rows_c


def ridge_logits(K_fit, K_rows, y, classes, ridges):
    """Logits of K_rows for every ridge: (ridges, rows, classes)."""
    K_fit_c, K_rows_c = centred_kernels(K_fit, K_rows)
    Y = F.one_hot(y, classes).to(K_fit.dtype)
    Y_mean = Y.mean(0)
    s, U = torch.linalg.eigh(K_fit_c)
    s = s.clamp_min(0)
    UtY = U.T @ (Y - Y_mean)
    ridges = torch.tensor(ridges, dtype=K_fit.dtype, device=K_fit.device)
    A = torch.einsum("ij,rj,jc->ric", U, 1 / (s[None] + ridges[:, None]), UtY)
    return torch.einsum("nm,rmc->rnc", K_rows_c, A) + Y_mean


def cv_pick(candidates, scores):
    """The best-scoring candidate; candidates are ordered weakest-regularised first, so ties go to the last."""
    best = max(range(len(scores)), key=lambda i: (scores[i], i))
    return candidates[best]


def linear_probe(train, y, val, y_val, classes, fold):
    train, val = standardise(train, val)
    scale = train.shape[1] ** -0.5
    train, val = (train * scale).double(), (val * scale).double()
    K, K_val = train @ train.T, val @ train.T
    hits = torch.zeros(len(RIDGES), device=K.device)
    for f in range(FOLDS):
        fit, held = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        logits = ridge_logits(K[fit][:, fit], K[held][:, fit], y[fit], classes, RIDGES)
        hits += (logits.argmax(2) == y[held]).sum(1)
    cv_scores = (hits / len(y)).tolist()
    best = cv_pick(range(len(RIDGES)), cv_scores)
    logits = ridge_logits(K, K_val, y, classes, RIDGES[best:best + 1])[0]
    return {"val_acc": accuracy(logits, y_val), "cv_acc": cv_scores[best], "ridge": RIDGES[best]}


def anova_f(x, y, classes):
    onehot = F.one_hot(y, classes).to(x.dtype)
    counts = onehot.sum(0)
    means = (onehot.T @ x) / counts[:, None]
    between = (counts[:, None] * (means - x.mean(0)) ** 2).sum(0) / (classes - 1)
    within = ((x - means[y]) ** 2).sum(0) / (len(y) - classes)
    return torch.nan_to_num(between / within.clamp_min(1e-12), nan=0.0)


def fold_min_f(x, y, classes, fold):
    return torch.stack([anova_f(x[fold != f], y[fold != f], classes) for f in range(FOLDS)]).min(0).values


def decorrelated_top(x, order, k):
    """Walk down `order`, keeping units whose |corr| with every kept unit is at most CORR_CUT."""
    cand = order[:SCREEN_CANDIDATES]
    z = standardise(x[:, cand])[0]
    corr = (z.T @ z / len(z)).abs()
    blocked = torch.zeros(len(cand), dtype=torch.bool, device=x.device)
    kept = []
    for i in range(len(cand)):
        if blocked[i]:
            continue
        kept.append(i)
        if len(kept) == k:
            break
        blocked |= corr[i] > CORR_CUT
    return cand[torch.tensor(kept, device=x.device)]


def selected_units(train, y, classes, fold, k):
    score = fold_min_f(train, y, classes, fold)
    return decorrelated_top(train, torch.argsort(score, descending=True), k)


def train_mlp(x, y, classes, decay, seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, classes)).to(x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=decay)
    for _ in range(MLP_STEPS):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def mlp_logits(model, x):
    return model(x)


def mlp_probe(train, y, val, y_val, classes, fold):
    train, val = standardise(train, val)

    def cv_accuracy(decay):
        hits = 0
        for f in range(FOLDS):
            fit, held = torch.from_numpy(fold != f), torch.from_numpy(fold == f)
            model = train_mlp(train[fit], y[fit], classes, decay)
            hits += (mlp_logits(model, train[held]).argmax(1) == y[held]).sum().item()
        return hits / len(y)
    decays = sorted(MLP_DECAYS)
    cv_scores = [cv_accuracy(decay) for decay in decays]
    decay = cv_pick(decays, cv_scores)
    model = train_mlp(train, y, classes, decay)
    return {"val_acc": accuracy(mlp_logits(model, val), y_val), "cv_acc": max(cv_scores), "decay": decay}
