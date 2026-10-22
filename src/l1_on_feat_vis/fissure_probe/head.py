"""Phase 4: the CT-skip segmentation head on selected units, trained on all three views, scored per view.

Architecture: seg_head.FissureHead (units 1x1 conv -> 16 ch -> bilinear to the
512 px canvas, concatenated with a shallow 2-conv CT branch, then 3 convs):
the units say roughly where a fissure is (6.9 px tokens), the CT places the line.
Data: every line slice of the given scans (views.py canvases), unit maps from
unit_maps.py. Training as in seg_head.py: 256 px crops (half centred on a line
pixel, half on a lung pixel), target = positive band (line + 1 px), BCE
(positive weight 10) + soft Dice, both inside the lung.
Threshold and minimum component size: the pair with the best tolerant Dice on
120 training slices (dropping small components removes short false fragments).
Metrics: seg_metrics.Scoreboard, per view and overall.
Variants: "units+ct" (default), "units" (CT zeroed), "ct" (units zeroed).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from l1_on_feat_vis.fissure_probe.probes import split_scans
from l1_on_feat_vis.fissure_probe.rows import fold_of_scans
from l1_on_feat_vis.fissure_probe.seg_head import FissureHead
from l1_on_feat_vis.fissure_probe.seg_metrics import Scoreboard
from l1_on_feat_vis.fissure_probe.store import canvas_grey
from l1_on_feat_vis.fissure_probe.unit_maps import slice_order, unit_path
from l1_on_feat_vis.fissure_probe.views import BITS, STORE, VIEWS

CROP, BATCH, STEPS, LR = 256, 8, 4000, 1e-3
THRESHOLDS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97])
MIN_SIZES = (0, 10, 20, 40, 80)  # post-processing: drop predicted components smaller than this (px)
HEADS = STORE / "heads"


class SliceData:
    """Canvases, labels and unit maps of all line slices of some scans."""

    def __init__(self, scans, units, signs):
        order = slice_order()
        wanted = set(scans)
        index = np.array([i for i, (scan, _) in enumerate(order) if scan in wanted])
        caches = {scan: np.load(STORE / "slices" / f"{scan}.npz") for scan in scans}
        self.scan = np.array([order[i][0] for i in index])
        local = [order[i][1] for i in index]
        self.hu = np.stack([caches[s]["hu"][k] for s, k in zip(self.scan, local)])
        self.labels = np.stack([caches[s]["labels"][k] for s, k in zip(self.scan, local)])
        self.view = np.array([caches[s]["view"][k] for s, k in zip(self.scan, local)])
        if units:
            maps = [np.load(unit_path(layer, unit), mmap_mode="r")[index] for layer, unit in units]
            self.tokens = np.stack(maps, 1).astype(np.float16) * np.asarray(signs, np.float16)[None, :, None, None]
        else:
            self.tokens = np.zeros((len(index), 1, 74, 74), np.float16)

    def __len__(self):
        return len(self.hu)

    def mask(self, name, index):
        return (self.labels[index] & BITS[name]) > 0


class Trainer:
    def __init__(self, train, device, variant="units+ct", seed=0, steps=STEPS):
        self.train, self.device, self.variant, self.steps = train, device, variant, steps
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        sample = train.tokens[self.rng.choice(len(train), min(200, len(train)), replace=False)].astype(np.float32)
        self.token_mean = sample.mean(axis=(0, 2, 3))[None, :, None, None]
        self.token_std = sample.std(axis=(0, 2, 3))[None, :, None, None] + 1e-6
        self.model = FissureHead(train.tokens.shape[1]).to(device)

    def inputs(self, data, index):
        tokens = (data.tokens[index].astype(np.float32) - self.token_mean) / self.token_std
        if self.variant == "ct":
            tokens = np.zeros_like(tokens)
        ct = canvas_grey(data.hu[index]).astype(np.float32)
        if self.variant == "units":
            ct = np.zeros_like(ct)
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(self.device)
        return to(tokens), to(ct)[:, None]

    def crop_corner(self, index):
        mask = self.train.mask("line", index) if self.rng.random() < 0.5 else self.train.mask("lung", index)
        points = np.argwhere(mask)
        r, c = points[self.rng.integers(len(points))]
        size = self.train.hu.shape[1]
        return int(np.clip(r - CROP // 2, 0, size - CROP)), int(np.clip(c - CROP // 2, 0, size - CROP))

    def batch(self):
        index = self.rng.integers(len(self.train), size=BATCH)
        tokens, ct = self.inputs(self.train, index)
        corners = [self.crop_corner(i) for i in index]
        crop = lambda volume: torch.stack([volume[b, ..., r:r + CROP, c:c + CROP] for b, (r, c) in enumerate(corners)])
        target = torch.from_numpy(self.train.mask("positive", index)).float().to(self.device)
        lung = torch.from_numpy(self.train.mask("lung", index)).float().to(self.device)
        return tokens, corners, crop(ct), crop(target), crop(lung)

    def loss(self, logits, target, lung):
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(10.0, device=self.device), reduction="none")
        bce = (bce * lung).sum() / lung.sum()
        probability = torch.sigmoid(logits) * lung
        dice = 1 - (2 * (probability * target).sum() + 1) / (probability.sum() + (target * lung).sum() + 1)
        return bce + dice

    def crop_logits(self, tokens, corners, ct_crops):
        """The token branch needs the whole slice's grid, so it runs on the full slice before cropping."""
        size = self.train.hu.shape[1:]
        upsampled = F.interpolate(self.model.tokens(tokens), size=size, mode="bilinear", align_corners=False)
        token_crops = torch.stack([upsampled[b, :, r:r + CROP, c:c + CROP] for b, (r, c) in enumerate(corners)])
        return self.model.fuse(torch.cat([token_crops, self.model.ct(ct_crops)], dim=1))[:, 0]

    def fit(self, log_every=1000):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.steps)
        for step in range(1, self.steps + 1):
            tokens, corners, ct, target, lung = self.batch()
            loss = self.loss(self.crop_logits(tokens, corners, ct), target, lung)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            if step % log_every == 0:
                print(f"    step {step} loss {loss.item():.3f}", flush=True)

    def predict(self, data, index):
        tokens, ct = self.inputs(data, np.array([index]))
        with torch.no_grad():
            return torch.sigmoid(self.model(tokens, ct, data.hu.shape[1:]))[0].cpu().numpy()

    def best_threshold(self, data, limit=120):
        """(threshold, min component size) with the best tolerant Dice on `limit` slices of `data`."""
        index = np.random.default_rng(1).choice(len(data), min(limit, len(data)), replace=False)
        predictions = [(self.predict(data, i), i) for i in index]
        boards = {(t, m): Scoreboard() for t in THRESHOLDS for m in MIN_SIZES}
        for p, i in predictions:
            for (t, m), board in boards.items():
                board.add(p, data.labels[i], VIEWS[data.view[i]], t, m)
        scores = {key: b.summarise()["all"]["tolerant_dice"] for key, b in boards.items()}
        t, m = max(scores, key=scores.get)
        return (float(t), int(m))

    def evaluate(self, data, threshold, keep_predictions=False):
        threshold, min_size = threshold if isinstance(threshold, tuple) else (threshold, 0)
        board, raw, per_scan, predictions = Scoreboard(), Scoreboard(), {}, []
        for i in range(len(data)):
            p = self.predict(data, i)
            view = VIEWS[data.view[i]]
            board.add(p, data.labels[i], view, threshold, min_size)
            raw.add(p, data.labels[i], view, threshold)
            per_scan.setdefault(data.scan[i], Scoreboard()).add(p, data.labels[i], view, threshold, min_size)
            if keep_predictions:
                predictions.append(p.astype(np.float16))
        result = {"threshold": threshold, "min_size": min_size, **board.summarise(), "no_postprocess": raw.summarise(),
                  "per_scan_tolerant_dice": {s: b.summarise()["all"]["tolerant_dice"] for s, b in per_scan.items()}}
        return (result, np.stack(predictions)) if keep_predictions else result


def read_units(path):
    data = json.loads(Path(path).read_text())
    entries = data["pool"] if isinstance(data, dict) else data
    return entries


def train_and_test(entries, k, variant, device, seed=0, steps=STEPS, save=None):
    train_scans, test_scans, near_empty = split_scans()
    chosen = entries[:k] if k else entries  # k = 0: the whole unit list (empty list: CT only)
    units, signs = [(e["layer"], e["unit"]) for e in chosen], [e["sign"] for e in chosen]
    train = SliceData(train_scans, units, signs)
    trainer = Trainer(train, device, variant, seed, steps)
    start = time.time()
    trainer.fit()
    threshold = trainer.best_threshold(train)
    result = {"k": len(chosen), "variant": variant, "seed": seed, "train_seconds": time.time() - start,
              "test": trainer.evaluate(SliceData(test_scans, units, signs), threshold),
              "near_empty": trainer.evaluate(SliceData(near_empty, units, signs), threshold)}
    if save:
        torch.save({"state": trainer.model.state_dict(), "units": chosen, "mean": trainer.token_mean,
                    "std": trainer.token_std, "threshold": threshold, "variant": variant}, save)
    return result


def out_of_fold(entries, k, device, seed=0, steps=STEPS):
    """Train on 8 train scans, predict the other 4, rotate: OOF probability maps of every train slice.

    Returns per fold {scans, threshold, result, predictions} and a pooled OOF summary
    (each fold's slices scored at that fold's own threshold, counts summed over folds).
    """
    train_scans, _, _ = split_scans()
    folds = fold_of_scans(train_scans)
    chosen = entries[:k] if k else entries
    units, signs = [(e["layer"], e["unit"]) for e in chosen], [e["sign"] for e in chosen]
    out, pooled = {}, Scoreboard()
    for f in range(3):
        fit_scans = [s for s in train_scans if folds[s] != f]
        held = [s for s in train_scans if folds[s] == f]
        trainer = Trainer(SliceData(fit_scans, units, signs), device, "units+ct", seed, steps)
        trainer.fit()
        threshold = trainer.best_threshold(trainer.train)
        held_data = SliceData(held, units, signs)
        result, predictions = trainer.evaluate(held_data, threshold, keep_predictions=True)
        for i, p in enumerate(predictions):
            pooled.add(p.astype(np.float32), held_data.labels[i], VIEWS[held_data.view[i]], *threshold)
        out[f] = {"scans": held, "threshold": threshold, "result": result, "predictions": predictions}
    return out, {"threshold": float("nan"), **pooled.summarise()}


def summary_line(name, result):
    a = result["all"]
    views = " ".join(f"{v[:2]} {result[v]['tolerant_dice']:.3f}" for v in VIEWS)
    return (f"{name:28s} tolDice {a['tolerant_dice']:.3f} (C {a['completeness']:.3f} / K {a['correctness']:.3f}) "
            f"clDice {a['cldice']:.3f} pxDice {a['pixel_dice']:.3f} AP {a['pixel_ap']:.3f} | {views} | "
            f"missed seg {a['missed_segment_rate']:.2f} false seg {a['false_segment_rate']:.2f} "
            f"(v {a['false_vessel']:.0f} p {a['false_pleura']:.0f} o {a['false_other']:.0f}) @ {result['threshold']:.2f} "
            f"min {result.get('min_size', 0)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=str, help="selection json (pool) or unit list; omit for CT only")
    parser.add_argument("--k", type=int, nargs="+", default=[0])
    parser.add_argument("--variants", nargs="+", default=["units+ct"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--name", required=True)
    parser.add_argument("--oof", action="store_true", help="out-of-fold CV on the train scans only (no test)")
    args = parser.parse_args()
    if args.oof:
        return run_oof(args)

    device = torch.device("cuda")
    HEADS.mkdir(parents=True, exist_ok=True)
    entries = read_units(args.units) if args.units else []
    results_path = HEADS / f"{args.name}.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    for k in args.k:
        for variant in args.variants:
            for seed in args.seeds:
                tag = f"{args.name} k={k} {variant} s{seed}"
                result = train_and_test(entries, k, variant, device, seed, args.steps,
                                        save=HEADS / f"{args.name}_k{k}_{variant}_s{seed}.pt")
                results.append(result)
                results_path.write_text(json.dumps(results, indent=1, default=float))
                print(summary_line(tag, result["test"]), flush=True)
                print(summary_line(tag + " [near-empty]", result["near_empty"]), flush=True)


def run_oof(args):
    device = torch.device("cuda")
    HEADS.mkdir(parents=True, exist_ok=True)
    entries = read_units(args.units)
    for k in args.k:
        folds, pooled = out_of_fold(entries, k, device, args.seeds[0], args.steps)
        tag = f"{args.name}_k{k}_oof"
        np.savez_compressed(HEADS / f"{tag}_predictions.npz",
                            **{f"fold{f}": part["predictions"] for f, part in folds.items()},
                            thresholds=np.array([part["threshold"] for part in folds.values()]))
        summary = {"k": k or len(entries), "pooled": pooled,
                   "folds": {f: {"scans": part["scans"], "threshold": part["threshold"], "all": part["result"]["all"]}
                             for f, part in folds.items()}}
        (HEADS / f"{tag}.json").write_text(json.dumps(summary, indent=1, default=float))
        print(summary_line(tag + " [OOF train CV]", pooled), flush=True)
        print("  per fold tolDice " + " ".join(f"{part['result']['all']['tolerant_dice']:.3f}" for part in folds.values()), flush=True)


if __name__ == "__main__":
    main()
