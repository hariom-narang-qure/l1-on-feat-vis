"""A small segmentation head on DINOv2 fissure units, with a CT skip connection.

The units (fissure-neurons.md) live on DINOv2's 74 x 74 token grid (~6.9 px,
~5 mm per token): they say *where* a fissure is, not which pixel. The head
adds the full-resolution CT through a skip connection to place the line:

  units (K, 74, 74) --1x1 conv--> 16 ch --bilinear x6.9--> (16, 512, 512) --+
                                                                            concat -> 3x3 conv x3 -> 1x1 -> logit
  CT (1, 512, 512), lung window --3x3 conv x2 (16 ch)------------------------+

The CT branch is shallow on purpose (5 px receptive field before the merge,
11 px after): enough to find a bright line, not enough to learn by itself
where in the lung fissures run. Variants (same architecture, inputs swapped):
  - units+ct: the 4 fissure units (res8 449, 355, 442 + mlp8 2235) and CT
  - units:    the 4 units, CT input zeroed
  - ct:       CT only, unit input zeroed
  - random+ct: 4 random res8 units (not in the top 30) and CT
  - res8+ct:  all 768 res8 dimensions and CT (upper bound)
  - pooledK+ct / pooledK: the first K units of the cross-model sparse probe
    (pooled_selection.py, maps from multi_cache.py), with / without CT

Data: axial slices, 30 fissure-rich + 10 other lung slices per scan, from the
train scans; scored on all sampled slices of the test scans (probes.split_scans).
Labels: the agent sheet dilated to 3 px for training. Loss: BCE (positive
weight 10) + soft Dice, both inside the lung only.
Metrics on test, inside the lung:
  - pixel AP against the dilated sheet
  - tolerant F1 at the threshold that is best on the train slices: a predicted
    pixel counts if it is within 2 px of the sheet, a sheet pixel is found if a
    prediction is within 2 px (the 2D analogue of surface Dice).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from sklearn.metrics import average_precision_score
from torch import nn

from l1_on_feat_vis.fissure_probe.data import LoadedScan, list_scans
from l1_on_feat_vis.fissure_probe.dino_features import GRID, DinoNetworks
from l1_on_feat_vis.fissure_probe.probes import split_scans

FISSURE_UNITS_RES8 = (449, 355, 442)
FISSURE_NEURON_MLP8 = 2235
WINDOW_HU = (-1000.0, -200.0)
FISSURE_SLICES, OTHER_SLICES = 30, 10
CROP, BATCH, STEPS = 256, 8, 2500
TOLERANCE_PX = 2
VARIANTS = ("units+ct", "units", "ct", "random+ct", "res8+ct")


# ---------- cache: per scan, the slices with CT, labels and DINOv2 features ----------

def pick_slices(scan, rng):
    sheet_counts = (scan.sheet & scan.lung).sum(axis=(0, 1))
    lung_slices = np.flatnonzero(scan.lung.sum(axis=(0, 1)) > 3000)
    rich = np.flatnonzero(sheet_counts >= 200)
    picked = list(rng.choice(rich, min(FISSURE_SLICES, len(rich)), replace=False))
    rest = np.setdiff1d(lung_slices, picked)
    picked += list(rng.choice(rest, min(OTHER_SLICES, len(rest)), replace=False))
    return sorted(int(z) for z in picked)


def build_cache(records, out_dir, device, seed=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    networks = DinoNetworks(device)
    networks.models = {"pretrained": networks.models["pretrained"]}
    rng = np.random.default_rng(seed)
    for record in records:
        path = out_dir / f"{record.short}.npz"
        if path.exists():
            continue
        scan = LoadedScan(record)
        slices = pick_slices(scan, rng)
        res8, neuron = [], []
        for z in slices:
            x, _ = networks.slice_input(scan.hu[:, :, z])
            blocks = networks.features(x)["pretrained"]
            res8.append(blocks["dino_res8"].T.reshape(-1, GRID, GRID).half().cpu().numpy())
            neuron.append(blocks["dino_mlp8"][:, FISSURE_NEURON_MLP8].reshape(GRID, GRID).half().cpu().numpy())
        np.savez(path, z=np.array(slices), hu=scan.hu[:, :, slices].transpose(2, 0, 1).astype(np.int16),
                 sheet=scan.sheet[:, :, slices].transpose(2, 0, 1), lung=scan.lung[:, :, slices].transpose(2, 0, 1),
                 res8=np.stack(res8), neuron2235=np.stack(neuron))
        print(f"cache: {record.short} {len(slices)} slices", flush=True)


class SliceSet:
    """All cached slices of some scans, with per-variant token inputs."""

    def __init__(self, cache_dir, scans, variant, random_units, multi_cache=None):
        parts = [np.load(cache_dir / f"{scan}.npz") for scan in scans]
        self.scan_of = np.concatenate([[scan] * len(p["z"]) for scan, p in zip(scans, parts)])
        self.hu = np.concatenate([p["hu"] for p in parts])
        self.sheet = np.concatenate([p["sheet"] for p in parts])
        self.lung = np.concatenate([p["lung"] for p in parts])
        res8 = np.concatenate([p["res8"] for p in parts])
        neuron = np.concatenate([p["neuron2235"] for p in parts])[:, None]
        if variant in ("units+ct", "units"):
            self.tokens = np.concatenate([res8[:, list(FISSURE_UNITS_RES8)], neuron], axis=1)
        elif variant == "random+ct":
            self.tokens = res8[:, list(random_units)]
        elif variant == "res8+ct":
            self.tokens = res8
        elif variant.startswith("pooled"):
            k = int(variant.removeprefix("pooled").removesuffix("+ct"))
            self.tokens = np.concatenate([np.load(multi_cache / f"{scan}.npz")["units"][:, :k] for scan in scans])
        else:
            self.tokens = np.zeros((len(res8), 1, GRID, GRID), np.float16)
        self.use_ct = variant.endswith("+ct") or variant == "ct"
        self.target = np.stack([ndimage.binary_dilation(s, iterations=1) for s in self.sheet])

    def __len__(self):
        return len(self.hu)


def ct_input(hu):
    return (np.clip(hu, *WINDOW_HU) - WINDOW_HU[0]) / (WINDOW_HU[1] - WINDOW_HU[0])


# ---------- model ----------

class FissureHead(nn.Module):
    def __init__(self, token_channels, width=16):
        super().__init__()
        self.tokens = nn.Conv2d(token_channels, width, kernel_size=1)
        self.ct = nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, 1, kernel_size=1),
        )

    def forward(self, tokens, ct, out_size):
        upsampled = F.interpolate(self.tokens(tokens), size=out_size, mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([upsampled, self.ct(ct)], dim=1))[:, 0]


# ---------- training ----------

class HeadTrainer:
    def __init__(self, train, device, seed=0):
        self.train, self.device = train, device
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        tokens = train.tokens.astype(np.float32)
        self.token_mean = tokens.mean(axis=(0, 2, 3), keepdims=True)[0]
        self.token_std = tokens.std(axis=(0, 2, 3), keepdims=True)[0] + 1e-6
        self.model = FissureHead(train.tokens.shape[1]).to(device)

    def inputs(self, slice_set, index):
        tokens = (slice_set.tokens[index].astype(np.float32) - self.token_mean) / self.token_std
        ct = ct_input(slice_set.hu[index]) if slice_set.use_ct else np.zeros(slice_set.hu[index].shape, np.float32)
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(self.device)
        return to(tokens), to(ct)[:, None]

    def crop_corner(self, index):
        """Half the crops are centred on a sheet pixel, half on a lung pixel."""
        mask = self.train.sheet[index] if self.rng.random() < 0.5 and self.train.sheet[index].any() else self.train.lung[index]
        points = np.argwhere(mask)
        cx, cy = points[self.rng.integers(len(points))]
        size = self.train.hu.shape[1]
        return int(np.clip(cx - CROP // 2, 0, size - CROP)), int(np.clip(cy - CROP // 2, 0, size - CROP))

    def batch(self):
        indices = self.rng.integers(len(self.train), size=BATCH)
        tokens, ct = self.inputs(self.train, indices)
        corners = [self.crop_corner(i) for i in indices]
        crop = lambda volume: torch.stack([volume[b, ..., x:x + CROP, y:y + CROP] for b, (x, y) in enumerate(corners)])
        target = torch.from_numpy(self.train.target[indices]).float().to(self.device)
        # The loss covers the lung plus a 3 px margin, so sheets on the lung edge count.
        lung = np.stack([ndimage.binary_dilation(self.train.lung[i], iterations=3) for i in indices])
        lung = torch.from_numpy(lung).float().to(self.device)
        return tokens, corners, crop(ct), crop(target), crop(lung)

    def loss(self, logits, target, lung):
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(10.0, device=self.device), reduction="none")
        bce = (bce * lung).sum() / lung.sum()
        probability = torch.sigmoid(logits) * lung
        dice = 1 - (2 * (probability * target).sum() + 1) / (probability.sum() + (target * lung).sum() + 1)
        return bce + dice

    def crop_logits(self, tokens, corners, ct_crops):
        """Run the head on whole-slice tokens, then crop: the token branch needs the slice's full grid."""
        size = self.train.hu.shape[1:]
        upsampled = F.interpolate(self.model.tokens(tokens), size=size, mode="bilinear", align_corners=False)
        token_crops = torch.stack([upsampled[b, :, x:x + CROP, y:y + CROP] for b, (x, y) in enumerate(corners)])
        return self.model.fuse(torch.cat([token_crops, self.model.ct(ct_crops)], dim=1))[:, 0]

    def fit(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        for step in range(1, STEPS + 1):
            tokens, corners, ct, target, lung = self.batch()
            loss = self.loss(self.crop_logits(tokens, corners, ct), target, lung)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 500 == 0:
                print(f"  step {step} loss {loss.item():.3f}", flush=True)

    def predict(self, slice_set, index):
        tokens, ct = self.inputs(slice_set, np.array([index]))
        with torch.no_grad():
            return torch.sigmoid(self.model(tokens, ct, slice_set.hu.shape[1:]))[0].cpu().numpy()


# ---------- evaluation ----------

def tolerant_counts(prediction, sheet, lung):
    """(predicted pixels near the sheet, predicted pixels, sheet pixels found, sheet pixels)."""
    near_sheet = ndimage.binary_dilation(sheet, iterations=TOLERANCE_PX)
    near_prediction = ndimage.binary_dilation(prediction, iterations=TOLERANCE_PX)
    prediction, sheet = prediction & lung, sheet & lung
    return np.array([(prediction & near_sheet).sum(), prediction.sum(), (sheet & near_prediction).sum(), sheet.sum()])


def f1_from_counts(counts):
    precision = counts[0] / max(counts[1], 1)
    recall = counts[2] / max(counts[3], 1)
    return 2 * precision * recall / max(precision + recall, 1e-9), precision, recall


def best_threshold(trainer, slice_set, thresholds=np.linspace(0.1, 0.9, 17), limit=60):
    indices = np.random.default_rng(1).choice(len(slice_set), min(limit, len(slice_set)), replace=False)
    predictions = [trainer.predict(slice_set, i) for i in indices]
    scores = []
    for threshold in thresholds:
        counts = sum(tolerant_counts(p > threshold, slice_set.sheet[i], slice_set.lung[i]) for p, i in zip(predictions, indices))
        scores.append(f1_from_counts(counts)[0])
    return float(thresholds[int(np.argmax(scores))])


def evaluate(trainer, test, threshold):
    counts, scores, labels, per_scan = np.zeros(4), [], [], {}
    for i in range(len(test)):
        p = trainer.predict(test, i)
        lung = test.lung[i]
        scores.append(p[lung])
        labels.append(test.target[i][lung])
        c = tolerant_counts(p > threshold, test.sheet[i], lung)
        counts += c
        per_scan[test.scan_of[i]] = per_scan.get(test.scan_of[i], 0) + c
    f1, precision, recall = f1_from_counts(counts)
    return {"pixel_ap": float(average_precision_score(np.concatenate(labels), np.concatenate(scores))),
            "pixel_prior": float(np.concatenate(labels).mean()),
            "tolerant_f1": float(f1), "tolerant_precision": float(precision), "tolerant_recall": float(recall),
            "threshold": threshold,
            "per_scan_f1": {s: float(f1_from_counts(c)[0]) for s, c in per_scan.items()}}


def figure(runs, out_path, picks=3, tile=320):
    """Rows: test slices. Columns: CT, sheet, then each variant's probability.
    `runs` maps variant -> (trainer, that variant's test SliceSet); slices line up across variants."""
    test = next(iter(runs.values()))[1]
    rng = np.random.default_rng(2)
    rich = [i for i in range(len(test)) if test.sheet[i].sum() > 400]
    rows = []
    for i in rng.choice(rich, min(picks, len(rich)), replace=False):
        ct = (ct_input(test.hu[i]) * 255).astype(np.uint8).T
        sheet = np.stack([ct] * 3, -1).astype(np.float32)
        sheet[test.sheet[i].T] = [255, 40, 40]
        panels = [Image.fromarray(ct).convert("RGB"), Image.fromarray(sheet.astype(np.uint8))]
        for trainer, variant_test in runs.values():
            prob = (trainer.predict(variant_test, i) * test.lung[i]).T
            panels.append(Image.fromarray((prob * 255).astype(np.uint8)).convert("RGB"))
        rows.append(panels)
    out = Image.new("RGB", (len(rows[0]) * tile, len(rows) * tile), "white")
    for r, panels in enumerate(rows):
        for c, panel in enumerate(panels):
            out.paste(panel.resize((tile, tile)), (c * tile, r * tile))
    out.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("outputs/fissure_probe/seg_cache"))
    parser.add_argument("--out", type=Path, default=Path("outputs/fissure_probe/seg_head"))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--multi-cache", type=Path, default=Path("outputs/fissure_probe/multi_cache"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_scans, test_scans, _ = split_scans()
    records = [r for r in list_scans() if r.short in set(train_scans) | set(test_scans)]
    build_cache(records, args.cache, device)

    top30 = {r["unit"] for r in json.load(open("outputs/fissure_probe/dino_units/pretrained_dino_res8.json"))["units"]}
    random_units = np.random.default_rng(0).choice(sorted(set(range(768)) - top30), 4, replace=False).tolist()
    args.out.mkdir(parents=True, exist_ok=True)
    results, runs = {"random_units": random_units}, {}
    for variant in args.variants:
        print(f"variant {variant}", flush=True)
        train = SliceSet(args.cache, train_scans, variant, random_units, args.multi_cache)
        test = SliceSet(args.cache, test_scans, variant, random_units, args.multi_cache)
        trainer = HeadTrainer(train, device)
        trainer.fit()
        threshold = best_threshold(trainer, train)
        results[variant] = evaluate(trainer, test, threshold)
        runs[variant] = (trainer, test)
        r = results[variant]
        print(f"{variant:10s} pixel AP {r['pixel_ap']:.3f} (prior {r['pixel_prior']:.3f}) | tolerant F1 {r['tolerant_f1']:.3f} "
              f"P {r['tolerant_precision']:.3f} R {r['tolerant_recall']:.3f} @ {threshold:.2f}", flush=True)
        (args.out / "results.json").write_text(json.dumps(results, indent=1))
    figure(runs, args.out / "test_slices.png")


if __name__ == "__main__":
    main()
