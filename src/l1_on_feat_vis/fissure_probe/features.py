"""Sample per-channel activations of Genesis (pretrained and random init) on CT patches.

Per scan, per patch (128 x 128 x 64 voxels, native spacing):
  1. Forward the patch through both networks, keep every probed block's output.
  2. Bring the labels to each block's resolution with a stride-s max-pool
     (a cell is "fissure" if any voxel in it is), and the lung mask with an
     average-pool (a cell is lung if most of it is).
  3. Drop cells within 16 voxels of the patch border (zero-padding effects).
  4. Sample cells:
     - uniform: random lung cells. Fissure appears at its natural frequency,
       so average precision on these is honest. Used for evaluation.
     - extra positives: random fissure cells, only to give the probe enough
       positives to train on.
  5. Store each sampled cell's channel vector plus its labels and baselines:
     intensity (mean of the cell) and Hessian sheetness (max over the cell).

Hessian sheetness (bright plate detector, the classical fissure enhancer):
eigenvalues |l1| <= |l2| <= |l3| of the scale-normalised Hessian at sigma 1 and
2 voxels; a bright sheet has l3 << 0 and l1, l2 ~ 0, scored
-l3 * exp(-(l2 / l3)^2 / (2 * 0.5^2)) when l3 < 0, max over the two scales
(after Descoteaux et al. 2005, simplified).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from l1_on_feat_vis.fissure_probe.data import LoadedScan, PatchSampler, hu_to_genesis_input, list_scans
from l1_on_feat_vis.fissure_probe.genesis import BLOCK_STRIDE, PROBED_BLOCKS, load_genesis

BORDER_VOXELS = 16
UNIFORM_SAMPLES = {1: 4000, 2: 3000, 4: 1500, 8: 400}
EXTRA_POSITIVES = {1: 1000, 2: 800, 4: 400, 8: 100}
LABELS = ("fissure", "lobe_boundary", "pleura", "ring")


def hessian_eigenvalues(volume, sigma):
    orders = {(0, 0): (2, 0, 0), (1, 1): (0, 2, 0), (2, 2): (0, 0, 2), (0, 1): (1, 1, 0), (0, 2): (1, 0, 1), (1, 2): (0, 1, 1)}
    hessian = np.zeros(volume.shape + (3, 3), dtype=np.float32)
    for (i, j), order in orders.items():
        second = ndimage.gaussian_filter(volume, sigma, order=order) * sigma**2
        hessian[..., i, j] = hessian[..., j, i] = second
    eigenvalues = np.linalg.eigvalsh(hessian)
    by_magnitude = np.argsort(np.abs(eigenvalues), axis=-1)
    return np.take_along_axis(eigenvalues, by_magnitude, axis=-1)


def sheetness(volume, sigmas=(1.0, 2.0)):
    scores = []
    for sigma in sigmas:
        eig = hessian_eigenvalues(volume, sigma)
        l2, l3 = eig[..., 1], eig[..., 2]
        flatness = np.exp(-((l2 / np.where(l3 == 0, 1e-6, l3)) ** 2) / (2 * 0.5**2))
        scores.append(np.where(l3 < 0, -l3 * flatness, 0.0))
    return np.max(scores, axis=0).astype(np.float32)


def pool(volume, stride, mode):
    tensor = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))[None, None]
    if stride == 1:
        return tensor[0, 0].numpy()
    pooled = F.max_pool3d(tensor, stride) if mode == "max" else F.avg_pool3d(tensor, stride)
    return pooled[0, 0].numpy()


def interior(shape, stride):
    margin = -(-BORDER_VOXELS // stride)
    keep = np.zeros(shape, dtype=bool)
    keep[margin:-margin, margin:-margin, margin:-margin] = True
    return keep


class FeatureSampler:
    def __init__(self, device, patches_per_scan=12, seed=0):
        self.device = device
        self.models = {"pretrained": load_genesis().to(device), "random": load_genesis(pretrained=False).to(device)}
        self.sampler = PatchSampler(seed=seed)
        self.patches_per_scan = patches_per_scan
        self.rng = np.random.default_rng(seed)

    def network_features(self, patch):
        x = torch.from_numpy(patch.astype(np.float32))[None, None].to(self.device)
        with torch.no_grad():
            return {name: model.forward_features(x) for name, model in self.models.items()}

    def pick_cells(self, lung, fissure, stride):
        valid = interior(lung.shape, stride) & lung
        lung_cells = np.flatnonzero(valid)
        fissure_cells = np.flatnonzero(valid & fissure)
        uniform = self.rng.choice(lung_cells, min(UNIFORM_SAMPLES[stride], len(lung_cells)), replace=False)
        extra = self.rng.choice(fissure_cells, min(EXTRA_POSITIVES[stride], len(fissure_cells)), replace=False)
        return np.concatenate([uniform, extra]), np.concatenate([np.ones(len(uniform), bool), np.zeros(len(extra), bool)])

    def sample_patch(self, scan, corner, patch_index, out):
        crop = lambda volume: self.sampler.crop(volume, corner)
        patch = hu_to_genesis_input(crop(scan.hu))
        maps = {"intensity": patch, "sheetness": sheetness(patch), "lung": crop(scan.lung)}
        maps.update({label: crop(getattr(scan, label)) for label in LABELS})
        features = self.network_features(patch)
        for block in PROBED_BLOCKS:
            stride = BLOCK_STRIDE[block]
            lung = pool(maps["lung"], stride, "avg") > 0.5
            labels = {label: pool(maps[label], stride, "max") > 0 for label in LABELS if label != "ring"}
            # A ring cell is mostly ring and holds no fissure voxel.
            labels["ring"] = (pool(maps["ring"], stride, "avg") > 0.5) & ~labels["fissure"]
            cells, uniform = self.pick_cells(lung, labels["fissure"], stride)
            rows = out.setdefault(block, {key: [] for key in ("uniform", "patch", "intensity", "sheetness", *LABELS, "pretrained", "random")})
            rows["uniform"].append(uniform)
            rows["patch"].append(np.full(len(cells), patch_index, np.int16))
            rows["intensity"].append(pool(maps["intensity"], stride, "avg").ravel()[cells])
            rows["sheetness"].append(pool(maps["sheetness"], stride, "max").ravel()[cells])
            for label in LABELS:
                rows[label].append(labels[label].ravel()[cells])
            for name in self.models:
                activation = features[name][block][0].flatten(1)[:, torch.from_numpy(cells).to(self.device)]
                rows[name].append(activation.T.half().cpu().numpy())

    def sample_scan(self, record, out_dir):
        scan = LoadedScan(record)
        out = {}
        for index, corner in enumerate(self.sampler.corners(scan, self.patches_per_scan)):
            self.sample_patch(scan, corner, index, out)
        arrays = {f"{block}/{key}": np.concatenate(values) for block, rows in out.items() for key, values in rows.items()}
        np.savez(out_dir / f"{record.short}.npz", **arrays)
        return scan.sheet_voxels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/fissure_probe/features"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patches-per-scan", type=int, default=12)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    sampler = FeatureSampler(torch.device(args.device), args.patches_per_scan)
    for record in list_scans():
        if (args.out / f"{record.short}.npz").exists():
            continue
        start = time.time()
        sheet_voxels = sampler.sample_scan(record, args.out)
        print(f"{record.short} {record.cohort} sheet voxels {sheet_voxels} done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
