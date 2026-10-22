"""Sample units of pretrained 3D nnU-Nets at the same slices and tokens as slice_features.py.

Any nnU-Net v2 results folder works (TotalSegmentator, VoxTell's image encoder, ...).
Per scan:
  1. Crop the lung bounding box (+15 mm) from the native CT.
  2. Preprocess like nnU-Net, from the model's own plans:
     - axes (x, y, z) -> (z, y, x), the order nnU-Net reads NIfTI in, then
       its transpose_forward;
     - resample to the plans' spacing (trilinear);
     - CT normalisation: clip to the dataset's [0.5, 99.5] foreground
       percentiles, then z-score with its foreground mean / std;
     - pad to a multiple of the network's total stride.
  3. One forward pass over the whole crop (fully convolutional, no grad),
     hooking every encoder and decoder stage.
     Sanity check: the model's own lobe prediction from this input is
     compared with the scan's lobes.nii.gz (Dice) before trusting the features.
  4. For the slices and tokens slice_features.py sampled, read each stage's
     features at the token centres (trilinear grid_sample in the stage's own
     grid). The slice/token choice is replayed with the same random generator
     and call order, so rows line up with the 2D models row for row.
Each network has a random-init twin (same plans, fresh weights).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from l1_on_feat_vis.fissure_probe.data import LoadedScan, list_scans
from l1_on_feat_vis.fissure_probe.dino_features import EXTRA_POSITIVES, GRID, LABELS, UNIFORM_TOKENS, to_grid

TS_LOBES = "/home/hariom.narang/.totalsegmentator/nnunet/results/Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
NETWORKS = {"ts_organs": TS_LOBES}
TS_LOBE_LABELS = {10: 10, 11: 11, 12: 12, 13: 13, 14: 14}  # TotalSegmentator v2 "total" ids == lobes.nii.gz ids
CROP_MARGIN_MM = 15.0


class SelectionReplay:
    """The slice and token picks of slice_features.SliceSampler, without the networks."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def pick_slices(self, scan):
        per_slice_sheet = (scan.fissure & scan.lung).sum(axis=(0, 1))
        lung_slices = np.flatnonzero(scan.lung.sum(axis=(0, 1)) > 2000)
        fissure_slices = np.flatnonzero(per_slice_sheet >= 200)
        picked = list(self.rng.choice(fissure_slices, min(6, len(fissure_slices)), replace=False))
        return picked + list(self.rng.choice(lung_slices, 4, replace=False))

    def pick_tokens(self, lung, fissure):
        lung_tokens, fissure_tokens = np.flatnonzero(lung), np.flatnonzero(lung & fissure)
        uniform = self.rng.choice(lung_tokens, min(UNIFORM_TOKENS, len(lung_tokens)), replace=False)
        extra = self.rng.choice(fissure_tokens, min(EXTRA_POSITIVES, len(fissure_tokens)), replace=False)
        return np.concatenate([uniform, extra]), np.concatenate([np.ones(len(uniform), bool), np.zeros(len(extra), bool)])


class NnUNet:
    def __init__(self, name, folder, device, pretrained=True, stage_hooks=True):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        predictor = nnUNetPredictor(device=device, perform_everything_on_device=True)
        predictor.initialize_from_trained_model_folder(folder, use_folds=(0,), checkpoint_name="checkpoint_final.pth")
        self.name, self.device = name, device
        self.network = predictor.network.to(device).eval()
        if pretrained:
            self.network.load_state_dict(predictor.list_of_parameters[0])
        else:
            torch.manual_seed(0)
            for module in self.network.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        self.plans = predictor.plans_manager
        self.config = predictor.configuration_manager
        intensity = self.plans.foreground_intensity_properties_per_channel["0"]
        self.clip = (intensity["percentile_00_5"], intensity["percentile_99_5"])
        self.mean, self.std = intensity["mean"], intensity["std"]
        self.spacing = np.array(self.config.spacing)  # in transposed (z, y, x) order
        self.stride = int(np.prod([s[0] for s in self.config.pool_op_kernel_sizes]))
        self.stages = {}
        if not stage_hooks:  # callers that tap the network themselves (store.py) skip the full-map copies
            return
        for i, stage in enumerate(self.network.encoder.stages):
            stage.register_forward_hook(lambda m, inp, out, i=i: self.stages.__setitem__(f"enc{i}", out))
        for i, stage in enumerate(self.network.decoder.stages):
            stage.register_forward_hook(lambda m, inp, out, i=i: self.stages.__setitem__(f"dec{i}", out))

    def preprocess(self, hu, native_spacing_xyz):
        """(x, y, z) HU crop -> normalised (1, 1, Z', Y', X') tensor and the zoom factors used."""
        volume = hu.transpose(2, 1, 0)  # (z, y, x)
        spacing = np.array(native_spacing_xyz)[::-1]
        transpose = self.plans.transpose_forward
        volume, spacing = volume.transpose(transpose), spacing[list(transpose)]
        zoom = spacing / self.spacing
        volume = ndimage.zoom(volume, zoom, order=1)
        volume = (np.clip(volume, *self.clip) - self.mean) / self.std
        pad = [(0, (-n) % self.stride) for n in volume.shape]
        volume = np.pad(volume, pad, constant_values=(self.clip[0] - self.mean) / self.std)
        return torch.from_numpy(volume.astype(np.float32))[None, None].to(self.device), zoom

    def forward(self, x):
        self.stages.clear()
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits = self.network(x)
        logits = logits[0] if isinstance(logits, (list, tuple)) else logits
        return logits, dict(self.stages)


def lung_crop(scan, spacing_xyz):
    xs, ys, zs = np.nonzero(scan.lung)
    margin = np.ceil(CROP_MARGIN_MM / np.array(spacing_xyz)).astype(int)
    lo = np.maximum(np.array([xs.min(), ys.min(), zs.min()]) - margin, 0)
    hi = np.minimum(np.array([xs.max(), ys.max(), zs.max()]) + margin + 1, scan.hu.shape)
    return lo, hi


def native_to_network(points_xyz, lo, zoom, transpose):
    """Native voxel (x, y, z) -> transposed network-grid coordinates (voxel units)."""
    local = (points_xyz - lo)[:, ::-1]  # (z, y, x)
    return local[:, list(transpose)] * zoom


def sample_stage(feature, coords, stage_stride, padded_shape):
    """Trilinear sample of (1, C, D, H, W) at network-grid coords (N, 3) given at full resolution."""
    shape = np.array(feature.shape[2:])
    cell = coords / (np.array(padded_shape) / shape)  # full-res voxel -> stage voxel
    normalised = 2 * (cell + 0.5) / shape - 1
    grid = torch.from_numpy(normalised[:, ::-1].copy().astype(np.float32)).to(feature.device)[None, None, None]
    sampled = F.grid_sample(feature.float(), grid, mode="bilinear", align_corners=False)
    return sampled[0, :, 0, 0].T  # (N, C)


def lobe_dice(logits, scan, lo, hi, zoom, transpose):
    """Dice of the network's lobe labels vs lobes.nii.gz inside the crop (sanity check)."""
    labels = logits.argmax(1)[0].cpu().numpy()
    inverse = np.argsort(transpose)
    crop_shape = (hi - lo)[::-1][list(transpose)]
    labels = ndimage.zoom(labels[: int(np.ceil(crop_shape[0] * zoom[0])), : int(np.ceil(crop_shape[1] * zoom[1])), : int(np.ceil(crop_shape[2] * zoom[2]))],
                          1 / zoom, order=0)[: crop_shape[0], : crop_shape[1], : crop_shape[2]]
    labels = labels.transpose(inverse).transpose(2, 1, 0)
    truth = scan.lobes[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]][: labels.shape[0], : labels.shape[1], : labels.shape[2]]
    labels = labels[: truth.shape[0], : truth.shape[1], : truth.shape[2]]
    dice = [2 * ((labels == k) & (truth == v)).sum() / max((labels == k).sum() + (truth == v).sum(), 1) for k, v in TS_LOBE_LABELS.items()]
    return float(np.mean(dice))


class NnUNetSampler:
    def __init__(self, names, device):
        self.device = device
        self.networks = {(name, kind): NnUNet(name, NETWORKS[name], device, pretrained=kind == "pretrained")
                         for name in names for kind in ("pretrained", "random")}

    def sample_scan(self, record, replay, out_dir, check_lobes):
        scan = LoadedScan(record)
        import nibabel as nib
        spacing = nib.load(record.folder / "scan.nii.gz").header.get_zooms()[:3]
        lo, hi = lung_crop(scan, spacing)
        crop = scan.hu[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        stage_features, report = {}, {}
        tick = time.time()
        log = lambda what: print(f"  {record.short} {what} at {time.time() - tick:.0f}s", flush=True)
        log(f"loaded, crop {crop.shape}")
        for (name, kind), network in self.networks.items():
            x, zoom = network.preprocess(crop, spacing)
            log(f"{name}/{kind} preprocessed to {tuple(x.shape[2:])}")
            logits, stages = network.forward(x)
            torch.cuda.synchronize()
            log(f"{name}/{kind} forward")
            if kind == "pretrained" and check_lobes:
                report[name] = lobe_dice(logits, scan, lo, hi, zoom, network.plans.transpose_forward)
                log(f"{name} lobe dice {report[name]:.3f}")
            stage_features[(name, kind)] = (stages, zoom, network.plans.transpose_forward, x.shape[2:])
        out = {}
        for index, z in enumerate(replay.pick_slices(scan)):
            self.sample_slice(scan, int(z), index, lo, stage_features, replay, out)
        arrays = {f"{block}/{key}": np.concatenate(values) for block, rows in out.items() for key, values in rows.items()}
        np.savez(out_dir / f"{record.short}.npz", **arrays)
        return report

    def sample_slice(self, scan, z, slice_index, lo, stage_features, replay, out):
        lung = to_grid(scan.lung[:, :, z], "avg") > 0.5
        labels = {label: to_grid(getattr(scan, label)[:, :, z], "max") > 0 for label in LABELS if label != "ring"}
        labels["ring"] = (to_grid(scan.ring[:, :, z], "avg") > 0.5) & ~labels["fissure"]
        tokens, uniform = replay.pick_tokens(lung.ravel(), labels["fissure"].ravel())
        if len(tokens) == 0:
            return
        size = scan.hu.shape[0]
        i, j = np.unravel_index(tokens, (GRID, GRID))
        points = np.stack([(i + 0.5) * size / GRID, (j + 0.5) * size / GRID, np.full(len(i), z)], axis=1)
        for (name, kind), (stages, zoom, transpose, padded) in stage_features.items():
            coords = native_to_network(points, lo, zoom, transpose)
            for stage, feature in stages.items():
                block = f"{name}/{stage}"
                rows = out.setdefault(block, {key: [] for key in ("uniform", "patch", *LABELS, "pretrained", "random", "intensity", "sheetness")})
                rows[kind].append(sample_stage(feature, coords, None, padded).half().cpu().numpy())
                if kind == "pretrained":
                    rows["uniform"].append(uniform)
                    rows["patch"].append(np.full(len(tokens), slice_index, np.int16))
                    rows["intensity"].append(np.zeros(len(tokens), np.float32))
                    rows["sheetness"].append(np.zeros(len(tokens), np.float32))
                    for label in LABELS:
                        rows[label].append(labels[label].ravel()[tokens])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--networks", nargs="+", default=list(NETWORKS))
    parser.add_argument("--scans", nargs="*", help="short ids; default all (replay needs every earlier scan's draws)")
    parser.add_argument("--out", type=Path, default=Path("outputs/fissure_probe/nnunet_features"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sampler = NnUNetSampler(args.networks, torch.device(args.device))
    replay = SelectionReplay()
    for record in list_scans():
        if args.scans and record.short not in args.scans:
            # Advance the replay exactly as slice_features did for this scan, without the networks.
            scan = LoadedScan(record)
            for z in replay.pick_slices(scan):
                lung = to_grid(scan.lung[:, :, int(z)], "avg") > 0.5
                fissure = to_grid(scan.fissure[:, :, int(z)], "max") > 0
                if lung.any():
                    replay.pick_tokens(lung.ravel(), fissure.ravel())
            continue
        start = time.time()
        report = sampler.sample_scan(record, replay, args.out, check_lobes=True)
        print(f"{record.short} {record.cohort} lobe dice {report} done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
