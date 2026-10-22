"""Evidence that chosen Genesis channels respond to fissures, on held-out scans.

Given one block and its candidate channels (from probes.py), on the test scans:

  1. Overlays: CT | sheet | each top channel | sparse probe, in the axial,
     coronal and sagittal planes through a fissure-rich patch centre.
  2. Erase (input intervention): replace the sheet's voxels with the 7^3 median
     of the CT around them (the local parenchyma, since a 3-voxel-thick sheet is
     a minority of a 7^3 cube). Measure the channel's mean response on the
     sheet's cells before and after.
     Control: erase a copy of the sheet mask shifted 12 voxels in x, kept only in
     lung away from any fissure, and measure there. A fissure detector should
     drop on real erasures and barely move on the control.
  3. Insert (input intervention): in lung patches with no fissure within 20
     voxels of the centre, add a synthetic structure through the centre with a
     Gaussian profile of amplitude equal to the measured sheet contrast:
     - plane: random orientation, sigma 0.7 voxel, a disk of radius 28 voxels
     - tube: random direction, sigma 1.0 voxel, length 56 voxels (vessel-like)
     Measure the response increase on the inserted structure's cells.
     A fissure detector should respond more to planes than to tubes.
  4. Stability: rank channels by train AP on four different train/test splits
     and report how often each candidate is in the top 10.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage

from l1_on_feat_vis.fissure_probe.data import LoadedScan, PatchSampler, hu_to_genesis_input, list_scans
from l1_on_feat_vis.fissure_probe.genesis import BLOCK_STRIDE, load_genesis
from l1_on_feat_vis.fissure_probe.probes import (BlockRows, Standardiser, fit_probe, signed_channel_ranking,
                                                 split_scans)

ERASE_SHIFT = 12
INSERT_CLEARANCE = 20
PLANE_RADIUS, PLANE_SIGMA = 28, 0.7
TUBE_HALF_LENGTH, TUBE_SIGMA = 28, 1.0


def to_tensor(volume, device):
    return torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))[None, None].to(device)


def upsample(activation, shape):
    return F.interpolate(activation, size=shape, mode="trilinear", align_corners=False)


def max_pool_mask(mask, stride):
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    return (F.max_pool3d(tensor, stride) if stride > 1 else tensor)[0, 0].numpy() > 0


class ChannelProbe:
    """The chosen channels of one block, with signs and a refit sparse probe."""

    def __init__(self, feature_dir, block, channels, device):
        self.block, self.channels, self.device = block, list(channels), device
        self.stride = BLOCK_STRIDE[block]
        self.model = load_genesis().to(device)
        train_scans, _, _ = split_scans()
        train = BlockRows(feature_dir, block, train_scans, "pretrained", uniform_only=False)
        signs, _ = signed_channel_ranking(train)
        self.signs = signs[self.channels]
        self.standardise = Standardiser(train.x)
        x = self.standardise(train.x)
        probe = fit_probe(x, train.fissure, np.array(self.channels), np.random.default_rng(0))
        self.weights, self.bias = probe.coef_[0], probe.intercept_[0]

    def activations(self, patch):
        """(channels, X/s, Y/s, Z/s) signed activations of the chosen channels."""
        with torch.no_grad():
            features = self.model.forward_features(to_tensor(patch, self.device))[self.block][0]
        chosen = features[self.channels].cpu().numpy()
        return chosen * self.signs[:, None, None, None]

    def probe_score(self, patch):
        with torch.no_grad():
            features = self.model.forward_features(to_tensor(patch, self.device))[self.block][0].cpu().numpy()
        x = features[self.channels]
        mean = self.standardise.mean[self.channels][:, None, None, None]
        std = self.standardise.std[self.channels][:, None, None, None]
        logit = np.tensordot(self.weights, (x - mean) / std, axes=1) + self.bias
        return 1 / (1 + np.exp(-logit))


# ---------- interventions ----------

def erase(patch, mask):
    filled = patch.copy()
    filled[mask] = ndimage.median_filter(patch, size=7)[mask]
    return filled


def shifted_control_mask(fissure, lung):
    """The sheet mask moved 12 voxels in x, kept in lung and away from any fissure."""
    shifted = np.roll(fissure, ERASE_SHIFT, axis=0)
    shifted[:ERASE_SHIFT] = False
    far_from_fissure = ~ndimage.binary_dilation(fissure, iterations=4)
    return shifted & lung & far_from_fissure


def mean_on(activation, mask):
    return activation[:, mask].mean(axis=1) if mask.any() else np.full(len(activation), np.nan)


def gaussian_plane(shape, centre, normal, sigma, radius):
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1) - centre
    distance = grid @ normal
    in_plane = np.linalg.norm(grid - distance[..., None] * normal, axis=-1)
    return np.exp(-distance**2 / (2 * sigma**2)) * (in_plane <= radius), np.abs(distance) <= 1.0


def gaussian_tube(shape, centre, direction, sigma, half_length):
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1) - centre
    along = grid @ direction
    radial = np.linalg.norm(grid - along[..., None] * direction, axis=-1)
    inside = np.abs(along) <= half_length
    return np.exp(-radial**2 / (2 * sigma**2)) * inside, (radial <= 1.0) & inside


def random_unit_vector(rng):
    vector = rng.normal(size=3)
    return vector / np.linalg.norm(vector)


class Interventions:
    def __init__(self, probe, scans, patches_per_scan=6, seed=1):
        self.probe, self.scans = probe, scans
        self.sampler = PatchSampler(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.patches_per_scan = patches_per_scan

    def sheet_contrast(self, patch, fissure, lung):
        background = np.median(patch[lung & ~ndimage.binary_dilation(fissure, iterations=3)])
        return float(np.median(patch[fissure & lung]) - background)

    def erase_rows(self, scan):
        rows = []
        corners = self.sampler.corners(scan, 2 * self.patches_per_scan)[0::2]  # fissure-centred ones
        for corner in corners:
            crop = lambda volume: self.sampler.crop(volume, corner)
            patch, fissure, lung = hu_to_genesis_input(crop(scan.hu)), crop(scan.fissure) & crop(scan.lung), crop(scan.lung)
            control = shifted_control_mask(fissure, lung)
            if not fissure.any() or not control.any():
                continue
            s = self.probe.stride
            cells, control_cells = max_pool_mask(fissure, s), max_pool_mask(control, s)
            before = self.probe.activations(patch)
            erased = self.probe.activations(erase(patch, fissure))
            controlled = self.probe.activations(erase(patch, control))
            rows.append({
                "fissure_before": mean_on(before, cells), "fissure_after": mean_on(erased, cells),
                "control_before": mean_on(before, control_cells), "control_after": mean_on(controlled, control_cells),
                "contrast": self.sheet_contrast(patch, fissure, lung),
            })
        return rows

    def clear_lung_corner(self, scan):
        far = scan.lung & ~ndimage.binary_dilation(scan.fissure, iterations=INSERT_CLEARANCE)
        points = np.argwhere(far)
        for _ in range(200):
            centre = points[self.rng.integers(len(points))]
            corner = np.clip(centre - self.sampler.shape // 2, 0, np.array(scan.hu.shape) - self.sampler.shape)
            if self.sampler.crop(scan.lung, corner).mean() > 0.7:
                return corner
        return None

    def insert_rows(self, scan, contrast):
        rows = []
        for _ in range(self.patches_per_scan):
            corner = self.clear_lung_corner(scan)
            if corner is None:
                continue
            patch = hu_to_genesis_input(self.sampler.crop(scan.hu, corner))
            centre = self.sampler.shape / 2
            before = self.probe.activations(patch)
            row = {}
            for kind in ("plane", "tube"):
                if kind == "plane":
                    profile, core = gaussian_plane(patch.shape, centre, random_unit_vector(self.rng), PLANE_SIGMA, PLANE_RADIUS)
                else:
                    profile, core = gaussian_tube(patch.shape, centre, random_unit_vector(self.rng), TUBE_SIGMA, TUBE_HALF_LENGTH)
                after = self.probe.activations(np.clip(patch + contrast * profile, 0, 1))
                cells = max_pool_mask(core, self.probe.stride)
                row[kind] = (mean_on(after, cells) - mean_on(before, cells))
            rows.append(row)
        return rows

    def run(self):
        erase_rows, insert_rows = [], []
        for record in self.scans:
            scan = LoadedScan(record)
            scan_erase = self.erase_rows(scan)
            erase_rows += scan_erase
            contrast = float(np.nanmedian([r["contrast"] for r in scan_erase])) if scan_erase else 0.03
            insert_rows += self.insert_rows(scan, contrast)
            print(f"interventions: {record.short} done", flush=True)
        return erase_rows, insert_rows


def summarise_interventions(channels, erase_rows, insert_rows):
    stack = lambda rows, key: np.stack([r[key] for r in rows])
    fb, fa = stack(erase_rows, "fissure_before"), stack(erase_rows, "fissure_after")
    cb, ca = stack(erase_rows, "control_before"), stack(erase_rows, "control_after")
    plane, tube = stack(insert_rows, "plane"), stack(insert_rows, "tube")
    summary = []
    for i, channel in enumerate(channels):
        summary.append({
            "channel": int(channel),
            "erase_fissure_relative_change": float(np.nanmean((fa[:, i] - fb[:, i]) / (np.abs(fb[:, i]) + 1e-6))),
            "erase_control_relative_change": float(np.nanmean((ca[:, i] - cb[:, i]) / (np.abs(cb[:, i]) + 1e-6))),
            "insert_plane_increase": float(np.nanmean(plane[:, i])),
            "insert_tube_increase": float(np.nanmean(tube[:, i])),
            "insert_plane_wins_frac": float(np.nanmean(plane[:, i] > tube[:, i])),
        })
    contrasts = [r["contrast"] for r in erase_rows]
    return {"channels": summary, "patches_erased": len(erase_rows), "patches_inserted": len(insert_rows),
            "median_sheet_contrast": float(np.nanmedian(contrasts))}


# ---------- stability ----------

def stability(feature_dir, block, channels, seeds=(0, 1, 2, 3), top=10):
    hits = {int(c): 0 for c in channels}
    for seed in seeds:
        train_scans, _, _ = split_scans(seed)
        _, train_ap = signed_channel_ranking(BlockRows(feature_dir, block, train_scans, "pretrained", uniform_only=False))
        top_channels = set(np.argsort(-train_ap)[:top].tolist())
        for c in hits:
            hits[c] += c in top_channels
    return {c: n / len(seeds) for c, n in hits.items()}


# ---------- overlays ----------

def to_gray(slice_, low, high):
    return Image.fromarray((np.clip((slice_ - low) / (high - low + 1e-9), 0, 1) * 255).astype(np.uint8))


def overlay_sheet(ct, sheet):
    rgb = np.stack([ct] * 3, -1).astype(np.float32)
    rgb[sheet] = 0.4 * rgb[sheet] + 0.6 * np.array([255, 60, 60])
    return Image.fromarray(rgb.astype(np.uint8))


def plane_slices(volume, centre):
    x, y, z = centre
    return {"axial": volume[:, :, z].T[::-1], "coronal": volume[:, y, :].T[::-1], "sagittal": volume[x, :, :].T[::-1]}


def overlay_figure(probe, record, out_path, tile=192):
    scan = LoadedScan(record)
    sampler = PatchSampler(seed=7)
    corner = sampler.corners(scan, 1)[0]
    crop = lambda volume: sampler.crop(volume, corner)
    hu, patch = crop(scan.hu), hu_to_genesis_input(crop(scan.hu))
    sheet = crop(scan.fissure) & crop(scan.lung)
    shape = patch.shape
    activations = upsample(torch.from_numpy(probe.activations(patch))[None], shape)[0].numpy()
    score = upsample(torch.from_numpy(probe.probe_score(patch)[None, None].astype(np.float32)), shape)[0, 0].numpy()
    columns = [("CT", hu, "ct"), ("sheet", hu, "sheet")] + \
              [(f"ch {c}", a, "act") for c, a in zip(probe.channels[:4], activations[:4])] + [("probe", score, "prob")]
    centre = np.array(shape) // 2
    rows = []
    for plane in ("axial", "coronal", "sagittal"):
        images = []
        for title, volume, kind in columns:
            sl = plane_slices(volume, centre)[plane]
            if kind == "ct":
                img = to_gray(sl, -1000, -300)
            elif kind == "sheet":
                img = overlay_sheet(np.asarray(to_gray(sl, -1000, -300)), plane_slices(sheet, centre)[plane])
            elif kind == "prob":
                img = to_gray(sl, 0, 1)
            else:
                img = to_gray(sl, 0, np.percentile(volume, 99.5))
            images.append((title, img.convert("RGB").resize((tile, int(tile * sl.shape[0] / sl.shape[1])))))
        rows.append((plane, images))
    height = sum(max(i.height for _, i in imgs) + 18 for _, imgs in rows)
    sheet_img = Image.new("RGB", (len(columns) * (tile + 4) + 70, height), "white")
    draw, y = ImageDraw.Draw(sheet_img), 0
    for plane, images in rows:
        draw.text((2, y + 30), plane, fill="black")
        for i, (title, img) in enumerate(images):
            x = 70 + i * (tile + 4)
            draw.text((x, y + 2), title, fill="black")
            sheet_img.paste(img, (x, y + 16))
        y += max(i.height for _, i in images) + 18
    sheet_img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", required=True)
    parser.add_argument("--channels", type=int, nargs="+", required=True)
    parser.add_argument("--features", type=Path, default=Path("outputs/fissure_probe/features"))
    parser.add_argument("--out", type=Path, default=Path("outputs/fissure_probe/evidence"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.set_num_threads(8)
    args.out.mkdir(parents=True, exist_ok=True)
    probe = ChannelProbe(args.features, args.block, args.channels, torch.device(args.device))
    _, test, _ = split_scans()
    test_records = [r for r in list_scans() if r.short in test]
    for record in test_records:
        overlay_figure(probe, record, args.out / f"overlay_{args.block}_{record.short}.png")
        print(f"overlay: {record.short} done", flush=True)
    erase_rows, insert_rows = Interventions(probe, test_records).run()
    result = {
        "block": args.block, "channels": args.channels,
        "interventions": summarise_interventions(args.channels, erase_rows, insert_rows),
        "stability_top10_frac": stability(args.features, args.block, args.channels),
    }
    (args.out / f"evidence_{args.block}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["interventions"]["channels"], indent=1))


if __name__ == "__main__":
    main()
