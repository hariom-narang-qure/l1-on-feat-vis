"""Evidence that chosen DINOv2 units respond to fissures, on held-out axial slices.

Units: dimensions of one DINOv2 block (MLP neurons or residual-stream units),
picked by fissure-vs-ring AUROC on the train scans (probes.py). Their sign is
chosen on train too. Everything below runs on the 5 test scans only.

  1. Overlays: CT (lung window) | sheet | each unit's map, on fissure-rich slices.
  2. Erase (input intervention, HU domain): replace the sheet's pixels with the
     7 x 7 median around them (local parenchyma). Mean response on the sheet's
     tokens before vs after. Control: the same erase on the sheet mask shifted
     12 px, kept in lung and > 4 px from any fissure.
  3. Insert, in lung > 20 px from any fissure. Amplitude = that slice's sheet
     contrast: median HU on the undilated sheet minus median parenchyma HU
     (the dilated 3-voxel label would dilute it with off-line pixels).
     - thin line: Gaussian profile sigma 0.8 px, 60 px long (a fissure in plane)
     - thick line: sigma 2.5 px, same peak (a vessel running in plane)
     - thick line, equal mass: sigma 2.5 px, peak scaled by 0.8 / 2.5 so it
       adds the same total intensity as the thin line
     - disc: radius 3 px, same peak (a vessel in cross-section)
     Response increase on the structure's tokens.
  Statistics are paired per slice: medians of differences and Wilcoxon
  signed-rank p-values.
  4. Dose response: thin lines at 0.25, 0.5, 1, 2 x the sheet contrast.
  5. Stability: how often each unit is in the train top 10 over 4 splits.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage, stats

from l1_on_feat_vis.fissure_probe.data import LoadedScan, list_scans
from l1_on_feat_vis.fissure_probe.dino_features import GRID, DinoNetworks, to_grid
from l1_on_feat_vis.fissure_probe.probes import BlockRows, local_channel_ranking, split_scans

ERASE_SHIFT_PX, CLEARANCE_PX = 12, 20
LINE_HALF_LENGTH = 30
THIN_SIGMA, THICK_SIGMA, DISC_RADIUS = 0.8, 2.5, 3.0
DOSES = (0.25, 0.5, 1.0, 2.0)


class DinoUnits:
    def __init__(self, feature_dir, block, units, device):
        self.block, self.units = block, list(units)
        self.networks = DinoNetworks(device)
        self.networks.models = {"pretrained": self.networks.models["pretrained"]}
        train_scans, _, _ = split_scans()
        signs, _ = local_channel_ranking(BlockRows(feature_dir, block, train_scans, "pretrained", uniform_only=False))
        self.signs = signs[self.units]

    def maps(self, hu_slice):
        """(units, GRID, GRID) signed activations."""
        x, _ = self.networks.slice_input(hu_slice)
        tokens = self.networks.features(x)["pretrained"][self.block]
        chosen = tokens[:, self.units].float().cpu().numpy().T.reshape(len(self.units), GRID, GRID)
        return chosen * self.signs[:, None, None]


def mean_on(maps, token_mask):
    return maps[:, token_mask].mean(axis=1) if token_mask.any() else np.full(len(maps), np.nan)


def tokens_of(pixel_mask):
    return to_grid(pixel_mask, "max") > 0


def erase(hu_slice, mask):
    filled = hu_slice.copy()
    filled[mask] = ndimage.median_filter(hu_slice, size=7)[mask]
    return filled


def shifted_control(fissure, lung):
    shifted = np.roll(fissure, ERASE_SHIFT_PX, axis=0)
    shifted[:ERASE_SHIFT_PX] = False
    return shifted & lung & ~ndimage.binary_dilation(fissure, iterations=4)


def line_profile(shape, centre, direction, sigma):
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1) - centre
    along = grid @ direction
    across = grid @ np.array([-direction[1], direction[0]])
    inside = np.abs(along) <= LINE_HALF_LENGTH
    return np.exp(-across**2 / (2 * sigma**2)) * inside, (np.abs(across) <= max(1.0, sigma)) & inside


def scaled(profile_and_core, factor):
    profile, core = profile_and_core
    return profile * factor, core


def disc_profile(shape, centre):
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1) - centre
    inside = np.linalg.norm(grid, axis=-1) <= DISC_RADIUS
    return inside.astype(np.float32), inside


class SliceInterventions:
    def __init__(self, units, records, slices_per_scan=10, seed=3):
        self.units, self.records = units, records
        self.rng = np.random.default_rng(seed)
        self.slices_per_scan = slices_per_scan

    def fissure_slices(self, scan):
        counts = (scan.fissure & scan.lung).sum(axis=(0, 1))
        candidates = np.flatnonzero(counts >= 200)
        return self.rng.choice(candidates, min(self.slices_per_scan, len(candidates)), replace=False)

    def sheet_contrast_hu(self, hu, sheet, fissure, lung):
        parenchyma = lung & ~ndimage.binary_dilation(fissure, iterations=3) & (hu < -600)
        return float(np.median(hu[sheet & lung]) - np.median(hu[parenchyma]))

    def erase_row(self, hu, fissure, lung):
        control = shifted_control(fissure, lung)
        before = self.units.maps(hu)
        return {
            "fissure_before": mean_on(before, tokens_of(fissure)),
            "fissure_after": mean_on(self.units.maps(erase(hu, fissure)), tokens_of(fissure)),
            "control_before": mean_on(before, tokens_of(control)),
            "control_after": mean_on(self.units.maps(erase(hu, control)), tokens_of(control)),
        }

    def clear_centre(self, fissure, lung):
        far = lung & ~ndimage.binary_dilation(fissure, iterations=CLEARANCE_PX)
        far &= ndimage.binary_erosion(lung, iterations=LINE_HALF_LENGTH // 2)
        points = np.argwhere(far)
        return points[self.rng.integers(len(points))] if len(points) else None

    def insert_row(self, hu, fissure, lung, contrast):
        centre = self.clear_centre(fissure, lung)
        if centre is None:
            return None
        before = self.units.maps(hu)
        angle = self.rng.uniform(0, np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        structures = {
            "thin_line": line_profile(hu.shape, centre, direction, THIN_SIGMA),
            "thick_line": line_profile(hu.shape, centre, direction, THICK_SIGMA),
            "thick_line_equal_mass": scaled(line_profile(hu.shape, centre, direction, THICK_SIGMA), THIN_SIGMA / THICK_SIGMA),
            "disc": disc_profile(hu.shape, centre),
        }
        row = {}
        for name, (profile, core) in structures.items():
            after = self.units.maps(hu + contrast * profile)
            row[name] = mean_on(after, tokens_of(core)) - mean_on(before, tokens_of(core))
        thin, core = structures["thin_line"]
        for dose in DOSES:
            row[f"dose_{dose}"] = mean_on(self.units.maps(hu + dose * contrast * thin), tokens_of(core)) - mean_on(before, tokens_of(core))
        return row

    def run(self):
        erase_rows, insert_rows, contrasts = [], [], []
        for record in self.records:
            scan = LoadedScan(record)
            for z in self.fissure_slices(scan):
                hu, fissure, lung = scan.hu[:, :, z], scan.fissure[:, :, z] & scan.lung[:, :, z], scan.lung[:, :, z]
                contrast = self.sheet_contrast_hu(hu, scan.sheet[:, :, z], fissure, lung)
                contrasts.append(contrast)
                erase_rows.append(self.erase_row(hu, fissure, lung))
                row = self.insert_row(hu, fissure, lung, contrast)
                if row is not None:
                    insert_rows.append(row)
            print(f"interventions: {record.short} done", flush=True)
        return erase_rows, insert_rows, contrasts


def wilcoxon_p(differences):
    differences = differences[np.isfinite(differences)]
    if len(differences) < 5 or not np.any(differences):
        return float("nan")
    return float(stats.wilcoxon(differences).pvalue)


def summarise(units, erase_rows, insert_rows, contrasts):
    """Per unit, paired over slices. A unit's response scale is set by its
    spread over the slices' fissure tokens, so effects are comparable across units."""
    stack = lambda rows, key: np.stack([r[key] for r in rows])
    fb, fa, cb, ca = (stack(erase_rows, k) for k in ("fissure_before", "fissure_after", "control_before", "control_after"))
    out = []
    for i, unit in enumerate(units):
        scale = np.nanstd(fb[:, i]) + 1e-6
        fissure_change, control_change = (fa[:, i] - fb[:, i]) / scale, (ca[:, i] - cb[:, i]) / scale
        entry = {
            "unit": int(unit),
            "erase_fissure_median_change_sd": float(np.nanmedian(fissure_change)),
            "erase_control_median_change_sd": float(np.nanmedian(control_change)),
            "erase_fissure_drop_frac": float(np.nanmean(fissure_change < 0)),
            "erase_fissure_p": wilcoxon_p(fissure_change),
            "erase_fissure_vs_control_p": wilcoxon_p(fissure_change - control_change),
        }
        thin = stack(insert_rows, "thin_line")[:, i]
        for name in ("thin_line", "thick_line", "thick_line_equal_mass", "disc", *[f"dose_{d}" for d in DOSES]):
            entry[f"insert_{name}_sd"] = float(np.nanmedian(stack(insert_rows, name)[:, i]) / scale)
        entry["insert_thin_p"] = wilcoxon_p(thin)
        for other in ("thick_line", "thick_line_equal_mass", "disc"):
            difference = thin - stack(insert_rows, other)[:, i]
            entry[f"thin_beats_{other}_frac"] = float(np.nanmean(difference > 0))
            entry[f"thin_vs_{other}_p"] = wilcoxon_p(difference)
        out.append(entry)
    return {"units": out, "slices_erased": len(erase_rows), "slices_inserted": len(insert_rows),
            "median_sheet_contrast_hu": float(np.median(contrasts))}


def stability(feature_dir, block, units, seeds=(0, 1, 2, 3), top=10):
    hits = {int(u): 0 for u in units}
    for seed in seeds:
        train_scans, _, _ = split_scans(seed)
        _, score = local_channel_ranking(BlockRows(feature_dir, block, train_scans, "pretrained", uniform_only=False))
        top_units = set(np.argsort(-score)[:top].tolist())
        for u in hits:
            hits[u] += u in top_units
    return {u: n / len(seeds) for u, n in hits.items()}


def window(hu):
    return (np.clip((hu + 1000) / 800, 0, 1) * 255).astype(np.uint8)


def overlay(units, record, out_path, tile=256):
    scan = LoadedScan(record)
    counts = (scan.fissure & scan.lung).sum(axis=(0, 1))
    z = int(np.argsort(counts)[-len(counts) // 10])  # a slice in the top 10% by fissure content
    hu, sheet = scan.hu[:, :, z], scan.fissure[:, :, z] & scan.lung[:, :, z]
    maps = units.maps(hu)
    up = F.interpolate(torch.from_numpy(maps)[None], size=hu.shape, mode="bilinear", align_corners=False)[0].numpy()
    ct = window(hu).T
    panels = [("CT", Image.fromarray(ct).convert("RGB"))]
    rgb = np.stack([ct] * 3, -1).astype(np.float32)
    rgb[sheet.T] = 0.3 * rgb[sheet.T] + 0.7 * np.array([255, 60, 60])
    panels.append(("sheet", Image.fromarray(rgb.astype(np.uint8))))
    lung = scan.lung[:, :, z]
    for unit, m in zip(units.units[:4], up[:4]):
        low, high = np.percentile(m[lung], 1), np.percentile(m[lung], 99.5)
        heat = (np.clip((m - low) / (high - low + 1e-9), 0, 1) * lung).T
        panels.append((f"unit {unit}", Image.fromarray((heat * 255).astype(np.uint8)).convert("RGB")))
    figure = Image.new("RGB", (len(panels) * (tile + 4), tile + 18), "white")
    draw = ImageDraw.Draw(figure)
    for i, (title, img) in enumerate(panels):
        draw.text((i * (tile + 4) + 2, 2), title, fill="black")
        figure.paste(img.resize((tile, tile)), (i * (tile + 4), 16))
    figure.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", required=True)
    parser.add_argument("--units", type=int, nargs="+", required=True)
    parser.add_argument("--features", type=Path, default=Path("outputs/fissure_probe/dino_features"))
    parser.add_argument("--out", type=Path, default=Path("outputs/fissure_probe/dino_evidence"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.set_num_threads(8)
    args.out.mkdir(parents=True, exist_ok=True)
    units = DinoUnits(args.features, args.block, args.units, torch.device(args.device))
    _, test, _ = split_scans()
    records = [r for r in list_scans() if r.short in test]
    for record in records:
        overlay(units, record, args.out / f"overlay_{args.block}_{record.short}.png")
    erase_rows, insert_rows, contrasts = SliceInterventions(units, records).run()
    result = {"block": args.block, "units": args.units,
              "interventions": summarise(args.units, erase_rows, insert_rows, contrasts),
              "stability_top10_frac": stability(args.features, args.block, args.units)}
    (args.out / f"evidence_{args.block}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["interventions"], indent=1))


if __name__ == "__main__":
    main()
