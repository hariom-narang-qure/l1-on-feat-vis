"""Top-activating tokens of DINOv2 units on held-out slices ("dataset examples").

For each unit, over 8 fissure-containing axial slices per test scan:
  1. Compute the unit's signed map over all lung tokens.
  2. Take the 48 highest tokens over all slices.
  3. precision@48 = share of them on the fissure (any sheet pixel in the token),
     and the same for the TotalSegmentator lobe boundary. Compare with the
     base rate: the share of all lung tokens on those slices that are fissure.
  4. Tile 56 x 56 px CT crops centred on each top token (lung window, the
     sheet drawn in red, the token outlined) so the picks can be checked by eye.
A token covers ~6.9 px, so a crop shows the token and its surroundings.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from l1_on_feat_vis.fissure_probe.data.data import LoadedScan, list_scans
from l1_on_feat_vis.fissure_probe.evidence.dino_evidence import DinoUnits, window
from l1_on_feat_vis.fissure_probe.sheet_probing.dino_features import GRID, to_grid
from l1_on_feat_vis.fissure_probe.sheet_probing.probes import split_scans

TOP_N, CROP_PX, SLICES_PER_SCAN = 48, 56, 8
TOKEN_PX = 512 / GRID


class TokenPool:
    """Every lung token of the chosen slices with its unit activations and labels."""

    def __init__(self, units, records, seed=5):
        rng = np.random.default_rng(seed)
        self.activations, self.fissure, self.lobe_boundary, self.where = [], [], [], []
        self.slices = {}
        for record in records:
            scan = LoadedScan(record)
            counts = (scan.fissure & scan.lung).sum(axis=(0, 1))
            for z in rng.choice(np.flatnonzero(counts >= 200), SLICES_PER_SCAN, replace=False):
                self.add_slice(units, scan, record.short, int(z))
            print(f"top activations: {record.short} done", flush=True)
        self.activations = np.concatenate(self.activations, axis=1)
        self.fissure, self.lobe_boundary = np.concatenate(self.fissure), np.concatenate(self.lobe_boundary)

    def add_slice(self, units, scan, short, z):
        hu = scan.hu[:, :, z]
        lung = to_grid(scan.lung[:, :, z], "avg") > 0.5
        fissure = to_grid(scan.fissure[:, :, z] & scan.lung[:, :, z], "max") > 0
        boundary = to_grid(scan.lobe_boundary[:, :, z], "max") > 0
        maps = units.maps(hu)
        tokens = np.argwhere(lung)
        self.activations.append(maps[:, lung])
        self.fissure.append(fissure[lung])
        self.lobe_boundary.append(boundary[lung])
        key = (short, z)
        self.slices[key] = (hu, scan.fissure[:, :, z] & scan.lung[:, :, z])
        self.where += [(key, int(i), int(j)) for i, j in tokens]


def crop_tile(hu, sheet, i, j):
    """A crop around token (i, j) in the slice's display orientation (rows = y)."""
    cx, cy = int((i + 0.5) * TOKEN_PX), int((j + 0.5) * TOKEN_PX)
    half = CROP_PX // 2
    padded_hu = np.pad(hu, half, constant_values=-1000)
    padded_sheet = np.pad(sheet, half)
    patch = window(padded_hu[cx:cx + CROP_PX, cy:cy + CROP_PX]).T
    marks = padded_sheet[cx:cx + CROP_PX, cy:cy + CROP_PX].T
    rgb = np.stack([patch] * 3, -1).astype(np.float32)
    rgb[marks] = 0.5 * rgb[marks] + 0.5 * np.array([255, 40, 40])
    tile = Image.fromarray(rgb.astype(np.uint8)).resize((CROP_PX * 2, CROP_PX * 2), Image.NEAREST)
    box = (CROP_PX - TOKEN_PX, CROP_PX - TOKEN_PX, CROP_PX + TOKEN_PX, CROP_PX + TOKEN_PX)
    ImageDraw.Draw(tile).rectangle(box, outline=(255, 220, 0))
    return tile


def unit_sheet(pool, unit_index, order, path, columns=12):
    tile = CROP_PX * 2
    rows = -(-len(order) // columns)
    figure = Image.new("RGB", (columns * (tile + 2), rows * (tile + 2)), "white")
    for n, index in enumerate(order):
        key, i, j = pool.where[index]
        hu, sheet = pool.slices[key]
        figure.paste(crop_tile(hu, sheet, i, j), ((n % columns) * (tile + 2), (n // columns) * (tile + 2)))
    figure.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", required=True)
    parser.add_argument("--units", type=int, nargs="+", required=True)
    parser.add_argument("--features", type=Path, default=Path("outputs/fissure_probe/dino_features"))
    parser.add_argument("--out", type=Path, default=Path("outputs/fissure_probe/dino_evidence"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.set_num_threads(8)
    units = DinoUnits(args.features, args.block, args.units, torch.device(args.device))
    _, test, _ = split_scans()
    pool = TokenPool(units, [r for r in list_scans() if r.short in test])
    result = {"block": args.block, "tokens": int(len(pool.fissure)),
              "fissure_base_rate": float(pool.fissure.mean()),
              "lobe_boundary_base_rate": float(pool.lobe_boundary.mean()), "units": []}
    for index, unit in enumerate(args.units):
        order = np.argsort(-pool.activations[index])[:TOP_N]
        result["units"].append({"unit": unit,
                                "fissure_precision_at_48": float(pool.fissure[order].mean()),
                                "lobe_boundary_precision_at_48": float(pool.lobe_boundary[order].mean())})
        unit_sheet(pool, index, order, args.out / f"top_{args.block}_unit{unit}.png")
    (args.out / f"top_activations_{args.block}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
