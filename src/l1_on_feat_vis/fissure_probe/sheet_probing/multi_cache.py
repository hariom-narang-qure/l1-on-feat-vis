"""Full 74 x 74 maps of the pooled units on the segmentation head's slices.

pooled_selection.py picks units by sampling tokens; the head needs every
token of every slice in seg_head's cache (seg_cache/<scan>.npz, 40 slices
per scan). Per scan:
  - 2D networks (slice_features.MODELS): one forward per slice, the block
    resampled to the shared grid exactly as in slice_features.py.
  - 3D nnU-Nets (nnunet_features.NETWORKS): one forward over the lung crop,
    then each stage sampled at every token centre of each slice.
Units are multiplied by their train sign, so "high = fissure" for every column.
Output: multi_cache/<scan>.npz with units (slices, K, 74, 74) float16.
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from l1_on_feat_vis.fissure_probe.data.data import LoadedScan, list_scans
from l1_on_feat_vis.fissure_probe.sheet_probing.dino_features import GRID, WINDOW_HU
from l1_on_feat_vis.fissure_probe.models.nnunet_features import NETWORKS, NnUNet, lung_crop, native_to_network, sample_stage
from l1_on_feat_vis.fissure_probe.sheet_probing.probes import split_scans
from l1_on_feat_vis.fissure_probe.sheet_probing.slice_features import MODELS, to_shared_grid


def model_input(grey, spec, device):
    """Same preprocessing as slice_features.MultiNetwork.model_input."""
    rgb = (np.repeat(grey[..., None], 3, -1) - spec.mean) / spec.std
    x = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32))[None].to(device)
    return torch.nn.functional.interpolate(x, size=(spec.input_px, spec.input_px), mode="bilinear", align_corners=False)


def grouped(units):
    """{model: {block: [(column index, unit, sign)]}} keeping the pooled column order."""
    groups = {}
    for column, u in enumerate(units):
        model = u["block"].split("/")[0]
        groups.setdefault(model, {}).setdefault(u["block"], []).append((column, u["unit"], u["sign"]))
    return groups


class UnitMaps:
    def __init__(self, units, device):
        self.units, self.device = units, device
        self.groups = grouped(units)
        self.vits = {m: MODELS[m] for m in self.groups if m in MODELS}
        self.vit_models = {m: spec.build(True).to(device).eval() for m, spec in self.vits.items()}
        self.nnunets = {m: NnUNet(m, NETWORKS[m], device) for m in self.groups if m in NETWORKS}

    def slice_maps_2d(self, hu_slice, out, slice_index):
        grey = (np.clip(hu_slice, *WINDOW_HU) - WINDOW_HU[0]) / (WINDOW_HU[1] - WINDOW_HU[0])
        for model, spec in self.vits.items():
            x = model_input(grey, spec, self.device)
            with torch.no_grad():
                blocks = spec.blocks(self.vit_models[model], x)
            for block, members in self.groups[model].items():
                grid = to_shared_grid(blocks[block])  # (GRID*GRID, C)
                for column, unit, sign in members:
                    out[slice_index, column] = (sign * grid[:, unit]).reshape(GRID, GRID).cpu().numpy()

    def scan_maps_3d(self, record, z_list, out):
        if not self.nnunets:
            return
        scan = LoadedScan(record)
        spacing = nib.load(record.folder / "scan.nii.gz").header.get_zooms()[:3]
        lo, hi = lung_crop(scan, spacing)
        crop = scan.hu[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        size = scan.hu.shape[0]
        i, j = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
        for model, network in self.nnunets.items():
            x, zoom = network.preprocess(crop, spacing)
            _, stages = network.forward(x)
            for slice_index, z in enumerate(z_list):
                points = np.stack([(i.ravel() + 0.5) * size / GRID, (j.ravel() + 0.5) * size / GRID, np.full(GRID * GRID, z)], 1)
                coords = native_to_network(points, lo, zoom, network.plans.transpose_forward)
                for block, members in self.groups[model].items():
                    sampled = sample_stage(stages[block.split("/")[1]], coords, None, x.shape[2:])
                    for column, unit, sign in members:
                        out[slice_index, column] = (sign * sampled[:, unit]).reshape(GRID, GRID).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--seg-cache", type=Path, default=Path("outputs/fissure_probe/seg_cache"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reverse", action="store_true", help="walk the scans last to first (a second process)")
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    units = next(p for p in selection["probes"] if p["k"] == args.k)["units"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "units.json").write_text(json.dumps(units, indent=1))
    maps = UnitMaps(units, torch.device(args.device))
    train, test, _ = split_scans()
    records = [r for r in list_scans() if r.short in set(train) | set(test)]
    for record in reversed(records) if args.reverse else records:
        path = args.out / f"{record.short}.npz"
        claim = path.with_suffix(".working")
        if path.exists() or claim.exists():
            continue
        claim.touch()  # two processes walking opposite ends never take the same scan
        cache = np.load(args.seg_cache / f"{record.short}.npz")
        out = np.zeros((len(cache["z"]), len(units), GRID, GRID), np.float32)
        for slice_index in range(len(cache["z"])):
            maps.slice_maps_2d(cache["hu"][slice_index], out, slice_index)
        maps.scan_maps_3d(record, [int(z) for z in cache["z"]], out)
        np.savez(path, units=out.astype(np.float16))
        claim.unlink()
        print(f"multi cache: {record.short} {out.shape}", flush=True)


if __name__ == "__main__":
    main()
