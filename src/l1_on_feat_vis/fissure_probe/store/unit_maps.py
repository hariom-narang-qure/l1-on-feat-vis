"""Full 74 x 74 maps of chosen units on every line slice (train, test, near-empty), for the head.

The store (store.py) holds sampled tokens only; the head needs every token of
every slice. Maps are computed exactly as the store computes its columns
(same canvases, models, inputs and resampling), so a unit's map at a stored
row equals its store value up to int8 rounding (checked by check_against_store).

Layout: maps/<model>__<layer>/<unit>.npy, one (slices, 74, 74) float16 array per
unit, raw (unsigned) values, slices in the order of maps/slices.json:
all line slices of every scan, scans in split order (train, test, near-empty).
A run computes only the units that are missing, one pass over the slices for
all of them (2D models: one forward per slice; nnU-Net: one forward per scan).
"""

import argparse
import json
import time

import numpy as np
import torch

from l1_on_feat_vis.fissure_probe.sheet_probing.probes import split_scans
from l1_on_feat_vis.fissure_probe.data.rows import GRID
from l1_on_feat_vis.fissure_probe.store.store import (MODELS_2D, Taps3D, canvas_grey, model_input, scan_folder,
                                                token_centres)
from l1_on_feat_vis.fissure_probe.data.views import STORE, VIEWS, ScanViews

MAPS = STORE / "maps"
ALL_TOKENS = np.arange(GRID * GRID)


def slice_order():
    train, test, near_empty = split_scans()
    order = []
    for scan in train + test + near_empty:
        n = len(np.load(STORE / "slices" / f"{scan}.npz")["k"])
        order += [(scan, s) for s in range(n)]
    return order


def unit_path(layer, unit):
    return MAPS / layer.replace("/", "__") / f"{unit}.npy"


def missing(units):
    return sorted({(layer, unit) for layer, unit in units if not unit_path(layer, unit).exists()})


class MapWriter:
    def __init__(self, units, device):
        self.device = device
        self.units = units
        self.by_layer = {}
        for layer, unit in units:
            self.by_layer.setdefault(layer, []).append(unit)
        models = {layer.split("/")[0] for layer in self.by_layer}
        self.specs = [MODELS_2D[m] for m in models if m in MODELS_2D]
        for spec in self.specs:
            spec.load(device)
        self.taps = {m: Taps3D(m, device) for m in models if m not in MODELS_2D}

    def run(self):
        order = slice_order()
        maps = {(layer, unit): np.zeros((len(order), GRID, GRID), np.float16) for layer, unit in self.units}
        scans = list(dict.fromkeys(scan for scan, _ in order))
        start = time.time()
        for n, scan in enumerate(scans):
            index = [i for i, (s, _) in enumerate(order) if s == scan]
            cache = np.load(STORE / "slices" / f"{scan}.npz")
            self.maps_2d(cache, index, maps)
            self.maps_3d(scan, cache, index, maps)
            print(f"  maps {scan} ({n + 1}/{len(scans)}) at {time.time() - start:.0f}s", flush=True)
        for (layer, unit), values in maps.items():
            path = unit_path(layer, unit)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, values)
        (MAPS / "slices.json").write_text(json.dumps(order))

    def maps_2d(self, cache, index, maps):
        tokens = torch.from_numpy(ALL_TOKENS).to(self.device)
        for s, i in enumerate(index):
            grey = canvas_grey(cache["hu"][s])
            for spec in self.specs:
                x = model_input(grey, spec.mean, spec.std, spec.input_px, self.device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    layers = spec.layers(x, tokens)
                for layer, values in layers.items():
                    key = f"{spec.name}/{layer}"
                    for unit in self.by_layer.get(key, []):
                        maps[(key, unit)][i] = values[:, unit].float().reshape(GRID, GRID).cpu().numpy()

    def maps_3d(self, scan, cache, index, maps):
        if not self.taps:
            return
        views = ScanViews(scan_folder(scan))
        p, q = token_centres(ALL_TOKENS)
        voxels = np.concatenate([views.geometry[VIEWS[cache["view"][s]]].canvas_to_voxel(p, q, int(cache["k"][s]))
                                 for s in range(len(index))])
        for name, taps in self.taps.items():
            wanted = {layer.split("/")[1]: units for layer, units in self.by_layer.items() if layer.startswith(name + "/")}
            layers = taps.layers(views.hu, views.spacing, views.lobes, voxels, wanted)
            for layer, values in layers.items():
                for unit, column in zip(wanted[layer], values.T):
                    maps[(f"{name}/{layer}", unit)][index] = column.numpy().reshape(len(index), GRID, GRID)


def load_maps(units, signs=None):
    """(slices, K, 74, 74) float32 maps for [(layer, unit)], multiplied by their signs."""
    stacked = np.stack([np.load(unit_path(layer, unit), mmap_mode="r") for layer, unit in units], 1).astype(np.float32)
    if signs is not None:
        stacked *= np.asarray(signs, np.float32)[None, :, None, None]
    return stacked


def check_against_store(layer, unit, scan_index=0):
    """Max |map - store| at the stored rows of one train scan, relative to the unit's range."""
    from l1_on_feat_vis.fissure_probe.data.rows import Rows
    from l1_on_feat_vis.fissure_probe.store.store import load_layer
    rows = Rows()
    scan = rows.scans[scan_index]
    index = rows.of_scan(scan_index)
    stored = load_layer(layer, [scan], [unit])[:, 0]
    order = slice_order()
    offset = next(i for i, (s, _) in enumerate(order) if s == scan)
    values = np.load(unit_path(layer, unit))[offset + rows.slice[index]].reshape(len(index), -1)
    mapped = values[np.arange(len(index)), rows.token[index]].astype(np.float32)
    return float(np.abs(mapped - stored).max() / (stored.max() - stored.min() + 1e-9))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=str, required=True, help="json list of {layer, unit} (or a selection json with 'pool')")
    args = parser.parse_args()
    data = json.loads(open(args.units).read())
    entries = data["pool"] if isinstance(data, dict) else data
    units = missing([(e["layer"], e["unit"]) for e in entries])
    print(f"{len(units)} unit maps to compute", flush=True)
    if units:
        torch.set_num_threads(8)
        MapWriter(units, torch.device("cuda")).run()


if __name__ == "__main__":
    main()
