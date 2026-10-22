"""Baseline: frozen whole layers instead of selected units.

Every layer of the store is scored with the same token-level probe and folds as
forward_select.py, on all of its units at once (ridge logistic regression,
standardised, fitted per fold on the other two folds). The best layers by CV AP
are the candidates for a head on the whole layer; nothing here sees the test scans.
Output: selection/whole_layers.json (per layer: units, CV AP, per view) and
selection/layer_<model>__<layer>.json (the layer as a unit list for head.py).
"""

import json
import time

import numpy as np
import torch

from l1_on_feat_vis.fissure_probe.selection.forward_select import SELECTION, Candidates, ForwardSelection
from l1_on_feat_vis.fissure_probe.data.rows import Rows
from l1_on_feat_vis.fissure_probe.store.store import layer_list, load_layer


def layer_units(layer, rows):
    width = load_layer(layer, rows.scans[:1]).shape[1]
    return [(layer, u, 1) for u in range(width)]


def score_layer(layer, rows, device):
    units = layer_units(layer, rows)
    candidates = Candidates(rows, device, units=units)
    selection = ForwardSelection(rows, candidates, device)
    report = selection.cv_report(selection.fit_folds(list(range(len(units))))[1])
    del candidates, selection
    torch.cuda.empty_cache()
    return len(units), report


def save_layer_pool(layer, rows):
    pool = [{"layer": layer, "unit": u, "sign": 1} for _, u, _ in layer_units(layer, rows)]
    path = SELECTION / f"layer_{layer.replace('/', '__')}.json"
    path.write_text(json.dumps({"pool": pool}))
    return path


def main():
    device = torch.device("cuda")
    rows = Rows()
    out_path = SELECTION / "whole_layers.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for layer in layer_list():
        if layer in results:
            continue
        start = time.time()
        width, report = score_layer(layer, rows, device)
        results[layer] = {"units": width, "ap_mean": report["ap_mean"], "ap": report["ap"],
                          "ap_view": {v: float(np.mean(a)) for v, a in report["ap_view"].items()},
                          "auc_ring": float(np.mean(report["auc_ring"]))}
        out_path.write_text(json.dumps(results, indent=1))
        print(f"{layer:22s} {width:5d} units | CV AP {report['ap_mean']:.4f} | {time.time() - start:.0f}s", flush=True)
    best = sorted(results, key=lambda l: -results[l]["ap_mean"])[:10]
    for layer in best:
        print(f"best {layer:22s} {results[layer]['units']:5d} units CV AP {results[layer]['ap_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
