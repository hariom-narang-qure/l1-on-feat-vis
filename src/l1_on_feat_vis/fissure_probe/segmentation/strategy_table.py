"""Phase 6: the strategy table from the test runs (heads/test_*.json), mean over seeds.

Every row is scored on the 5 test scans once (two seeds each); the near-empty
scans are reported separately. Rows were chosen on train-only CV before any of
these numbers existed.
"""

import json

import numpy as np

from l1_on_feat_vis.fissure_probe.segmentation.head import HEADS
from l1_on_feat_vis.fissure_probe.data.views import VIEWS

ROWS = [
    ("CT only", "ct_only"),
    ("E0 old recipe (30 blocks, top-8, L1 path)", "test_e0"),
    ("random generalisers, 84 (control)", "test_random84"),
    ("E1 all layers, CV forward selection, 84", "test_e1_step8"),
    ("random generalisers, 168 (control)", "test_random168"),
    ("E1 all layers, 168", "test_e1_step28"),
    ("E1 168 + 16 recall units", "test_p5_recall"),
    ("E1 168 + 16 precision units", "test_p5_precision"),
    ("random generalisers, 200 (control)", "test_random200"),
    ("E1 168 + recall + precision units, 200", "test_p5_both"),
]
METRICS = ("tolerant_dice", "completeness", "correctness", "cldice", "pixel_dice", "pixel_ap")


def mean_of(runs, part, key, view="all"):
    values = [r[part][view][key] for r in runs]
    return float(np.mean(values)), float(np.ptp(values)) if len(values) > 1 else 0.0


def table():
    lines = ["| Row | units | seeds | tolerant Dice (± half range) | completeness | correctness | clDice | pixel Dice | pixel AP | "
             "axial / coronal / sagittal | near-empty |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, name in ROWS:
        path = HEADS / f"{name}.json"
        if not path.exists():
            continue
        runs = [r for r in json.loads(path.read_text()) if r.get("test")]
        if name == "test_e0":
            runs += [r for r in json.loads((HEADS / "e0_old_recipe.json").read_text()) if r["seed"] not in {x["seed"] for x in runs}]
        td, spread = mean_of(runs, "test", "tolerant_dice")
        others = [f"{mean_of(runs, 'test', m)[0]:.3f}" for m in METRICS[1:]]
        views = " / ".join(f"{mean_of(runs, 'test', 'tolerant_dice', v)[0]:.3f}" for v in VIEWS)
        near = mean_of(runs, "near_empty", "tolerant_dice")[0]
        lines.append(f"| {label} | {runs[0]['k']} | {len(runs)} | {td:.3f} (±{spread / 2:.3f}) | " + " | ".join(others)
                     + f" | {views} | {near:.3f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
