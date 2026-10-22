"""E0: the old pooled-64 recipe (pooled_selection.py), re-run on the new rows and ground truth.

Old recipe: the 30 sampled blocks of the 7 networks (stage outputs and a few
ViT layers), per block the 8 units with the best train fissure-vs-ring AUROC,
then the L1 path over the 240 pooled candidates on line-vs-ring rows; the
first 64 to enter are the pool. Here the same steps run on all 12 train
scans' rows of the new store; layer names map to the store's:
  ts_organs/enc{s} -> enc{s}c1 (last conv of the stage), dec{s} -> dec{s}c1
  convnext_b/stage{1,2,3} -> s1b2, s2b26, s3b2 (last block of the stage)
  resnet50/stage{2,3,4} -> l2b3, l3b5, l4b2
"""

import json

import numpy as np

from l1_on_feat_vis.fissure_probe.selection.cv_scores import load_scores
from l1_on_feat_vis.fissure_probe.selection.forward_select import SELECTION
from l1_on_feat_vis.fissure_probe.sheet_probing.probes import Standardiser, l1_entry_order
from l1_on_feat_vis.fissure_probe.data.rows import Rows
from l1_on_feat_vis.fissure_probe.store.store import load_layer

OLD_LAYERS = ([f"ts_organs/enc{s}c1" for s in range(6)] + [f"ts_organs/dec{s}c1" for s in range(5)]
              + ["convnext_b/s1b2", "convnext_b/s2b26", "convnext_b/s3b2"]
              + [f"dinov2_l/res{l}" for l in (8, 12, 16, 20, 23)] + [f"dinov2_s/res{l}" for l in (5, 8, 11)]
              + [f"rad_dino/res{l}" for l in (5, 8, 11)] + ["resnet50/l2b3", "resnet50/l3b5", "resnet50/l4b2"]
              + [f"dinov2_b/res{l}" for l in (8, 11)])
PER_BLOCK, K = 8, 64


def pooled_auc_ring(metrics):
    """Mean over folds of the signed line-vs-ring AUROC (view all): the old 'train ring AUROC'."""
    return metrics[:, :, 0, 0].mean(1)


def select():
    rows = Rows()
    candidates, columns = [], []
    for layer in OLD_LAYERS:
        metrics, sign = load_scores(layer)
        top = np.argsort(-pooled_auc_ring(metrics))[:PER_BLOCK]
        x = load_layer(layer, rows.scans, list(top)) * sign[top]
        candidates.append(x)
        columns += [{"layer": layer, "unit": int(u), "sign": int(sign[u])} for u in top]
    x = np.concatenate(candidates, 1)
    keep = rows.line | rows.ring
    x = Standardiser(x)(x)
    order = l1_entry_order(x[keep], rows.line[keep], np.random.default_rng(0))
    pool = [columns[i] for i in order[:K]]
    (SELECTION / "old_recipe.json").write_text(json.dumps({"pool": pool, "candidates": columns}, indent=1))
    return pool


if __name__ == "__main__":
    pool = select()
    print(f"{len(pool)} units from {sorted({p['layer'].split('/')[0] for p in pool})}")
