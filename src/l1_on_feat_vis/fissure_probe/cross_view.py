"""Are the head's "false" segments real fissure that this view's agent did not trace?

Each view's lines cover only about half of what the other views trace through the
same slice (view_agreement.py). So for every test slice, the other two views'
line voxels lying in this slice's plane give sparse "cross points" of fissure
evidence independent of this view's labels. Per predicted skeleton segment:
  - confirmed: some cross point lies within TOL px of the segment.
The confirmation rate of false segments is compared with that of true segments
(segments matching this view's line): if false segments are confirmed nearly as
often as true ones, many of them are real fissure.
Voxel -> agent view pixel (inverse of views.agent_view):
  axial (r, c) = (Y-1-y, X-1-x); coronal (Z-1-z, X-1-x); sagittal (Z-1-z, Y-1-y).
"""

import argparse
import json

import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import skeletonize

from l1_on_feat_vis.fissure_probe.head import HEADS, SliceData, Trainer
from l1_on_feat_vis.fissure_probe.probes import split_scans
from l1_on_feat_vis.fissure_probe.seg_metrics import EIGHT, drop_small_components
from l1_on_feat_vis.fissure_probe.store import scan_folder
from l1_on_feat_vis.fissure_probe.views import BITS, CANVAS, STORE, VIEW_AXIS, VIEWS, ScanViews

TOL = 2


def voxel_to_view(points, view, shape_xyz):
    X, Y, Z = shape_xyz
    x, y, z = points.T
    if view == "axial":
        return Y - 1 - y, X - 1 - x
    if view == "coronal":
        return Z - 1 - z, X - 1 - x
    return Z - 1 - z, Y - 1 - y


def line_points(views):
    """{view: (n, 3) voxel coordinates of that view's line}, computed once per scan."""
    return {view: np.argwhere(views.lines[view]) for view in VIEWS}


def cross_points(views, points_by_view, view, k):
    """Canvas mask of other views' line voxels in this slice's plane."""
    mask = np.zeros((CANVAS, CANVAS), bool)
    axis = VIEW_AXIS[view]
    for other in VIEWS:
        if other == view:
            continue
        points = points_by_view[other]
        points = points[points[:, axis] == k]
        if not len(points):
            continue
        r, c = voxel_to_view(points, view, views.hu.shape)
        p, q = views.geometry[view].view_to_canvas(r, c)
        p, q = np.round(p).astype(int), np.round(q).astype(int)
        keep = (p >= 0) & (p < CANVAS) & (q >= 0) & (q < CANVAS)
        mask[p[keep], q[keep]] = True
    return mask


def segment_confirmation(prediction, labels, crosses):
    """(true segments, true confirmed, false segments, false confirmed, false length, false confirmed length)."""
    lung = (labels & BITS["lung"]) > 0
    line = (labels & BITS["line"]) > 0
    skeleton = skeletonize(prediction & lung) if prediction.any() else prediction
    near_line = ndimage.distance_transform_edt(~line) <= TOL if line.any() else np.zeros_like(line)
    near_cross = ndimage.distance_transform_edt(~crosses) <= TOL if crosses.any() else np.zeros_like(line)
    components, n = ndimage.label(skeleton, EIGHT)
    out = np.zeros(6)
    for k in range(1, n + 1):
        pixels = components == k
        true = (pixels & near_line).sum() >= 0.5 * pixels.sum()
        confirmed = (pixels & near_cross).any()
        if true:
            out[0] += 1
            out[1] += confirmed
        else:
            out[2] += 1
            out[3] += confirmed
            out[4] += pixels.sum()
            out[5] += pixels.sum() * confirmed
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", required=True, help="saved head .pt in heads/")
    args = parser.parse_args()
    device = torch.device("cuda")
    saved = torch.load(HEADS / args.head, weights_only=False)
    units = [(e["layer"], e["unit"]) for e in saved["units"]]
    signs = [e["sign"] for e in saved["units"]]
    _, test_scans, _ = split_scans()
    data = SliceData(test_scans, units, signs)
    trainer = Trainer(data, device, saved["variant"])
    trainer.model.load_state_dict(saved["state"])
    trainer.token_mean, trainer.token_std = saved["mean"], saved["std"]
    threshold, min_size = saved["threshold"] if isinstance(saved["threshold"], (tuple, list)) else (saved["threshold"], 0)
    totals = {v: np.zeros(6) for v in VIEWS}
    cache_views = {}
    order = {scan: np.load(STORE / "slices" / f"{scan}.npz")["k"] for scan in test_scans}
    seen = {scan: 0 for scan in test_scans}
    for i in range(len(data)):
        scan = data.scan[i]
        if scan not in cache_views:
            cache_views.clear()  # one scan's volumes at a time
            loaded = ScanViews(scan_folder(scan))
            cache_views[scan] = (loaded, line_points(loaded))
        views, points_by_view = cache_views[scan]
        k = int(order[scan][seen[scan]])
        seen[scan] += 1
        view = VIEWS[data.view[i]]
        p = trainer.predict(data, i)
        prediction = drop_small_components(p > threshold, min_size)
        totals[view] += segment_confirmation(prediction, data.labels[i], cross_points(views, points_by_view, view, k))
    report = {}
    for view, t in {**totals, "all": sum(totals.values())}.items():
        report[view] = {"true_segments": t[0], "true_confirmed": t[1] / max(t[0], 1),
                        "false_segments": t[2], "false_confirmed": t[3] / max(t[2], 1),
                        "false_length_confirmed": t[5] / max(t[4], 1)}
        print(f"{view:9s} true segments {t[0]:5.0f} confirmed by another view {t[1] / max(t[0], 1):.2f} | "
              f"false segments {t[2]:5.0f} confirmed {t[3] / max(t[2], 1):.2f} (by length {t[5] / max(t[4], 1):.2f})")
    (HEADS / f"{args.head.removesuffix('.pt')}_cross_view.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
