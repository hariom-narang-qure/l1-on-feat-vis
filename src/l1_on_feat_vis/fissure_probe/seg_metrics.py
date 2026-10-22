"""Segmentation metrics for 1 px fissure lines, per slice, summed per view.

Pixel metrics punish a correct line drawn one pixel off as much as a missed
fissure, so the headline numbers are length-based, on 1 px skeletons
(Wiedemann et al. 1998, "Empirical evaluation of automatically extracted road
axes"; Shit et al. 2021, "clDice"):
  - completeness = GT line length with a predicted skeleton within TOL px / GT length   (missed segmentation)
  - correctness  = predicted skeleton length with GT line within TOL px / predicted length (false segmentation)
  - tolerant Dice = harmonic mean of the two (headline)
  - clDice: skeleton of each inside the other's mask (GT mask = positive band)
Object level, on 8-connected components of each slice:
  - a GT segment is missed if < 50% of its pixels have a prediction within TOL px
  - a predicted segment is false if < 50% of its pixels have GT within TOL px;
    false segments are typed by where most of their pixels lie:
    vessel (within 2 px of a vessel pixel), pleura band, or other
Also exact pixel Dice against the positive band, and pixel AP (histogram, 1000 bins)
over lung pixels. Everything is restricted to the lung (grown lobes).
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from l1_on_feat_vis.fissure_probe.views import BITS, VIEWS

TOL = 2
EIGHT = np.ones((3, 3), bool)
BINS = 1000
COUNTS = ("gt_len", "gt_found", "pred_len", "pred_true", "cl_pred_in_gt", "cl_gt_in_pred",
          "px_inter", "px_pred", "px_gt", "gt_segments", "missed_segments", "missed_len",
          "pred_segments", "false_segments", "false_len", "false_vessel", "false_pleura", "false_other")


def near(mask, tol=TOL):
    if not mask.any():
        return np.zeros_like(mask)
    return ndimage.distance_transform_edt(~mask) <= tol


def segment_hits(mask, reference_near):
    """Per 8-connected component: (pixel count, fraction of pixels in reference_near)."""
    labels, n = ndimage.label(mask, EIGHT)
    if n == 0:
        return np.zeros(0), np.zeros(0), labels
    index = np.arange(1, n + 1)
    size = ndimage.sum(np.ones_like(mask, float), labels, index)
    hit = ndimage.sum(reference_near.astype(float), labels, index)
    return size, hit / size, labels


def drop_small_components(prediction, min_size):
    """Remove 8-connected predicted components smaller than min_size pixels (0 keeps everything)."""
    if min_size <= 0 or not prediction.any():
        return prediction
    labels, n = ndimage.label(prediction, EIGHT)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def slice_counts(prediction, labels):
    """Counts of one slice; prediction is a boolean map."""
    lung = (labels & BITS["lung"]) > 0
    line = (labels & BITS["line"]) > 0
    positive = (labels & BITS["positive"]) > 0
    prediction = prediction & lung
    skeleton = skeletonize(prediction) if prediction.any() else prediction
    near_gt, near_pred = near(line), near(skeleton)
    c = dict.fromkeys(COUNTS, 0.0)
    c["gt_len"], c["gt_found"] = line.sum(), (line & near_pred).sum()
    c["pred_len"], c["pred_true"] = skeleton.sum(), (skeleton & near_gt).sum()
    c["cl_pred_in_gt"], c["cl_gt_in_pred"] = (skeleton & positive).sum(), (line & prediction).sum()
    c["px_inter"], c["px_pred"], c["px_gt"] = (prediction & positive).sum(), prediction.sum(), positive.sum()
    size, found, _ = segment_hits(line, near_pred)
    missed = found < 0.5
    c["gt_segments"], c["missed_segments"], c["missed_len"] = len(size), missed.sum(), size[missed].sum()
    size, true, components = segment_hits(skeleton, near_gt)
    false = np.flatnonzero(true < 0.5)
    c["pred_segments"], c["false_segments"], c["false_len"] = len(size), len(false), size[false].sum()
    vessel = near((labels & BITS["vessel"]) > 0)
    pleura = (labels & BITS["pleura"]) > 0
    for k in false + 1:
        pixels = components == k
        kind = np.argmax([(pixels & vessel).sum(), (pixels & pleura).sum(), (pixels & ~vessel & ~pleura).sum()])
        c[("false_vessel", "false_pleura", "false_other")[kind]] += 1
    return c


class Scoreboard:
    """Accumulates counts and AP histograms per view; summarise() gives the metrics."""

    def __init__(self):
        self.counts = {v: dict.fromkeys(COUNTS, 0.0) for v in VIEWS}
        self.hist = {v: np.zeros((2, BINS)) for v in VIEWS}

    def add(self, probability, labels, view, threshold, min_size=0):
        prediction = drop_small_components(probability > threshold, min_size)
        for key, value in slice_counts(prediction, labels).items():
            self.counts[view][key] += float(value)
        lung = (labels & BITS["lung"]) > 0
        positive = (labels & BITS["positive"]) > 0
        bins = np.minimum((probability[lung] * BINS).astype(int), BINS - 1)
        self.hist[view][1] += np.bincount(bins[positive[lung]], minlength=BINS)
        self.hist[view][0] += np.bincount(bins[~positive[lung]], minlength=BINS)

    @staticmethod
    def metrics(c, hist):
        ratio = lambda a, b: a / b if b else float("nan")
        completeness, correctness = ratio(c["gt_found"], c["gt_len"]), ratio(c["pred_true"], c["pred_len"])
        t_prec, t_sens = ratio(c["cl_pred_in_gt"], c["pred_len"]), ratio(c["cl_gt_in_pred"], c["gt_len"])
        harmonic = lambda a, b: 2 * a * b / (a + b) if a + b > 0 else 0.0
        negative, positive = hist[0][::-1].cumsum(), hist[1][::-1].cumsum()  # descending thresholds
        precision = positive / np.maximum(positive + negative, 1)
        new_positives = np.diff(np.concatenate([[0], positive]))
        return {
            "tolerant_dice": harmonic(completeness, correctness), "completeness": completeness, "correctness": correctness,
            "cldice": harmonic(t_prec, t_sens), "pixel_dice": ratio(2 * c["px_inter"], c["px_pred"] + c["px_gt"]),
            "pixel_ap": float((precision * new_positives).sum() / max(positive[-1], 1)),
            "missed_segment_rate": ratio(c["missed_segments"], c["gt_segments"]), "missed_len_rate": ratio(c["missed_len"], c["gt_len"]),
            "false_segment_rate": ratio(c["false_segments"], c["pred_segments"]), "false_len_rate": ratio(c["false_len"], c["pred_len"]),
            "false_vessel": c["false_vessel"], "false_pleura": c["false_pleura"], "false_other": c["false_other"],
            "gt_segments": c["gt_segments"], "pred_segments": c["pred_segments"],
        }

    def summarise(self):
        out = {v: self.metrics(self.counts[v], self.hist[v]) for v in VIEWS}
        total = {key: sum(self.counts[v][key] for v in VIEWS) for key in COUNTS}
        out["all"] = self.metrics(total, sum(self.hist[v] for v in VIEWS))
        return out
