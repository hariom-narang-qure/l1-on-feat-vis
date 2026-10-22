"""Sanity figures for the canvas slice cache (views.py): CT with every label stratum overlaid."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from l1_on_feat_vis.fissure_probe.views import BITS, STORE, VIEWS

COLOURS = {"ring": (60, 120, 255), "pleura": (255, 200, 0), "vessel": (200, 0, 200), "positive": (255, 40, 40)}
WINDOW_HU = (-1000.0, -200.0)


def grey(hu):
    return ((np.clip(hu, *WINDOW_HU) - WINDOW_HU[0]) / (WINDOW_HU[1] - WINDOW_HU[0]) * 255).astype(np.uint8)


def overlay(hu, labels):
    base = np.repeat(grey(hu)[..., None], 3, -1).astype(np.float32)
    lung = (labels & BITS["lung"]) > 0
    base[~lung] *= 0.55
    for name, colour in COLOURS.items():
        mask = (labels & BITS[name]) > 0
        alpha = 1.0 if name == "positive" else 0.45
        base[mask] = (1 - alpha) * base[mask] + alpha * np.array(colour)
    return base.astype(np.uint8)


def panel(hu, labels, title, tile):
    pair = np.concatenate([np.repeat(grey(hu)[..., None], 3, -1), overlay(hu, labels)], axis=1)
    image = Image.fromarray(pair).resize((2 * tile, tile))
    ImageDraw.Draw(image).text((6, 4), title, fill=(255, 255, 0))
    return image


def sheet(picks, out_path, tile=360):
    """picks: list of (scan, slice index in its cache). Two columns per pick: CT, CT + labels."""
    columns = 2
    rows = int(np.ceil(len(picks) / columns))
    out = Image.new("RGB", (columns * 2 * tile, rows * tile), "black")
    caches = {}
    for i, (scan, index) in enumerate(picks):
        cache = caches.setdefault(scan, np.load(STORE / "slices" / f"{scan}.npz"))
        title = f"{scan} {VIEWS[cache['view'][index]]} k={cache['k'][index]}"
        out.paste(panel(cache["hu"][index], cache["labels"][index], title, tile), ((i % columns) * 2 * tile, (i // columns) * tile))
    out.save(out_path)


def default_picks(scans, per_view=1, seed=0):
    rng = np.random.default_rng(seed)
    picks = []
    for scan in scans:
        cache = np.load(STORE / "slices" / f"{scan}.npz")
        line_px = ((cache["labels"] & BITS["line"]) > 0).sum((1, 2))
        for v in range(3):
            candidates = np.flatnonzero((cache["view"] == v) & (line_px >= np.percentile(line_px, 50)))
            candidates = candidates if len(candidates) else np.flatnonzero(cache["view"] == v)
            picks += [(scan, int(i)) for i in rng.choice(candidates, min(per_view, len(candidates)), replace=False)]
    return picks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scans", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sheet(default_picks(args.scans), args.out)


if __name__ == "__main__":
    main()
