"""Randomly initialised conv nets: how much fissure signal do untrained features carry?

Forty nets, one architecture, seeds 0-39, never trained (earlier runs used seeds 0-9 and 0-19):
  6 blocks of conv 3x3 -> ReLU, 64 channels each, strides 1, 2, 2, 2, 2, 1.
  At 1024 px input (like ConvNeXt / ResNet in store.py) the blocks run at
  1024, 512, 256, 128, 64, 64 px; the last block sees ~125 input px (~62 canvas
  px), so the blocks span thin-line to lobe-scale context.
  He-normal weights (fan-in, ReLU gain) and zero bias, so activations neither
  vanish nor explode with depth.
Input: the canvas lung window, grey copied to 3 channels, standardised with a
fixed mean / std of 0.5 / 0.25 (there is no training set statistic to use).
Every block output is a store layer "rand_conv{seed}/b{block}", resized to the
74 x 74 token grid like the pretrained conv nets.

Usage:
  python -m ...models.random_convs extract          # store columns at the rows of rows.py
  python -m ...models.random_convs maps --units J   # full maps for a unit list (head input)
"""

import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from l1_on_feat_vis.fissure_probe.data.rows import Rows
from l1_on_feat_vis.fissure_probe.store.store import MODELS_2D, StoreWriter, to_token_grid

WIDTH = 64
GREY_MEAN, GREY_STD = np.array([0.5] * 3), np.array([0.25] * 3)
# family: (strides per block, seeds). rand_deep keeps the same 4 early downsamplings, then 15 stride-1 blocks
# at 64 px, so the last block sees 513 input px (~256 canvas px, most of the lung); its seeds differ from
# rand_conv's so the two families share no weights.
# rand_deep16 is rand_deep with 16 channels per block (same seeds and strides), to isolate channel width.
FAMILIES = {
    "rand_conv": ((1, 2, 2, 2, 2, 1), range(40)),
    "rand_deep": ((1, 2, 2, 2, 2) + (1,) * 15, range(100, 110)),
    "rand_deep16": ((1, 2, 2, 2, 2) + (1,) * 15, range(100, 110)),
}
FAMILY_WIDTH = {"rand_deep16": 16}  # every other family: WIDTH
STRIDES, SEEDS = FAMILIES["rand_conv"]


def random_net(seed, strides=STRIDES, width=WIDTH):
    torch.manual_seed(seed)
    blocks, channels = [], 3
    for stride in strides:
        conv = nn.Conv2d(channels, width, 3, stride=stride, padding=1)
        nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(conv.bias)
        blocks.append(nn.Sequential(conv, nn.ReLU()))
        channels = width
    return nn.Sequential(*blocks)


class RandomConvNet:
    """Same interface as store.Conv: load(device), layers(x, tokens) -> {"b{i}": (tokens, width)}."""

    def __init__(self, seed, family="rand_conv"):
        self.name, self.seed, self.strides = f"{family}{seed}", seed, FAMILIES[family][0]
        self.width = FAMILY_WIDTH.get(family, WIDTH)
        self.mean, self.std, self.input_px = GREY_MEAN, GREY_STD, 1024

    def load(self, device):
        self.model = random_net(self.seed, self.strides, self.width).to(device).eval()

    def layers(self, x, tokens):
        out = {}
        for i, block in enumerate(self.model):
            x = block(x)
            out[f"b{i}"] = to_token_grid(x)[tokens]
        return out


def register(family="rand_conv"):
    """Make the nets visible to store.py and unit_maps.py, which look models up in MODELS_2D."""
    for f, (_, seeds) in FAMILIES.items():  # every family, so maps of a mixed unit list resolve
        for seed in seeds:
            MODELS_2D.setdefault(f"{f}{seed}", RandomConvNet(seed, f))
    return [f"{family}{seed}" for seed in FAMILIES[family][1]]


def layer_names(family="rand_conv"):
    strides, seeds = FAMILIES[family]
    return [f"{family}{seed}/b{i}" for seed in seeds for i in range(len(strides))]


def extract(family="rand_conv"):
    """Store columns for every random-net layer at every row (the pretrained store's done markers are not touched).

    Nets whose layers are already in the store are skipped, so raising SEEDS only extracts the new ones.
    """
    from l1_on_feat_vis.fissure_probe.store.store import ACTS
    scans = Rows().scans
    last = f"b{len(FAMILIES[family][0]) - 1}"
    names = [name for name in register(family) if not all((ACTS / name / last / f"{scan}.npy").exists() for scan in scans)]
    print(f"extracting {len(names)} random nets: {names}", flush=True)
    if not names:
        return
    writer = StoreWriter(names, [], torch.device("cuda"))
    for scan in scans:
        start = time.time()
        widths = writer.write_scan(scan)
        print(f"{scan}: {len(widths)} layers, {sum(widths.values())} units in {time.time() - start:.0f}s", flush=True)


def maps(units_path):
    from l1_on_feat_vis.fissure_probe.store.unit_maps import MapWriter, missing
    register()
    data = json.loads(open(units_path).read())
    entries = data["pool"] if isinstance(data, dict) else data
    units = missing([(e["layer"], e["unit"]) for e in entries])
    print(f"{len(units)} unit maps to compute", flush=True)
    if units:
        MapWriter(units, torch.device("cuda")).run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["extract", "maps"])
    parser.add_argument("--units")
    parser.add_argument("--family", default="rand_conv", choices=list(FAMILIES))
    args = parser.parse_args()
    torch.set_num_threads(8)
    extract(args.family) if args.command == "extract" else maps(args.units)


if __name__ == "__main__":
    main()
