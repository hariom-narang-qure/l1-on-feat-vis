"""Randomly initialised conv nets: how much fissure signal do untrained features carry?

Ten nets, one architecture, seeds 0-9, never trained:
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

SEEDS = range(10)
WIDTH, STRIDES = 64, (1, 2, 2, 2, 2, 1)
GREY_MEAN, GREY_STD = np.array([0.5] * 3), np.array([0.25] * 3)


def random_net(seed):
    torch.manual_seed(seed)
    blocks, channels = [], 3
    for stride in STRIDES:
        conv = nn.Conv2d(channels, WIDTH, 3, stride=stride, padding=1)
        nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(conv.bias)
        blocks.append(nn.Sequential(conv, nn.ReLU()))
        channels = WIDTH
    return nn.Sequential(*blocks)


class RandomConvNet:
    """Same interface as store.Conv: load(device), layers(x, tokens) -> {"b{i}": (tokens, 64)}."""

    def __init__(self, seed):
        self.name, self.seed = f"rand_conv{seed}", seed
        self.mean, self.std, self.input_px = GREY_MEAN, GREY_STD, 1024

    def load(self, device):
        self.model = random_net(self.seed).to(device).eval()

    def layers(self, x, tokens):
        out = {}
        for i, block in enumerate(self.model):
            x = block(x)
            out[f"b{i}"] = to_token_grid(x)[tokens]
        return out


def register():
    """Make the nets visible to store.py and unit_maps.py, which look models up in MODELS_2D."""
    for seed in SEEDS:
        MODELS_2D.setdefault(f"rand_conv{seed}", RandomConvNet(seed))
    return [f"rand_conv{seed}" for seed in SEEDS]


def layer_names():
    return [f"rand_conv{seed}/b{i}" for seed in SEEDS for i in range(len(STRIDES))]


def extract():
    """Store columns for every random-net layer at every row (the pretrained store's done markers are not touched)."""
    names = register()
    writer = StoreWriter(names, [], torch.device("cuda"))
    for scan in Rows().scans:
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
    args = parser.parse_args()
    torch.set_num_threads(8)
    extract() if args.command == "extract" else maps(args.units)


if __name__ == "__main__":
    main()
