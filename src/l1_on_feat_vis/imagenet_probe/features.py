"""Frozen, globally pooled features for the low-data probes.

Images: a fixed pool of 200 train images per class (seeded draw) plus the whole val split. Probes later
draw their 10 / 25 / 100 per-class training subsets from the pool, so nothing needs re-extracting.

Random nets reuse fissure_probe's random_net (conv 3x3 -> ReLU, He-normal fan-in, zero bias). Every net is
built at depth 40 with strides 1, 2, 2, 2, 2, then 1 forever:
  224 -> 224, 112, 56, 28, 14, 14, ... px.
torch.manual_seed fixes the init order, so the first D blocks of the depth-40 net of seed s are exactly
the depth-D net of seed s. Depth D is therefore read as a block prefix, and depths differ only by the
blocks they add (the shallow blocks are literally shared).
Residual control ("res"): why depth might fail, tested on the same weights.
  Plain deep ReLU stacks at init send every input towards one direction exponentially fast with depth;
  residual ones only polynomially (Poole et al. 2016; Yang & Schoenholz 2017, "Mean field residual networks").
  1. Reuse exactly the plain net's conv weights (same seed, same draws).
  2. After the 5 downsampling blocks, wire each block as a pre-activation residual step
     x <- (x + RES_SCALE * conv(relu(x))) / sqrt(1 + RES_SCALE^2).
     The division keeps the stream's scale in float16 range; the convs are bias-free and ReLU is positively
     homogeneous, so it only rescales later blocks by a constant, which the probes' standardisation removes.
  3. RES_SCALE is fixed (not 1/depth), so depth prefixes still nest.
  A res vs plain gap at equal depth is then the effect of signal propagation alone.
  A post-activation branch, x + relu(conv(x)), was tried first and dropped: it only ever adds non-negative
  values, so the stream piles up a shared positive mean and collapsed faster than plain (1.1 vs 1.7
  effective dims at block 40); mean-field residual theory assumes zero-mean branches.
Each block output gives two numbers per channel: its global mean and its global max over the image.
Stored per net as (images, 40 blocks, width, 2) float16.

Pretrained references, one vector per image at 224 px:
  pixels      the image average-pooled to 16 x 16 x 3 (a depth-0 control)
  dinov2_*    final layer CLS token and mean patch token (registers dropped)
  rad_dino    the same, for an out-of-domain (chest X-ray) pretraining
  resnet50    timm resnet50.a1_in1k pre-logits; it was trained on ImageNet-1k labels, so it is a ceiling
Usage:
  python -m l1_on_feat_vis.imagenet_probe.features random --width 64 --seeds 0-119 [--arch res]
  python -m l1_on_feat_vis.imagenet_probe.features refs
"""

import argparse
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F
from transformers import AutoModel

from l1_on_feat_vis.fissure_probe.models.random_convs import random_net
from l1_on_feat_vis.fissure_probe.store.store import IMAGENET_MEAN, IMAGENET_STD, RAD_DINO_MEAN, RAD_DINO_STD
from l1_on_feat_vis.imagenet_probe.data import ROOT, Dataset

DATASETS = ("nette", "woof")
POOL_PER_CLASS = 200
MAX_DEPTH = 40
DOWNSAMPLING_BLOCKS = 5
RES_SCALE = 0.5
BATCH = 250
FEATURES = ROOT / "features"
VITS = {
    "dinov2_s": ("facebook/dinov2-with-registers-small", IMAGENET_MEAN, IMAGENET_STD),
    "dinov2_b": ("facebook/dinov2-with-registers-base", IMAGENET_MEAN, IMAGENET_STD),
    "dinov2_l": ("facebook/dinov2-with-registers-large", IMAGENET_MEAN, IMAGENET_STD),
    "rad_dino": ("microsoft/rad-dino", RAD_DINO_MEAN, RAD_DINO_STD),
}


def depth_strides(depth):
    return ((1, 2, 2, 2, 2) + (1,) * MAX_DEPTH)[:depth]


def pool_and_val(labels, is_val, seed=0):
    """Image indices: POOL_PER_CLASS seeded train images per class, then every val image."""
    rng = np.random.default_rng(seed)
    pool = [rng.choice(np.flatnonzero((labels == c) & ~is_val), POOL_PER_CLASS, replace=False)
            for c in np.unique(labels)]
    return np.concatenate([np.sort(np.concatenate(pool)), np.flatnonzero(is_val)])


def index_path(name):
    return FEATURES / name / "index.npy"


def image_index(name):
    """Saved once so every feature file and probe shares the same image order."""
    path = index_path(name)
    if not path.exists():
        data = Dataset(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, pool_and_val(data.labels, data.is_val))
    return np.load(path)


def image_batches(name, mean, std, device):
    data, index = Dataset(name), image_index(name)
    mean = torch.tensor(mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    for start in range(0, len(index), BATCH):
        x = torch.from_numpy(np.ascontiguousarray(data.images[index[start:start + BATCH]])).to(device)
        yield (x.permute(0, 3, 1, 2).float() / 255 - mean) / std


def residual_step(block, x):
    conv = block[0]
    return (x + RES_SCALE * conv(torch.relu(x))) / (1 + RES_SCALE ** 2) ** 0.5


def pooled_blocks(net, x, residual=False):
    out = []
    for i, block in enumerate(net):
        x = residual_step(block, x) if residual and i >= DOWNSAMPLING_BLOCKS else block(x)
        out.append(torch.stack([x.mean((2, 3)), x.amax((2, 3))], -1))
    return torch.stack(out, 1)  # (batch, blocks, width, 2)


def random_path(name, width, seed, arch="plain"):
    family = f"rand_w{width}" if arch == "plain" else f"rand_{arch}_w{width}"
    return FEATURES / name / family / f"seed{seed}.npy"


@torch.no_grad()
def extract_random(width, seeds, device, arch="plain"):
    for name in DATASETS:
        for seed in seeds:
            path = random_path(name, width, seed, arch)
            if path.exists():
                continue
            start = time.time()
            net = random_net(seed, depth_strides(MAX_DEPTH), width).to(device).eval()
            feats = [pooled_blocks(net, x, arch == "res").half().cpu() for x in image_batches(name, IMAGENET_MEAN, IMAGENET_STD, device)]
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, torch.cat(feats).numpy())
            print(f"{name} {arch} w{width} seed {seed}: {time.time() - start:.1f}s", flush=True)


def ref_path(name, model):
    return FEATURES / name / "refs" / f"{model}.npy"


def pixels(x):
    return F.adaptive_avg_pool2d(x, 16).flatten(1)


def vit_features(model):
    skip = 1 + getattr(model.config, "num_register_tokens", 0)

    def run(x):
        tokens = model(pixel_values=x).last_hidden_state
        return torch.cat([tokens[:, 0], tokens[:, skip:].mean(1)], 1)
    return run


def resnet_features(model):
    return lambda x: model.forward_head(model.forward_features(x), pre_logits=True)


def ref_extractors(device):
    yield "pixels", IMAGENET_MEAN, IMAGENET_STD, lambda: pixels
    for model, (repo, mean, std) in VITS.items():
        yield model, mean, std, lambda repo=repo: vit_features(AutoModel.from_pretrained(repo, use_safetensors=True).to(device).eval())
    yield "resnet50", IMAGENET_MEAN, IMAGENET_STD, lambda: resnet_features(timm.create_model("resnet50.a1_in1k", pretrained=True).to(device).eval())


@torch.no_grad()
def extract_refs(device):
    for model, mean, std, build in ref_extractors(device):
        run = None
        for name in DATASETS:
            path = ref_path(name, model)
            if path.exists():
                continue
            run = run or build()
            feats = [run(x).float().cpu() for x in image_batches(name, mean, std, device)]
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, torch.cat(feats).numpy())
            print(f"{name} {model}: {feats[0].shape[1]} features", flush=True)


def seed_range(text):
    first, last = map(int, text.split("-"))
    return range(first, last + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["random", "refs"])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seeds", type=seed_range, default=range(120))
    parser.add_argument("--arch", default="plain", choices=["plain", "res"])
    args = parser.parse_args()
    device = torch.device("cuda")
    extract_random(args.width, args.seeds, device, args.arch) if args.command == "random" else extract_refs(device)


if __name__ == "__main__":
    main()
