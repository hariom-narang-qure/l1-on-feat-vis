"""Sample units of many 2D networks on axial CT slices, in the features.py row format.

Same protocol as dino_features.py, generalised to any network:
  - Slices: 6 among slices with >= 200 sheet voxels, 4 random lung slices.
  - Input: lung window [-1000, -200] HU -> [0, 1], copied to RGB, the model's
    own normalisation, resized to the model's input size.
  - Every probed block's output is resampled to one shared 74 x 74 grid
    (bilinear, antialiased when shrinking), so units from different networks
    describe the same ~6.9 px cells and can be pooled into one head later.
    Finer conv stages lose detail this way; the head gets it back from its CT skip.
  - Labels and sampled tokens exactly as in dino_features.py.
  - Each network has a random-init twin with the same architecture.

Block names are "<model>/<block>", e.g. "dinov2_l/res16", "convnext_b/stage2".
"""

import argparse
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from l1_on_feat_vis.fissure_probe.data import LoadedScan, list_scans
from l1_on_feat_vis.fissure_probe.dino_features import (EXTRA_POSITIVES, GRID, IMAGENET_MEAN, IMAGENET_STD, LABELS,
                                                       UNIFORM_TOKENS, WINDOW_HU, ridge_2d, to_grid)

RAD_DINO_MEAN, RAD_DINO_STD = np.array([0.5307, 0.5307, 0.5307]), np.array([0.2583, 0.2583, 0.2583])


class VitSpec:
    """A Hugging Face DINOv2-style ViT; blocks are residual-stream layers."""

    def __init__(self, name, repo, layers, input_px, mean, std):
        self.name, self.repo, self.layers, self.input_px = name, repo, layers, input_px
        self.mean, self.std = mean, std

    def build(self, pretrained):
        if pretrained:
            return AutoModel.from_pretrained(self.repo, use_safetensors=True)
        torch.manual_seed(0)
        return AutoModel.from_config(AutoConfig.from_pretrained(self.repo))

    def blocks(self, model, x):
        out = model(pixel_values=x, output_hidden_states=True)
        skip = 1 + getattr(model.config, "num_register_tokens", 0)
        side = self.input_px // model.config.patch_size
        return {f"{self.name}/res{l}": out.hidden_states[l + 1][0, skip:].T.reshape(-1, side, side) for l in self.layers}


class TimmSpec:
    """A timm conv net with features_only; blocks are stage outputs."""

    def __init__(self, name, timm_name, stages, input_px):
        self.name, self.timm_name, self.stages, self.input_px = name, timm_name, stages, input_px
        self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD

    def build(self, pretrained):
        torch.manual_seed(0)
        return timm.create_model(self.timm_name, pretrained=pretrained, features_only=True, out_indices=self.stages)

    def blocks(self, model, x):
        return {f"{self.name}/stage{s}": f[0] for s, f in zip(self.stages, model(x))}


MODELS = {
    "dinov2_s": VitSpec("dinov2_s", "facebook/dinov2-with-registers-small", (5, 8, 11), 1036, IMAGENET_MEAN, IMAGENET_STD),
    "dinov2_b": VitSpec("dinov2_b", "facebook/dinov2-with-registers-base", (8, 11), 1036, IMAGENET_MEAN, IMAGENET_STD),
    "dinov2_l": VitSpec("dinov2_l", "facebook/dinov2-with-registers-large", (8, 12, 16, 20, 23), 1036, IMAGENET_MEAN, IMAGENET_STD),
    "rad_dino": VitSpec("rad_dino", "microsoft/rad-dino", (5, 8, 11), 1036, RAD_DINO_MEAN, RAD_DINO_STD),
    "convnext_b": TimmSpec("convnext_b", "convnext_base.fb_in22k_ft_in1k", (1, 2, 3), 1024),
    "resnet50": TimmSpec("resnet50", "resnet50.a1_in1k", (2, 3, 4), 1024),
}


def to_shared_grid(feature_map):
    """(C, h, w) -> (GRID * GRID, C) on the shared 74 x 74 grid."""
    resized = F.interpolate(feature_map[None].float(), size=(GRID, GRID), mode="bilinear", align_corners=False,
                            antialias=feature_map.shape[-1] > GRID)
    return resized[0].flatten(1).T


class MultiNetwork:
    def __init__(self, names, device):
        self.device = device
        self.specs = [MODELS[name] for name in names]
        self.models = {(spec.name, kind): spec.build(kind == "pretrained").to(device).eval()
                       for spec in self.specs for kind in ("pretrained", "random")}

    def model_input(self, grey, spec):
        rgb = (np.repeat(grey[..., None], 3, -1) - spec.mean) / spec.std
        x = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32))[None].to(self.device)
        return F.interpolate(x, size=(spec.input_px, spec.input_px), mode="bilinear", align_corners=False)

    def features(self, grey):
        """{block: {"pretrained" | "random": (GRID*GRID, C)}} for one slice."""
        out = {}
        for spec in self.specs:
            x = self.model_input(grey, spec)
            for kind in ("pretrained", "random"):
                with torch.no_grad():
                    blocks = spec.blocks(self.models[(spec.name, kind)], x)
                for block, feature_map in blocks.items():
                    out.setdefault(block, {})[kind] = to_shared_grid(feature_map)
        return out


class SliceSampler:
    def __init__(self, names, device, seed=0):
        self.networks = MultiNetwork(names, device)
        self.rng = np.random.default_rng(seed)

    def pick_slices(self, scan):
        per_slice_sheet = (scan.fissure & scan.lung).sum(axis=(0, 1))
        lung_slices = np.flatnonzero(scan.lung.sum(axis=(0, 1)) > 2000)
        fissure_slices = np.flatnonzero(per_slice_sheet >= 200)
        picked = list(self.rng.choice(fissure_slices, min(6, len(fissure_slices)), replace=False))
        return picked + list(self.rng.choice(lung_slices, 4, replace=False))

    def pick_tokens(self, lung, fissure):
        lung_tokens, fissure_tokens = np.flatnonzero(lung), np.flatnonzero(lung & fissure)
        uniform = self.rng.choice(lung_tokens, min(UNIFORM_TOKENS, len(lung_tokens)), replace=False)
        extra = self.rng.choice(fissure_tokens, min(EXTRA_POSITIVES, len(fissure_tokens)), replace=False)
        return np.concatenate([uniform, extra]), np.concatenate([np.ones(len(uniform), bool), np.zeros(len(extra), bool)])

    def sample_slice(self, scan, z, slice_index, out):
        hu = scan.hu[:, :, z]
        grey = (np.clip(hu, *WINDOW_HU) - WINDOW_HU[0]) / (WINDOW_HU[1] - WINDOW_HU[0])
        lung = to_grid(scan.lung[:, :, z], "avg") > 0.5
        labels = {label: to_grid(getattr(scan, label)[:, :, z], "max") > 0 for label in LABELS if label != "ring"}
        labels["ring"] = (to_grid(scan.ring[:, :, z], "avg") > 0.5) & ~labels["fissure"]
        tokens, uniform = self.pick_tokens(lung.ravel(), labels["fissure"].ravel())
        if len(tokens) == 0:
            return
        index = torch.from_numpy(tokens).to(self.networks.device)
        intensity, ridge = to_grid(grey, "avg").ravel()[tokens], to_grid(ridge_2d(grey), "max").ravel()[tokens]
        for block, by_kind in self.networks.features(grey).items():
            rows = out.setdefault(block, {key: [] for key in ("uniform", "patch", "intensity", "sheetness", *LABELS, "pretrained", "random")})
            rows["uniform"].append(uniform)
            rows["patch"].append(np.full(len(tokens), slice_index, np.int16))
            rows["intensity"].append(intensity)
            rows["sheetness"].append(ridge)
            for label in LABELS:
                rows[label].append(labels[label].ravel()[tokens])
            for kind, features in by_kind.items():
                rows[kind].append(features[index].half().cpu().numpy())

    def sample_scan(self, record, out_dir):
        scan = LoadedScan(record)
        out = {}
        for index, z in enumerate(self.pick_slices(scan)):
            self.sample_slice(scan, int(z), index, out)
        arrays = {f"{block}/{key}": np.concatenate(values) for block, rows in out.items() for key, values in rows.items()}
        np.savez(out_dir / f"{record.short}.npz", **arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--scans", nargs="*", help="short ids; default all")
    parser.add_argument("--out", type=Path, default=Path("outputs/fissure_probe/slice_features"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.set_num_threads(8)
    args.out.mkdir(parents=True, exist_ok=True)
    sampler = SliceSampler(args.models, torch.device(args.device))
    for record in list_scans():
        if (args.scans and record.short not in args.scans) or (args.out / f"{record.short}.npz").exists():
            continue
        start = time.time()
        sampler.sample_scan(record, args.out)
        print(f"{record.short} {record.cohort} done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
