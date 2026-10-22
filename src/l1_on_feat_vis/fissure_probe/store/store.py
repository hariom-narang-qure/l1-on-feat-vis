"""Activation store: every unit of every layer of the 7 pretrained networks, at the rows of rows.py.

Layers (one store column block each):
  - DINOv2-S / B / L (with registers), RAD-DINO: the residual stream after
    every layer ("res{l}"); for S, B and RAD-DINO also the MLP hidden units
    after GELU ("mlp{l}"). Input 1036 px, so the patch grid is the 74 x 74 token grid.
  - ConvNeXt-B: every block output ("s{stage}b{block}"); ResNet-50: every
    bottleneck output ("l{layer}b{block}"). Input 1024 px; maps are resized to
    74 x 74 (bilinear, antialiased when shrinking), as in slice_features.py.
  - TotalSegmentator organs nnU-Net (3D, 1.5 mm): every conv block output
    ("enc{stage}c{conv}", "dec{stage}c{conv}") and the final organ logits
    ("seg"). One forward over the lung crop per scan, then each map is sampled
    (trilinear) at the voxel centre of every row's token, for all three views.
2D inputs: the canvas (views.py), lung window [-1000, -200] HU, grey to RGB,
the model's own normalisation, bilinear resize. bf16 autocast.

Storage: per (model, layer, scan) an int8 matrix (rows of that scan x units)
with a per-unit min / max (asymmetric, 255 levels). Per-scan scales let scans
be written one at a time; rank-based scores (AUROC, AP) barely notice the rounding.
Layout: acts/<model>/<layer>/<scan>.npy and <scan>.minmax.npy.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from transformers import AutoModel

from l1_on_feat_vis.fissure_probe.models.nnunet_features import NETWORKS, NnUNet, native_to_network, sample_stage
from l1_on_feat_vis.fissure_probe.data.rows import GRID, Rows
from l1_on_feat_vis.fissure_probe.sheet_probing.slice_features import RAD_DINO_MEAN, RAD_DINO_STD
from l1_on_feat_vis.fissure_probe.sheet_probing.dino_features import IMAGENET_MEAN, IMAGENET_STD
from l1_on_feat_vis.fissure_probe.data.views import CANVAS, STORE, VIEWS, ScanViews, SCANS_ROOT

WINDOW_HU = (-1000.0, -200.0)
ACTS = STORE / "acts"
CROP_MARGIN_MM = 15.0


def canvas_grey(hu):
    return (np.clip(hu, *WINDOW_HU) - WINDOW_HU[0]) / (WINDOW_HU[1] - WINDOW_HU[0])


def model_input(grey, mean, std, input_px, device):
    rgb = (np.repeat(grey[..., None], 3, -1) - mean) / std
    x = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32))[None].to(device)
    return F.interpolate(x, size=(input_px, input_px), mode="bilinear", align_corners=False)


def to_token_grid(feature_map):
    """(1, C, h, w) -> (GRID * GRID, C)."""
    resized = F.interpolate(feature_map.float(), size=(GRID, GRID), mode="bilinear", align_corners=False,
                            antialias=feature_map.shape[-1] > GRID)
    return resized[0].flatten(1).T


class Vit:
    def __init__(self, name, repo, mean, std, mlp):
        self.name, self.repo, self.mean, self.std, self.with_mlp = name, repo, mean, std, mlp
        self.input_px = 1036

    def load(self, device):
        self.model = AutoModel.from_pretrained(self.repo, use_safetensors=True).to(device).eval()
        self.skip = 1 + getattr(self.model.config, "num_register_tokens", 0)
        self.mlp_out = {}
        if self.with_mlp:
            for l, layer in enumerate(self.model.encoder.layer):
                layer.mlp.activation_fn.register_forward_hook(lambda m, i, o, l=l: self.mlp_out.__setitem__(l, o))

    def layers(self, x, tokens):
        """{layer: (len(tokens), C)} on the GPU."""
        self.mlp_out.clear()
        hidden = self.model(pixel_values=x, output_hidden_states=True).hidden_states
        out = {f"res{l}": h[0, self.skip:][tokens] for l, h in enumerate(hidden[1:])}
        out.update({f"mlp{l}": o[0, self.skip:][tokens] for l, o in self.mlp_out.items()})
        return out


class Conv:
    def __init__(self, name, timm_name, block_lists):
        self.name, self.timm_name, self.block_lists = name, timm_name, block_lists
        self.mean, self.std, self.input_px = IMAGENET_MEAN, IMAGENET_STD, 1024

    def load(self, device):
        self.model = timm.create_model(self.timm_name, pretrained=True).to(device).eval()
        self.taps = {}
        for prefix, blocks in self.block_lists(self.model):
            for b, block in enumerate(blocks):
                block.register_forward_hook(lambda m, i, o, key=f"{prefix}b{b}": self.taps.__setitem__(key, o))

    def layers(self, x, tokens):
        self.taps.clear()
        self.model.forward_features(x)
        return {key: to_token_grid(o)[tokens] for key, o in self.taps.items()}


def convnext_blocks(model):
    return [(f"s{i}", stage.blocks) for i, stage in enumerate(model.stages)]


def resnet_blocks(model):
    return [(f"l{i}", getattr(model, f"layer{i}")) for i in range(1, 5)]


MODELS_2D = {
    "dinov2_s": Vit("dinov2_s", "facebook/dinov2-with-registers-small", IMAGENET_MEAN, IMAGENET_STD, mlp=True),
    "dinov2_b": Vit("dinov2_b", "facebook/dinov2-with-registers-base", IMAGENET_MEAN, IMAGENET_STD, mlp=True),
    "dinov2_l": Vit("dinov2_l", "facebook/dinov2-with-registers-large", IMAGENET_MEAN, IMAGENET_STD, mlp=False),
    "rad_dino": Vit("rad_dino", "microsoft/rad-dino", RAD_DINO_MEAN, RAD_DINO_STD, mlp=True),
    "convnext_b": Conv("convnext_b", "convnext_base.fb_in22k_ft_in1k", convnext_blocks),
    "resnet50": Conv("resnet50", "resnet50.a1_in1k", resnet_blocks),
}
MODELS_3D = ("ts_organs",)


def token_centres(tokens):
    """Canvas (row, col) pixel index of each token's centre."""
    i, j = np.divmod(tokens.astype(int), GRID)
    step = CANVAS / GRID
    return (i + 0.5) * step - 0.5, (j + 0.5) * step - 0.5


class Taps3D:
    """One nnU-Net with every conv block (and the logits) sampled at given voxels during its forward pass."""

    def __init__(self, name, device):
        self.net = NnUNet(name, NETWORKS[name], device, stage_hooks=False)
        self.coords, self.padded, self.out, self.wanted = None, None, {}, None
        network = self.net.network
        for s, stage in enumerate(network.encoder.stages):
            for c, conv in enumerate(stage[0].convs):
                conv.register_forward_hook(self.sampler(f"enc{s}c{c}"))
        for s, stage in enumerate(network.decoder.stages):
            for c, conv in enumerate(stage.convs):
                conv.register_forward_hook(self.sampler(f"dec{s}c{c}"))
        network.decoder.seg_layers[-1].register_forward_hook(self.sampler("seg"))

    def sampler(self, key):
        def hook(module, inputs, output):
            if self.wanted is None:
                self.out[key] = sample_stage(output, self.coords, None, self.padded).half().cpu()
            elif key in self.wanted:
                self.out[key] = sample_stage(output[:, self.wanted[key]], self.coords, None, self.padded).half().cpu()
        return hook

    def layers(self, hu_volume, spacing, lobes, voxels, wanted=None):
        """{layer: (len(voxels), C)}; `wanted` = {layer: [units]} samples only those units (columns in that order)."""
        self.wanted = wanted
        lo, hi = lung_crop(lobes, spacing)
        crop = hu_volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
        x, zoom = self.net.preprocess(crop, spacing)
        self.coords = native_to_network(voxels, lo, zoom, self.net.plans.transpose_forward)
        self.padded = x.shape[2:]
        self.out = {}
        self.net.forward(x)
        return self.out


def lung_crop(lobes, spacing_xyz):
    xs, ys, zs = np.nonzero(lobes > 0)
    margin = np.ceil(CROP_MARGIN_MM / np.array(spacing_xyz)).astype(int)
    lo = np.maximum(np.array([xs.min(), ys.min(), zs.min()]) - margin, 0)
    hi = np.minimum(np.array([xs.max(), ys.max(), zs.max()]) + margin + 1, lobes.shape)
    return lo, hi


def quantise(x):
    """(rows, C) float -> int8 and a (2, C) min / max."""
    lo, hi = x.min(0), x.max(0)
    scale = np.where(hi > lo, hi - lo, 1.0)
    q = np.round((x - lo) / scale * 254) - 127
    return q.astype(np.int8), np.stack([lo, hi]).astype(np.float32)


def dequantise(q, minmax):
    lo, hi = minmax
    return (q.astype(np.float32) + 127) / 254 * np.where(hi > lo, hi - lo, 1.0) + lo


class StoreWriter:
    def __init__(self, models_2d, models_3d, device):
        self.device = device
        self.models_2d = [MODELS_2D[m] for m in models_2d]
        for spec in self.models_2d:
            spec.load(device)
        self.models_3d = {m: Taps3D(m, device) for m in models_3d}
        self.rows = Rows()

    def scan_rows(self, scan):
        index = self.rows.of_scan(self.rows.scans.index(scan))
        return index, self.rows.slice[index], self.rows.token[index]

    def extract_2d(self, scan, slices, tokens):
        cache = np.load(STORE / "slices" / f"{scan}.npz")
        out = {}
        for s in np.unique(slices):
            these = torch.from_numpy(tokens[slices == s].astype(np.int64)).to(self.device)
            grey = canvas_grey(cache["hu"][s])
            for spec in self.models_2d:
                x = model_input(grey, spec.mean, spec.std, spec.input_px, self.device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    layers = spec.layers(x, these)
                for layer, values in layers.items():
                    out.setdefault(f"{spec.name}/{layer}", []).append(values.float().half().cpu().numpy())
        return {key: np.concatenate(parts) for key, parts in out.items()}

    def extract_3d(self, scan, slices, tokens):
        if not self.models_3d:
            return {}
        cache = np.load(STORE / "slices" / f"{scan}.npz")
        views = ScanViews(scan_folder(scan))
        p, q = token_centres(tokens)
        voxels = np.zeros((len(tokens), 3))
        for s in np.unique(slices):
            rows = slices == s
            geometry = views.geometry[VIEWS[cache["view"][s]]]
            voxels[rows] = geometry.canvas_to_voxel(p[rows], q[rows], int(cache["k"][s]))
        out = {}
        for name, taps in self.models_3d.items():
            for layer, values in taps.layers(views.hu, views.spacing, views.lobes, voxels).items():
                out[f"{name}/{layer}"] = values.numpy()
        return out

    def write_scan(self, scan):
        _, slices, tokens = self.scan_rows(scan)
        columns = {**self.extract_2d(scan, slices, tokens), **self.extract_3d(scan, slices, tokens)}
        for key, x in columns.items():
            folder = ACTS / key
            folder.mkdir(parents=True, exist_ok=True)
            q, minmax = quantise(x.astype(np.float32))
            np.save(folder / f"{scan}.minmax.npy", minmax)
            np.save(folder / f"{scan}.npy", q)
        return {key: x.shape[1] for key, x in columns.items()}


def scan_folder(scan):
    matches = [p for p in SCANS_ROOT.glob("*/flex_step4") if p.parent.name.endswith(scan)]
    assert len(matches) == 1, (scan, matches)
    return matches[0]


def layer_list():
    return sorted(str(p.relative_to(ACTS)) for p in ACTS.glob("*/*") if p.is_dir())


def load_layer(layer, scans, units=None):
    """(rows of these scans, in row-table order, x units) float32."""
    parts = []
    for scan in scans:
        q = np.load(ACTS / layer / f"{scan}.npy", mmap_mode="r")
        minmax = np.load(ACTS / layer / f"{scan}.minmax.npy")
        if units is not None:
            q, minmax = q[:, units], minmax[:, units]
        parts.append(dequantise(np.asarray(q), minmax))
    return np.concatenate(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-2d", nargs="*", default=list(MODELS_2D))
    parser.add_argument("--models-3d", nargs="*", default=list(MODELS_3D))
    parser.add_argument("--scans", nargs="*")
    parser.add_argument("--reverse", action="store_true", help="walk the scans last to first (a second process)")
    args = parser.parse_args()

    torch.set_num_threads(8)
    writer = StoreWriter(args.models_2d, args.models_3d, torch.device("cuda"))
    scans = args.scans or writer.rows.scans
    for scan in reversed(scans) if args.reverse else scans:
        done, claim = ACTS / f"{scan}.done", ACTS / f"{scan}.working"
        if done.exists() or claim.exists():
            continue
        ACTS.mkdir(parents=True, exist_ok=True)
        claim.touch()  # two processes walking opposite ends never take the same scan
        start = time.time()
        widths = writer.write_scan(scan)
        done.write_text(json.dumps(widths))
        claim.unlink()
        print(f"{scan}: {len(widths)} layers, {sum(widths.values())} units in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
