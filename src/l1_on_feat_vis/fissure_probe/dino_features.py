"""Sample DINOv2 activations on axial CT slices, in the same row format as features.py.

Model: facebook/dinov2-with-registers-base (ViT-B/14, 12 layers, 4 register
tokens; Darcet et al. 2024). Registers absorb the high-norm artefact tokens of
plain DINOv2, so patch tokens track local content better.

Per scan:
  1. Pick axial slices: 6 among slices with >= 200 sheet voxels, 4 random lung slices.
  2. Lung window [-1000, -200] HU -> [0, 1] grey, copied to RGB, ImageNet
     normalisation, resized 512 -> 1036 px, so the patch grid is 74 x 74 and one
     token covers ~6.9 original pixels. (Genesis's [-1000, 1000] window leaves a
     fissure ~0.05 above parenchyma; this one makes it plainly visible.)
  3. Labels to the token grid with an adaptive pool: fissure / lobe_boundary /
     pleura = max (any pixel), lung = mean > 0.5, ring = mean > 0.5 and no fissure.
  4. Sample tokens like features.py: uniform lung tokens + extra fissure tokens.
  5. Store, per sampled token:
     - dino_res{l}: the residual stream after layer l (768 dims)
     - dino_mlp{l}: the MLP hidden units of layer l after GELU (3072 "neurons")
     plus intensity (token mean) and a 2D ridge score (token max) as baselines.
  The "random" network is the same architecture with random init.
2D ridge score: largest-magnitude Hessian eigenvalue, negated, at sigma 1.5 px
(bright line detector), the 2D analogue of the 3D sheetness baseline.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from transformers import AutoConfig, AutoModel

from l1_on_feat_vis.fissure_probe.data import LoadedScan, list_scans

MODEL_NAME = "facebook/dinov2-with-registers-base"
RESIDUAL_LAYERS = (2, 5, 8, 11)
MLP_LAYERS = (5, 8, 11)
GRID, INPUT_PX = 74, 1036
WINDOW_HU = (-1000.0, -200.0)
IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
UNIFORM_TOKENS, EXTRA_POSITIVES = 1200, 400
LABELS = ("fissure", "lobe_boundary", "pleura", "ring")
DINO_BLOCKS = [f"dino_res{l}" for l in RESIDUAL_LAYERS] + [f"dino_mlp{l}" for l in MLP_LAYERS]


def ridge_2d(image, sigma=1.5):
    hxx = ndimage.gaussian_filter(image, sigma, order=(2, 0)) * sigma**2
    hyy = ndimage.gaussian_filter(image, sigma, order=(0, 2)) * sigma**2
    hxy = ndimage.gaussian_filter(image, sigma, order=(1, 1)) * sigma**2
    half_trace, root = (hxx + hyy) / 2, np.sqrt(((hxx - hyy) / 2) ** 2 + hxy**2)
    most_negative = half_trace - root
    return np.clip(-most_negative, 0, None).astype(np.float32)


def to_grid(mask_or_values, mode):
    tensor = torch.from_numpy(np.ascontiguousarray(mask_or_values, dtype=np.float32))[None, None]
    pooled = F.adaptive_max_pool2d(tensor, GRID) if mode == "max" else F.adaptive_avg_pool2d(tensor, GRID)
    return pooled[0, 0].numpy()


class DinoNetworks:
    def __init__(self, device):
        self.device = device
        pretrained = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True)
        torch.manual_seed(0)
        random_init = AutoModel.from_config(AutoConfig.from_pretrained(MODEL_NAME))
        self.models = {"pretrained": pretrained.to(device).eval(), "random": random_init.to(device).eval()}
        self.skip = 1 + pretrained.config.num_register_tokens  # CLS + registers come before patch tokens

    def slice_input(self, hu_slice):
        grey = (np.clip(hu_slice, *WINDOW_HU) - WINDOW_HU[0]) / (WINDOW_HU[1] - WINDOW_HU[0])
        rgb = (np.repeat(grey[..., None], 3, -1) - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32))[None].to(self.device)
        return F.interpolate(x, size=(INPUT_PX, INPUT_PX), mode="bilinear", align_corners=False), grey

    def features(self, x):
        """{network: {block: (GRID*GRID, dims)}} for one slice."""
        out = {}
        for name, model in self.models.items():
            mlp = {}
            hooks = [model.encoder.layer[l].mlp.activation_fn.register_forward_hook(
                lambda m, i, o, l=l: mlp.__setitem__(l, o)) for l in MLP_LAYERS]
            with torch.no_grad():
                hidden = model(pixel_values=x, output_hidden_states=True).hidden_states
            for hook in hooks:
                hook.remove()
            blocks = {f"dino_res{l}": hidden[l + 1][0, self.skip:] for l in RESIDUAL_LAYERS}
            blocks.update({f"dino_mlp{l}": mlp[l][0, self.skip:] for l in MLP_LAYERS})
            out[name] = blocks
        return out


class DinoSampler:
    def __init__(self, device, seed=0):
        self.networks = DinoNetworks(device)
        self.rng = np.random.default_rng(seed)

    def pick_slices(self, scan):
        per_slice_sheet = (scan.fissure & scan.lung).sum(axis=(0, 1))
        lung_slices = np.flatnonzero(scan.lung.sum(axis=(0, 1)) > 2000)
        fissure_slices = np.flatnonzero(per_slice_sheet >= 200)
        picked = list(self.rng.choice(fissure_slices, min(6, len(fissure_slices)), replace=False))
        picked += list(self.rng.choice(lung_slices, 4, replace=False))
        return picked

    def pick_tokens(self, lung, fissure):
        lung_tokens = np.flatnonzero(lung)
        fissure_tokens = np.flatnonzero(lung & fissure)
        uniform = self.rng.choice(lung_tokens, min(UNIFORM_TOKENS, len(lung_tokens)), replace=False)
        extra = self.rng.choice(fissure_tokens, min(EXTRA_POSITIVES, len(fissure_tokens)), replace=False)
        return np.concatenate([uniform, extra]), np.concatenate([np.ones(len(uniform), bool), np.zeros(len(extra), bool)])

    def sample_slice(self, scan, z, slice_index, out):
        x, grey = self.networks.slice_input(scan.hu[:, :, z])
        lung = to_grid(scan.lung[:, :, z], "avg") > 0.5
        labels = {label: to_grid(getattr(scan, label)[:, :, z], "max") > 0 for label in LABELS if label != "ring"}
        labels["ring"] = (to_grid(scan.ring[:, :, z], "avg") > 0.5) & ~labels["fissure"]
        tokens, uniform = self.pick_tokens(lung.ravel(), labels["fissure"].ravel())
        if len(tokens) == 0:
            return
        features = self.networks.features(x)
        index = torch.from_numpy(tokens).to(self.networks.device)
        for block in DINO_BLOCKS:
            rows = out.setdefault(block, {key: [] for key in ("uniform", "patch", "intensity", "sheetness", *LABELS, "pretrained", "random")})
            rows["uniform"].append(uniform)
            rows["patch"].append(np.full(len(tokens), slice_index, np.int16))
            rows["intensity"].append(to_grid(grey, "avg").ravel()[tokens])
            rows["sheetness"].append(to_grid(ridge_2d(grey), "max").ravel()[tokens])
            for label in LABELS:
                rows[label].append(labels[label].ravel()[tokens])
            for name in self.networks.models:
                rows[name].append(features[name][block][index].half().cpu().numpy())

    def sample_scan(self, record, out_dir):
        scan = LoadedScan(record)
        out = {}
        for index, z in enumerate(self.pick_slices(scan)):
            self.sample_slice(scan, int(z), index, out)
        arrays = {f"{block}/{key}": np.concatenate(values) for block, rows in out.items() for key, values in rows.items()}
        np.savez(out_dir / f"{record.short}.npz", **arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/fissure_probe/dino_features"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    sampler = DinoSampler(torch.device(args.device))
    for record in list_scans():
        if (args.out / f"{record.short}.npz").exists():
            continue
        start = time.time()
        sampler.sample_scan(record, args.out)
        print(f"{record.short} {record.cohort} done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
