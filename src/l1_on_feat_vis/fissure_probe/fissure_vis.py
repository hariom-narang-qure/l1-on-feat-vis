"""Sparse activation matching (pipeline.py) for DINOv2 fissure units on CT slices.

Question: what is the smallest part of a CT slice that still drives a fissure
unit's map? If the units look at the fissure line, the sparse image should
keep the line and turn the rest of the lung to air.

Per (slice, unit):
  1. Input: the axial slice in the lung window [-1000, -200] HU -> [0, 1];
     0 is air. DINOv2 sees it as in dino_features.py (RGB copy, ImageNet
     normalisation, 1036 px, 74 x 74 tokens). The forward stops after block 8.
  2. Response map: the unit's signed value (sign from the train scans) minus
     its lung median on the original slice, through a ReLU. Residual units have
     no natural zero, so "off" is "at or below the slice's typical lung value".
  3. Target: response cells >= `threshold_frac` x the peak inside the lung,
     everything else 0. With `--target fissure` only cells the sheet passes
     through are kept (threshold relative to the peak there): a unit's peak on
     one slice need not be on the fissure, even if most of its peaks are.
  4. Optimise one copy of the slice per seed, pixels clamped to [0, 1]:
       loss = MSE(response(T(x)), T(target)) + l1 * mean_lung(x) + tv * TV_lung(x)
     - Only lung pixels (lung mask dilated 3 px) are free; the body stays as
       in the scan so the network keeps its context.
     - Baseline (what "removed" means): `air` pushes pixels to 0 (-1000 HU),
       like the cat runs' black background. A whole lung of air is an
       unnatural hard edge, though, so `blur` uses a smoothed copy of the lung
       instead (Gaussian sigma 4 px, normalised inside the lung so the body
       does not bleed in; Fong & Vedaldi 2017 use a blurred image the same way).
       L1 and TV act on the difference from the baseline, so only detail the
       unit needs survives.
     - T: per-seed random shift / scale / rotation, applied to the image
       (filled with the original slice) and the target (filled with 0).
       The last `fine_tune_steps` steps use no transform.
     - Weights: the cat pipeline's L1 weight was 0.012 x the value that equals
       the L1 and MSE gradient norms at the start (TV = L1 / 3). That ratio is
       reused here, scaled by `l1_scale`, since DINOv2's activation scale differs.
  5. Average the seeds.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from l1_on_feat_vis.fissure_probe.data import LoadedScan, list_scans
from l1_on_feat_vis.fissure_probe.dino_features import (GRID, IMAGENET_MEAN, IMAGENET_STD, INPUT_PX, MODEL_NAME,
                                                       to_grid)
from l1_on_feat_vis.pipeline import identity_theta, random_affine_theta, warp
from scipy import ndimage
from transformers import AutoModel

WINDOW_HU = (-1000.0, -200.0)
CAT_RATIO = 0.012  # l1 weight / gradient-balance weight that worked on the cats
UNITS = {"res8:355": ("res", 355, -1), "res8:442": ("res", 442, -1), "mlp8:2235": ("mlp", 2235, 1)}


class TruncatedDino:
    """DINOv2 up to block 8, returning one unit's signed (74 x 74) map per image."""

    def __init__(self, device, block=8):
        self.model = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.block, self.device = block, device
        self.skip = 1 + self.model.config.num_register_tokens
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device)[None, :, None, None]
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device)[None, :, None, None]
        self.mlp = {}
        self.model.encoder.layer[block].mlp.activation_fn.register_forward_hook(lambda m, i, o: self.mlp.update(out=o))

    def unit_map(self, grey, kind, index, sign):
        x = (grey.repeat(1, 3, 1, 1) - self.mean) / self.std
        x = F.interpolate(x, size=(INPUT_PX, INPUT_PX), mode="bilinear", align_corners=False)
        # bf16 autocast on the GPU (~2x); the pixels, losses and Adam stay float32.
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            hidden = self.model.embeddings(x)
            for layer in self.model.encoder.layer[: self.block + 1]:
                output = layer(hidden)
                hidden = output[0] if isinstance(output, tuple) else output
        tokens = hidden if kind == "res" else self.mlp["out"]
        return sign * tokens[:, self.skip:, index].float().reshape(-1, GRID, GRID)


def window(hu):
    return np.clip((hu - WINDOW_HU[0]) / (WINDOW_HU[1] - WINDOW_HU[0]), 0, 1).astype(np.float32)


class FissureMatch:
    def __init__(self, dino, scan, z, unit, threshold_frac, seeds, target="lung", baseline="air"):
        self.dino, self.device = dino, dino.device
        self.kind, self.index, self.sign = UNITS[unit]
        self.original = torch.from_numpy(window(scan.hu[:, :, z]))[None, None].to(self.device)
        lung = ndimage.binary_dilation(scan.lung[:, :, z], iterations=3)
        self.free = torch.from_numpy(lung.astype(np.float32))[None, None].to(self.device)
        self.lung_tokens = torch.from_numpy(to_grid(scan.lung[:, :, z], "avg") > 0.5).to(self.device)
        with torch.no_grad():
            raw = self.raw_map(self.original)[0]
        self.offset = raw[self.lung_tokens].median()
        response = torch.relu(raw - self.offset) * self.lung_tokens
        if target == "fissure":
            sheet_tokens = to_grid(scan.sheet[:, :, z] & scan.lung[:, :, z], "max") > 0
            response = response * torch.from_numpy(sheet_tokens).to(self.device)
        self.threshold = threshold_frac * response.max()
        self.target = torch.where(response >= self.threshold, response, 0.0)[None, None]
        self.seeds = seeds
        self.baseline = self.blurred_lung(scan.lung[:, :, z]) if baseline == "blur" else torch.zeros_like(self.original)

    def blurred_lung(self, lung, sigma=4.0):
        """Gaussian blur of the slice inside the lung only (normalised convolution)."""
        image = self.original[0, 0].cpu().numpy()
        weight = ndimage.binary_dilation(lung, iterations=3).astype(np.float32)
        blurred = ndimage.gaussian_filter(image * weight, sigma) / np.maximum(ndimage.gaussian_filter(weight, sigma), 1e-3)
        return torch.from_numpy(blurred.astype(np.float32))[None, None].to(self.device)

    def raw_map(self, grey):
        return self.dino.unit_map(grey, self.kind, self.index, self.sign)

    def response(self, grey):
        return torch.relu(self.raw_map(grey) - self.offset)[:, None]

    def compose(self, free_pixels):
        """Free lung pixels from the optimiser, everything else from the scan."""
        return self.free * free_pixels + (1 - self.free) * self.original

    def terms(self, free_pixels, theta):
        image = self.compose(free_pixels)
        warped = warp(image, theta, fill=0.0) if theta is not None else image
        target = warp(self.target.expand(len(image), -1, -1, -1), theta, fill=0.0) if theta is not None else self.target.expand(len(image), -1, -1, -1)
        mse = ((self.response(warped) - target) ** 2).mean(dim=(1, 2, 3))
        area = self.free.sum()
        detail = free_pixels - self.baseline
        l1 = (detail.abs() * self.free).sum(dim=(1, 2, 3)) / area
        dx = ((detail[..., :, 1:] - detail[..., :, :-1]).abs() * self.free[..., :, 1:]).sum(dim=(1, 2, 3))
        dy = ((detail[..., 1:, :] - detail[..., :-1, :]).abs() * self.free[..., 1:, :]).sum(dim=(1, 2, 3))
        return mse, l1, (dx + dy) / area

    def balance_weight(self):
        """The L1 weight at which the L1 and MSE gradient norms are equal at the start."""
        pixels = self.original.clone().requires_grad_(True)
        mse, l1, _ = self.terms(pixels, None)
        g_mse, = torch.autograd.grad(mse.sum(), pixels, retain_graph=True)
        g_l1, = torch.autograd.grad(l1.sum(), pixels)
        return float(g_mse.norm() / (g_l1.norm() + 1e-12))

    def run(self, l1_scale, steps=768, fine_tune_steps=96, lr=0.01):
        balance = self.balance_weight()
        l1_weight = CAT_RATIO * l1_scale * balance
        tv_weight = l1_weight / 3
        pixels = self.original.repeat(self.seeds, 1, 1, 1).clone().requires_grad_(True)
        rngs = [random.Random(seed) for seed in range(self.seeds)]
        optimizer = torch.optim.Adam([pixels], lr=lr)
        size = self.original.shape[-1]
        for step in range(1, steps + 1):
            transforms = step <= steps - fine_tune_steps
            rows = [random_affine_theta(rng, size, size) if transforms else identity_theta() for rng in rngs]
            theta = torch.tensor(rows, dtype=torch.float32, device=self.device)
            mse, l1, tv = self.terms(pixels, theta)
            loss = (mse + l1_weight * l1 + tv_weight * tv).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pixels.clamp_(0.0, 1.0)
        with torch.no_grad():
            final = self.compose(pixels.detach())
            maps = self.response(final)[:, 0]
            mean_image = final.mean(0, keepdim=True)
            mean_map = self.response(mean_image)[0, 0]
        return {"images": final[:, 0].cpu().numpy(), "mean_image": mean_image[0, 0].cpu().numpy(),
                "seed_mse": ((maps - self.target[0, 0]) ** 2).mean(dim=(1, 2)).cpu().numpy(),
                "mean_map": mean_map.cpu().numpy(), "target": self.target[0, 0].cpu().numpy(),
                "l1_weight": l1_weight, "balance": balance}


def lung_air_fraction(image, lung):
    """Share of lung pixels pushed to air (< 0.02 in the window, i.e. < -984 HU)."""
    return float((image[lung] < 0.02).mean())


def sheet_capture(image, sheet, lung, original, baseline):
    """Kept detail = lung pixels > 0.02 away from the baseline. Returns the share of
    kept detail within 2 px of the sheet, and the share of the sheet that kept its
    detail. The original slice's detail gives the base rate."""
    kept = (np.abs(image - baseline) > 0.02) & lung
    near_sheet = ndimage.binary_dilation(sheet, iterations=2) & lung
    original_detail = (np.abs(original - baseline) > 0.02) & lung
    base = (original_detail & near_sheet).sum() / max(original_detail.sum(), 1)
    return {"kept_near_sheet": float((kept & near_sheet).sum() / max(kept.sum(), 1)),
            "sheet_kept": float((kept & sheet).sum() / max((sheet & lung).sum(), 1)),
            "base_rate_near_sheet": float(base)}


def panel(image, sheet=None, tile=384):
    grey = (np.clip(image, 0, 1) * 255).astype(np.uint8).T
    rgb = np.stack([grey] * 3, -1)
    if sheet is not None:
        rgb = rgb.copy()
        rgb[sheet.T] = [255, 40, 40]
    return Image.fromarray(rgb).resize((tile, tile))


def map_panel(values, peak, tile=384):
    grid = (np.clip(values / (peak + 1e-9), 0, 1) * 255).astype(np.uint8).T
    return Image.fromarray(grid).convert("RGB").resize((tile, tile), Image.NEAREST)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", required=True)
    parser.add_argument("--slices", type=int, nargs="+", required=True)
    parser.add_argument("--units", nargs="+", default=list(UNITS))
    parser.add_argument("--l1-scales", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    parser.add_argument("--threshold-frac", type=float, default=0.6)
    parser.add_argument("--target", choices=["lung", "fissure"], default="lung")
    parser.add_argument("--baseline", choices=["air", "blur"], default="air")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=768)
    parser.add_argument("--fine-tune-steps", type=int, default=96)
    parser.add_argument("--out", type=Path, default=Path("outputs/fissure_probe/fissure_vis"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    dino = TruncatedDino(device)
    record = next(r for r in list_scans() if r.short == args.scan)
    scan = LoadedScan(record)
    summary = []
    for z in args.slices:
        lung, sheet = scan.lung[:, :, z], scan.sheet[:, :, z] & scan.lung[:, :, z]
        original = window(scan.hu[:, :, z])
        for unit in args.units:
            match = FissureMatch(dino, scan, z, unit, args.threshold_frac, args.seeds, args.target, args.baseline)
            baseline = match.baseline[0, 0].cpu().numpy()
            row_panels = [panel(original), panel(original, sheet), panel(baseline * lung + original * ~lung)]
            for scale in args.l1_scales:
                result = match.run(scale, steps=args.steps, fine_tune_steps=args.fine_tune_steps)
                stats = {"scan": args.scan, "slice": z, "unit": unit, "l1_scale": scale,
                         "target_cells": int((result["target"] > 0).sum()),
                         "seed_mse_median": float(np.median(result["seed_mse"])),
                         "mean_image_mse": float(((result["mean_map"] - result["target"]) ** 2).mean()),
                         "lung_air_fraction": lung_air_fraction(result["mean_image"], lung),
                         **sheet_capture(result["mean_image"], sheet, lung, original, baseline)}
                summary.append(stats)
                print(json.dumps(stats), flush=True)
                row_panels.append(panel(result["mean_image"]))
                np.save(args.out / f"{args.scan}_z{z}_{unit.replace(':', '-')}_l1x{scale:g}_mean.npy", result["mean_image"])
            peak = result["target"].max()
            row_panels += [map_panel(result["target"], peak), map_panel(result["mean_map"], peak)]
            figure = Image.new("RGB", (len(row_panels) * 388, 404), "white")
            draw = ImageDraw.Draw(figure)
            titles = ["CT", "sheet", f"baseline ({args.baseline})"] + [f"mean, L1 x{s:g}" for s in args.l1_scales] + ["target map", "map of mean"]
            for i, (title, img) in enumerate(zip(titles, row_panels)):
                draw.text((i * 388 + 4, 2), f"{unit} {title}", fill="black")
                figure.paste(img, (i * 388, 18))
            figure.save(args.out / f"{args.scan}_z{z}_{unit.replace(':', '-')}_{args.target}_{args.baseline}.png")
    (args.out / f"summary_{args.scan}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
