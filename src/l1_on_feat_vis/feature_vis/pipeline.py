"""Find sparse images that reproduce one neuron's thresholded activation map.

How we got here: experiment-log.md (this is experiment 10 + seed averaging).

Per reference image:
  1. Target: forward the image (short side `size`, InceptionV1 input
     x * 255 - 117), read the neuron's post-ReLU spatial map, set every cell
     below `threshold` to 0. With `threshold_frac` > 0 the threshold is that
     fraction of each image's own peak instead (an absolute value tuned on one
     photo skips photos whose peak is lower). Images with no cell at or above
     the threshold are skipped (empty target).
  2. Optimise one copy of the image per seed, as raw RGB pixels in [0, 1]:
       loss_s = MSE(map(T_s(x_s)), T_s(target)) + l1 * mean|x_s| + tv * TV(x_s)
     - T_s: a random shift (+-12 px) / scale (0.9-1.1) / rotation (+-10 deg)
       drawn per seed per step. The same normalised affine warps the image
       (around grey) and the target map (around 0), so map cells stay aligned.
       The last `fine_tune_steps` steps use no transform, so the final fit is
       on the image that gets saved.
     - TV is anisotropic: mean |horizontal diff| + mean |vertical diff|.
       It turns L1's isolated specks into connected strokes.
     - Pixels are clamped to [0, 1] after every Adam step.
     Seeds run as one batch: per-seed losses are summed and Adam is
     elementwise, so this equals separate runs.
     The forward pass stops right after the target layer (the hook raises
     StopForward), since later layers never affect the loss.
     With `bf16` the network forward runs under bfloat16 autocast (about 2x on
     an H200); pixels, losses and Adam stay float32.
  3. Average the seeds. A single seed fits the map almost exactly with
     seed-specific pixel edits; the average keeps only the structure all seeds
     share (a sparse version of the real image content).

Output layout:
  <out>/<image>.png                       seed-averaged image (only files here)
  <out>/intermediates/<image>/            input, map, target, per-seed images,
                                          per-seed spread, mean-image map, summary
  <out>/intermediates/run_summary.json    every image, including skipped ones
"""

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lucent.modelzoo import inceptionv1
from PIL import Image

from l1_on_feat_vis.feature_vis.activation_map import load_rgb, to_grayscale

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_SHIFT_PX = 12
SCALES = [1 + (i - 5) / 50.0 for i in range(11)]
ANGLES_DEG = list(range(-10, 11)) + 5 * [0]


class StopForward(Exception):
    pass


@dataclass
class MatchConfig:
    threshold: float = 600.0
    threshold_frac: float = 0.0
    size: int = 448
    l1_weight: float = 3e5
    tv_weight: float = 1e5
    lr: float = 0.01
    steps: int = 768
    fine_tune_steps: int = 96
    seeds: int = 20
    bf16: bool = True


def parse_neuron(neuron):
    layer, channel = neuron.split(":")
    return layer, int(channel)


def to_inceptionv1_input(pixels):
    return pixels * 255 - 117


def total_variation_per_sample(images):
    dx = (images[..., :, 1:] - images[..., :, :-1]).abs().mean(dim=(1, 2, 3))
    dy = (images[..., 1:, :] - images[..., :-1, :]).abs().mean(dim=(1, 2, 3))
    return dx + dy


def random_affine_theta(rng, height, width):
    """Output->input mapping in normalised coords for one shift/scale/rotate.

    Rotation happens in pixel space, so the normalised x/y axes (different
    pixel lengths for a non-square image) are rescaled around it.
    """
    angle = math.radians(rng.choice(ANGLES_DEG))
    scale = rng.choice(SCALES)
    shift_x = rng.randint(-MAX_SHIFT_PX, MAX_SHIFT_PX) / (width / 2)
    shift_y = rng.randint(-MAX_SHIFT_PX, MAX_SHIFT_PX) / (height / 2)
    aspect = height / width
    cos, sin = math.cos(angle) / scale, math.sin(angle) / scale
    return [[cos, -sin * aspect, shift_x], [sin / aspect, cos, shift_y]]


def identity_theta():
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def warp(tensor, theta, fill):
    grid = F.affine_grid(theta, tensor.shape, align_corners=False)
    return F.grid_sample(tensor - fill, grid, align_corners=False) + fill


def image_to_tensor(image):
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)


def save_rgb(pixels, path):
    array = pixels.permute(1, 2, 0).cpu().numpy()
    Image.fromarray((array.clip(0, 1) * 255).round().astype(np.uint8)).save(path)


def save_map_pair(produced, target, path, cell_px=16):
    """Side by side: target | produced, both scaled by the target's peak."""
    peak = target.max()
    divider = np.full((target.shape[0], 1), peak)
    pair = np.concatenate([target, divider, np.clip(produced, 0, peak)], axis=1)
    size = (pair.shape[1] * cell_px, pair.shape[0] * cell_px)
    to_grayscale(pair).resize(size, resample=Image.NEAREST).save(path)


class SparseActivationMatcher:
    def __init__(self, layer, channel, config, device):
        self.layer = layer
        self.channel = channel
        self.config = config
        self.device = device
        self.model = inceptionv1(pretrained=True).to(device).eval()
        self.captured = {}
        getattr(self.model, layer).register_forward_hook(self.capture_and_stop)

    def capture_and_stop(self, module, inputs, output):
        self.captured["output"] = output
        raise StopForward

    def channel_maps(self, pixels):
        use_bf16 = self.config.bf16 and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            try:
                self.model(to_inceptionv1_input(pixels))
            except StopForward:
                pass
        return self.captured["output"][:, self.channel : self.channel + 1].float()

    def threshold_for(self, activation_map):
        if self.config.threshold_frac > 0:
            return self.config.threshold_frac * activation_map.max().item()
        return self.config.threshold

    def build_target(self, reference):
        with torch.no_grad():
            activation_map = self.channel_maps(reference)
        threshold = self.threshold_for(activation_map)
        target = torch.where(activation_map >= threshold, activation_map, 0.0)
        return activation_map[0, 0], target, threshold

    def thetas(self, rngs, height, width, use_transforms):
        rows = [random_affine_theta(rng, height, width) if use_transforms else identity_theta() for rng in rngs]
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def optimise(self, reference, target, log_prefix="", log_every=64):
        cfg = self.config
        _, _, height, width = reference.shape
        images = reference.repeat(cfg.seeds, 1, 1, 1).clone().requires_grad_(True)
        targets = target.expand(cfg.seeds, -1, -1, -1)
        rngs = [random.Random(seed) for seed in range(cfg.seeds)]
        optimizer = torch.optim.Adam([images], lr=cfg.lr)
        for step in range(1, cfg.steps + 1):
            use_transforms = step <= cfg.steps - cfg.fine_tune_steps
            theta = self.thetas(rngs, height, width, use_transforms)
            produced = self.channel_maps(warp(images, theta, fill=0.5))
            mse = ((produced - warp(targets, theta, fill=0.0)) ** 2).mean(dim=(1, 2, 3))
            l1 = images.abs().mean(dim=(1, 2, 3))
            tv = total_variation_per_sample(images)
            loss = (mse + cfg.l1_weight * l1 + cfg.tv_weight * tv).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                images.clamp_(0.0, 1.0)
            if step % log_every == 0 or step == cfg.steps:
                print(f"{log_prefix}step {step}/{cfg.steps}  median seed mse {mse.median().item():.1f}", flush=True)
        return images.detach()

    def final_maps(self, images):
        with torch.no_grad():
            return self.channel_maps(images)[:, 0]


def per_seed_summary(images, maps, target):
    mse = ((maps - target[0]) ** 2).mean(dim=(1, 2))
    rows = []
    for seed in range(images.shape[0]):
        rows.append({
            "seed": seed,
            "map_mse": mse[seed].item(),
            "map_peak": maps[seed].max().item(),
            "exact_zero_frac": (images[seed].amax(0) == 0).float().mean().item(),
            "mean_abs": images[seed].abs().mean().item(),
        })
    return rows


def process_image(matcher, image_path, out_dir):
    cfg = matcher.config
    name = image_path.stem
    work_dir = out_dir / "intermediates" / name
    (work_dir / "seeds").mkdir(parents=True, exist_ok=True)

    image = load_rgb(image_path, cfg.size)
    reference = image_to_tensor(image).to(matcher.device)
    image.save(work_dir / "input.png")
    activation_map, target, threshold = matcher.build_target(reference)
    np.save(work_dir / "activation_map.npy", activation_map.cpu().numpy())
    np.save(work_dir / "target.npy", target[0, 0].cpu().numpy())
    to_grayscale(target[0, 0].cpu().numpy()).resize(image.size, Image.NEAREST).save(work_dir / "target.png")

    summary = {
        "image": name,
        "map_peak": activation_map.max().item(),
        "threshold": threshold,
        "target_cells": int((target > 0).sum()),
    }
    if summary["target_cells"] == 0:
        summary["skipped"] = f"map peak {summary['map_peak']:.1f} is below threshold {threshold:g}"
        (work_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        return summary

    images = matcher.optimise(reference, target, log_prefix=f"{name}: ")
    maps = matcher.final_maps(images)
    for seed in range(cfg.seeds):
        save_rgb(images[seed], work_dir / "seeds" / f"seed_{seed:02d}.png")
        save_map_pair(maps[seed].cpu().numpy(), target[0, 0].cpu().numpy(), work_dir / "seeds" / f"seed_{seed:02d}_maps.png")

    mean_image = images.mean(0)
    mean_map = matcher.final_maps(mean_image[None])[0]
    save_rgb(mean_image, out_dir / f"{name}.png")
    np.save(work_dir / "mean_image.npy", mean_image.cpu().numpy())
    to_grayscale(images.std(0).amax(0).cpu().numpy()).save(work_dir / "std_across_seeds.png")
    save_map_pair(mean_map.cpu().numpy(), target[0, 0].cpu().numpy(), work_dir / "mean_image_maps.png")

    summary["seeds"] = per_seed_summary(images, maps, target[0])
    summary["mean_image"] = {
        "map_mse": ((mean_map - target[0, 0]) ** 2).mean().item(),
        "map_peak": mean_map.max().item(),
        "mean_abs": mean_image.abs().mean().item(),
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def find_images(inputs):
    paths = []
    for path in inputs:
        if path.is_dir():
            paths += sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        else:
            paths.append(path)
    return paths


def main():
    defaults = MatchConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+", help="image files or folders of images")
    parser.add_argument("--neuron", default="mixed4e:55", help="layer:channel")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=8,
                        help="CPU threads; torch's default (one per core) oversubscribes on this model")
    for field, value in asdict(defaults).items():
        flag = f"--{field.replace('_', '-')}"
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(flag, type=type(value), default=value)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    config = MatchConfig(**{field: getattr(args, field) for field in asdict(defaults)})
    layer, channel = parse_neuron(args.neuron)
    matcher = SparseActivationMatcher(layer, channel, config, torch.device(args.device))
    args.out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for image_path in find_images(args.inputs):
        summary = process_image(matcher, image_path, args.out)
        summaries.append(summary)
        if "skipped" in summary:
            print(f"{summary['image']}: skipped ({summary['skipped']})", flush=True)
        else:
            seed_mse = [row["map_mse"] for row in summary["seeds"]]
            print(f"{summary['image']}: {summary['target_cells']} target cells, "
                  f"seed map MSE median {np.median(seed_mse):.1f}, "
                  f"mean-image map MSE {summary['mean_image']['map_mse']:.1f}", flush=True)

    write_run_summary(args.out, args.neuron, config)


def write_run_summary(out_dir, neuron, config):
    """Rebuilt from every image's summary.json, so several processes (e.g. one
    per GPU, each with part of the images) can share one output folder."""
    intermediates = out_dir / "intermediates"
    images = [json.loads(path.read_text()) for path in sorted(intermediates.glob("*/summary.json"))]
    run = {"neuron": neuron, "config": asdict(config), "images": images}
    (intermediates / "run_summary.json").write_text(json.dumps(run, indent=1))


if __name__ == "__main__":
    main()
