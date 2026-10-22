"""Average the final images of several seeded runs of match_spatial_map.py.

Runs that start from the same reference differ only in their random
transform sequence, so the per-pixel mean keeps structure every seed agrees
on and fades seed-specific flecks. Outputs:
  - mean image, and a contrast-stretched copy (mean / its 99.9th percentile)
    because sparse runs barely use the top of the [0, 1] range
  - per-pixel std across seeds (grayscale, scaled to its max): where seeds disagree
  - the channel's activation map of the *mean image* vs the target
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from lucent.modelzoo import inceptionv1

from l1_on_feat_vis.activation_map import capture_layer, to_grayscale
from l1_on_feat_vis.match_spatial_map import save_map_pair
from l1_on_feat_vis.sparse_visualize import save_image


def contrast_stretch(image, percentile=99.9):
    return np.clip(image / np.percentile(image, percentile), 0, 1)


def channel_map_of(image, layer, channel, device):
    model = inceptionv1(pretrained=True).to(device).eval()
    x = torch.from_numpy(image)[None].to(device) * 255 - 117
    return capture_layer(model, layer, x)[0, channel].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path, nargs="+", help="*_image.npy files, shape (3, H, W)")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--layer", default="mixed4e")
    parser.add_argument("--channel", type=int, default=55)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    stack = np.stack([np.load(path) for path in args.images])
    mean, std = stack.mean(0), stack.std(0)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "mean_image.npy", mean)
    save_image(torch.from_numpy(mean)[None], args.out / "mean_image.png")
    save_image(torch.from_numpy(contrast_stretch(mean))[None], args.out / "mean_image_stretched.png")
    to_grayscale(std.max(0)).save(args.out / "std_across_seeds.png")

    target = np.load(args.target)
    produced = channel_map_of(mean, args.layer, args.channel, torch.device(args.device))
    save_map_pair(produced, target, args.out / "mean_image_maps.png")
    print(f"{len(stack)} runs  mean|x|={mean.mean():.4f}  exactly-0 in all runs={(stack.max(axis=(0, 1)) == 0).mean():.3f}")
    print(f"map of mean image: max={produced.max():.1f}  mse={np.mean((produced - target) ** 2):.1f}")


if __name__ == "__main__":
    main()
