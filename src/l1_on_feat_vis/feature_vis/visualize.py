"""Feature visualisation of InceptionV1 (GoogLeNet) channels with lucent.

Two kinds of images are produced for a channel:
  - plain: maximise the mean activation of the channel, using lucent's default
    FFT-parameterised, decorrelated image and standard transform robustness.
  - diversity: optimise a batch of images for the channel jointly with a
    diversity penalty (negative pairwise cosine similarity of the batch's
    Gram matrices at an earlier layer), so each image shows a different facet.
Reference: Olah et al., "Feature Visualization", Distill 2017.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from lucent.modelzoo import inceptionv1
from lucent.optvis import objectives, param, render
from PIL import Image


def load_model(device):
    return inceptionv1(pretrained=True).to(device).eval()


def save_images(images, out_dir, stem):
    for i, image in enumerate(images):
        pixels = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(pixels).save(out_dir / f"{stem}_{i}.png")


def visualize_plain(model, objective, seeds, out_dir, stem, steps):
    for seed in seeds:
        torch.manual_seed(seed)
        images = render.render_vis(
            model, objective, thresholds=(steps,), show_image=False, progress=False
        )
        save_images(images[-1], out_dir, f"{stem}_seed{seed}")


def visualize_diversity(model, objective, diversity_layer, batch, out_dir, stem, steps):
    objective = objective - 1e2 * objectives.diversity(diversity_layer)
    param_f = lambda: param.image(224, batch=batch)
    images = render.render_vis(
        model, objective, param_f, thresholds=(steps,), show_image=False, progress=False
    )
    save_images(images[-1], out_dir, f"{stem}_diversity")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", default="mixed4e")
    parser.add_argument("--channel", type=int, default=55)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--diversity-batch", type=int, default=4)
    parser.add_argument("--diversity-layer", default="mixed4a")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    target = f"{args.layer}:{args.channel}"
    stem = f"{args.layer}_{args.channel}"
    args.out.mkdir(parents=True, exist_ok=True)

    visualize_plain(model, target, args.seeds, args.out, stem, args.steps)
    visualize_diversity(
        model,
        objectives.channel(args.layer, args.channel),
        args.diversity_layer,
        args.diversity_batch,
        args.out,
        stem,
        args.steps,
    )
    print(f"Saved visualisations of {target} to {args.out.resolve()}")


if __name__ == "__main__":
    main()
