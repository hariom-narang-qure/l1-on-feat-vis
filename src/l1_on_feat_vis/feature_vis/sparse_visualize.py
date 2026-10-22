"""Feature visualisation with an L1 sparsity penalty on the input image.

Differences from lucent's default `render_vis` pipeline:
  - Parameterisation: lucent optimises FFT coefficients, decorrelates colour
    and squashes with a sigmoid. Here the leaf tensor *is* the RGB image in
    [0, 1], so the L1 penalty acts on actual pixel values.
  - Normalisation: InceptionV1 expects inputs in [-117, 138]; lucent maps
    [0, 1] -> x * 255 - 117 as its last transform. We keep that transform (and
    the jitter/scale/rotate robustness transforms) inside the graph, so
    gradients flow through them back to the raw image.
  - Validity: after every Adam step pixels are projected back to [0, 1]
    (projected gradient descent) instead of using a sigmoid, which would turn
    the optimised variable back into a reparameterisation.
  - Padding uses 0 (black) instead of lucent's 0.5 grey, so the border
    matches the background that the L1 term pushes pixels towards.

Loss = -mean_activation(layer:channel) + l1_weight * mean(|image|)
Using the mean (not the sum) of |image| keeps `l1_weight` independent of
image size.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from lucent.modelzoo import inceptionv1
from lucent.optvis import objectives, transform
from lucent.optvis.render import hook_model
from PIL import Image


def robustness_transforms():
    return transform.compose(
        [
            transform.pad(12, mode="constant", constant_value=0.0),
            transform.jitter(8),
            transform.random_scale([1 + (i - 5) / 50.0 for i in range(11)]),
            transform.random_rotate(list(range(-10, 11)) + 5 * [0]),
            transform.jitter(4),
            transform.preprocess_inceptionv1(),
        ]
    )


def init_image(size, device, seed):
    generator = torch.Generator().manual_seed(seed)
    image = 0.5 + 0.01 * torch.randn(1, 3, size, size, generator=generator)
    return image.to(device).requires_grad_(True)


def l1_penalty(image):
    return image.abs().mean()


def sparsity(image, threshold=0.02):
    return (image.abs() < threshold).float().mean().item()


class SparseChannelVisualizer:
    def __init__(self, model, layer, channel, device, size=224, lr=0.05):
        self.model = model
        self.layer = layer
        self.objective = objectives.channel(layer, channel)
        self.device = device
        self.size = size
        self.lr = lr
        self.transform_f = robustness_transforms()

    def run(self, l1_weight, steps, seed):
        torch.manual_seed(seed)  # the random transforms use torch's global RNG
        image = init_image(self.size, self.device, seed)
        hook = hook_model(self.model, lambda: image)
        optimizer = torch.optim.Adam([image], lr=self.lr)
        for _ in range(steps):
            optimizer.zero_grad()
            self.model(self.transform_f(image))
            activation_loss = self.objective(hook)
            loss = activation_loss + l1_weight * l1_penalty(image)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                image.clamp_(0.0, 1.0)
        return image.detach(), -activation_loss.item()


def save_image(image, path):
    pixels = image[0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray((pixels * 255).round().astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", default="mixed4e")
    parser.add_argument("--channel", type=int, default=55)
    parser.add_argument("--l1-weights", type=float, nargs="+", default=[0.0, 10.0, 100.0])
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("outputs/l1"))
    # torch's default (one thread per core, 56 here) oversubscribes on this
    # small model and is ~50x slower per step than 8 threads on CPU.
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    model = inceptionv1(pretrained=True).to(device).eval()
    visualizer = SparseChannelVisualizer(model, args.layer, args.channel, device, lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)

    for l1_weight in args.l1_weights:
        image, activation = visualizer.run(l1_weight, args.steps, args.seed)
        save_image(image, args.out / f"{args.layer}_{args.channel}_l1_{l1_weight:g}.png")
        print(
            f"l1_weight={l1_weight:g}  activation={activation:.2f}  "
            f"mean|x|={l1_penalty(image).item():.4f}  near-zero frac={sparsity(image):.3f}"
        )


if __name__ == "__main__":
    main()
