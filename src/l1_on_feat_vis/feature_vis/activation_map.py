"""Spatial activation map of one InceptionV1 channel on a real image.

Preprocessing mirrors what lucent feeds the network during visualisation:
RGB in [0, 1] -> x * 255 - 117 (the original TF InceptionV1 input range).
The image is resized so its short side is `--size`, keeping aspect ratio;
InceptionV1 is fully convolutional up to mixed5b, so non-square inputs work.

The activation is read from the post-ReLU concat module of the layer
(e.g. `model.mixed4e`) via a forward hook. It is saved as grayscale scaled
to [0, max]; zero stays black, so the absolute dead areas are visible.
Two versions: the native-resolution map (stride 16 at mixed4e) and one
upsampled with nearest-neighbour to the input size for side-by-side viewing.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from lucent.modelzoo import inceptionv1
from PIL import Image


def load_rgb(path, short_side):
    image = Image.open(path).convert("RGB")
    scale = short_side / min(image.size)
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.BICUBIC)


def preprocess_inceptionv1(image):
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
    return pixels.permute(2, 0, 1).unsqueeze(0) * 255 - 117


def capture_layer(model, layer, x):
    captured = {}
    handle = getattr(model, layer).register_forward_hook(
        lambda module, inputs, output: captured.update(output=output)
    )
    with torch.no_grad():
        model(x)
    handle.remove()
    return captured["output"]


def to_grayscale(activation_map):
    peak = activation_map.max()
    scaled = activation_map / peak if peak > 0 else activation_map
    return Image.fromarray((scaled * 255).round().astype(np.uint8), mode="L")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--layer", default="mixed4e")
    parser.add_argument("--channel", type=int, default=55)
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--out", type=Path, default=Path("outputs/activation_maps"))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    model = inceptionv1(pretrained=True).eval()
    image = load_rgb(args.image, args.size)
    activations = capture_layer(model, args.layer, preprocess_inceptionv1(image))
    activation_map = activations[0, args.channel].numpy()

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.image.stem}_{args.layer}_{args.channel}"
    np.save(args.out / f"{stem}.npy", activation_map)
    gray = to_grayscale(activation_map)
    gray.save(args.out / f"{stem}.png")
    gray.resize(image.size, Image.NEAREST).save(args.out / f"{stem}_upsampled.png")
    image.save(args.out / f"{args.image.stem}_input.png")
    print(
        f"map shape={activation_map.shape}  max={activation_map.max():.2f}  "
        f"mean={activation_map.mean():.2f}  argmax(y,x)="
        f"{np.unravel_index(activation_map.argmax(), activation_map.shape)}"
    )


if __name__ == "__main__":
    main()
