"""Optimise an image so a channel's spatial activation map matches a target map.

Target:
  1. Forward the reference image (e.g. the cat photo) once with no
     augmentation, read `layer[:, channel]` post-ReLU (activation_map.py).
  2. Zero every cell below `--threshold`; cells above keep their real value.

Regularisation (lucent's standard pair, Olah et al. 2017):
  - Parameterisation (`--param fft`): lucent's FFT spectrum with 1/f scaling,
    colour decorrelation and sigmoid. Biases towards natural-image statistics
    instead of the adversarial high-frequency noise raw pixels converge to.
    `--param pixel` optimises the RGB image directly (clamped to [0, 1]).
    `--param mask` (Fong & Vedaldi, "Interpretable Explanations of Black
    Boxes by Meaningful Perturbation", ICCV 2017): optimise a per-pixel mask
    m in [0, 1] (one value shared by RGB, starts at 1) and render
    image = m * reference. Every surviving pixel is exactly a reference pixel,
    so "stay close to the reference" and "prefer 0" never fight over a pixel's
    value: the only freedom is *which* pixels to keep. L1 / TV act on the mask.
    `--param mask-ste`: same, but the mask is binary in the forward pass.
    A soft mask settles on partial values (e.g. 0.3: costs less L1 than 1.0
    but keeps most of the activation). Here mask = hard + soft - soft.detach()
    with soft = sigmoid(logits), hard = (soft > 0.5): the forward value is
    exactly 0 or 1, the backward pass uses the sigmoid's gradient
    (straight-through estimator, Bengio et al. 2013). L1 then counts kept
    pixels. Logits start at +1 (soft = 0.73) so the image starts as the reference.
    `--param mask-gumbel`: straight-through Gumbel-sigmoid (binary concrete;
    Jang et al. 2017, Maddison et al. 2017). Each step samples
    m ~ hard(sigmoid((a + logistic noise) / tau)), so P(m = 1) = sigmoid(a).
    A logit near 0 gets sampled on and off, making its loss noisy and costly in
    expectation, so the optimiser is pushed towards confident 0 / 1 logits.
    L1 / TV act on p = sigmoid(a): L1 is then the expected number of kept
    pixels. The saved mask is deterministic: p > 0.5.
  - Robustness transforms (`--transforms`): every step samples one affine
    (shift up to +-12 px, scale 0.9-1.1, rotate +-10 deg), like lucent's
    jitter/scale/rotate. Because the loss is a per-cell MSE, the *same* affine
    is applied to the target map, so the target moves with the content.
    affine_grid works in normalised [-1, 1] coordinates, so one theta warps
    both the image and the stride-16 map consistently. This ignores the
    small offset of the map cells' receptive-field centres - fine at stride 16.
    The image is warped around grey (0.5) and the target around 0.

Loss = MSE(map(T(image)), T(target)) + l1_weight * mean(|s|) + tv_weight * TV(s)
where s is the image itself (pixel / fft) or the mask (mask).
TV is anisotropic: mean |horizontal neighbour diff| + mean |vertical diff|.
L1 alone favours isolated bright specks (nothing couples neighbours); TV
makes the surviving pixels form connected regions instead.
Only the InceptionV1 normalisation x * 255 - 117 follows the transform.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lucent.modelzoo import inceptionv1
from lucent.optvis import param
from lucent.optvis.param import color, spatial

from l1_on_feat_vis.feature_vis.activation_map import capture_layer, load_rgb, preprocess_inceptionv1, to_grayscale
from l1_on_feat_vis.feature_vis.sparse_visualize import l1_penalty, save_image

MAX_SHIFT_PX = 12
SCALES = [1 + (i - 5) / 50.0 for i in range(11)]
ANGLES_DEG = list(range(-10, 11)) + 5 * [0]


def total_variation(image):
    dx = (image[..., :, 1:] - image[..., :, :-1]).abs().mean()
    dy = (image[..., 1:, :] - image[..., :-1, :]).abs().mean()
    return dx + dy


def threshold_map(activation_map, threshold):
    return np.where(activation_map >= threshold, activation_map, 0.0).astype(np.float32)


def build_target(model, layer, channel, image_path, short_side, threshold, device):
    image = load_rgb(image_path, short_side)
    x = preprocess_inceptionv1(image).to(device)
    activation_map = capture_layer(model, layer, x)[0, channel].cpu().numpy()
    return threshold_map(activation_map, threshold), (image.height, image.width)


def preprocess_reference(image_path, short_side):
    pixels = np.asarray(load_rgb(image_path, short_side), dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)


def identity(x, sample=True):
    return x


def pixel_param(height, width, seed, device):
    generator = torch.Generator().manual_seed(seed)
    image = (0.5 + 0.01 * torch.randn(1, 3, height, width, generator=generator)).to(device).requires_grad_(True)

    def project_to_valid_range():
        with torch.no_grad():
            image.clamp_(0.0, 1.0)

    return [image], lambda: image, identity, project_to_valid_range


def pixel_param_from_image(image):
    image = image.clone().requires_grad_(True)

    def project_to_valid_range():
        with torch.no_grad():
            image.clamp_(0.0, 1.0)

    return [image], lambda: image, identity, project_to_valid_range


def mask_param_from_image(reference):
    mask = torch.ones(1, 1, *reference.shape[2:], device=reference.device, requires_grad=True)

    def project_to_valid_range():
        with torch.no_grad():
            mask.clamp_(0.0, 1.0)

    return [mask], lambda: mask, lambda m, sample=True: m * reference, project_to_valid_range


def straight_through_binary(logits):
    soft = torch.sigmoid(logits)
    hard = (soft > 0.5).float()  # comparison carries no gradient
    return hard + soft - soft.detach()


def mask_ste_param_from_image(reference, init_logit=1.0):
    logits = torch.full((1, 1, *reference.shape[2:]), init_logit, device=reference.device, requires_grad=True)
    return [logits], lambda: straight_through_binary(logits), lambda m, sample=True: m * reference, lambda: None


def straight_through_gumbel_sigmoid(prob, temperature):
    # Clamp so logit(prob) stays finite once a pixel's logit saturates.
    prob = prob.clamp(1e-6, 1 - 1e-6)
    u = torch.rand_like(prob).clamp(1e-6, 1 - 1e-6)
    logistic_noise = torch.log(u) - torch.log1p(-u)
    soft = torch.sigmoid((torch.log(prob) - torch.log1p(-prob) + logistic_noise) / temperature)
    hard = (soft > 0.5).float()
    return hard + soft - soft.detach()


def mask_gumbel_param_from_image(reference, init_logit=3.0, temperature=0.5):
    logits = torch.full((1, 1, *reference.shape[2:]), init_logit, device=reference.device, requires_grad=True)

    def render(prob, sample=True):
        mask = straight_through_gumbel_sigmoid(prob, temperature) if sample else (prob > 0.5).float()
        return mask * reference

    return [logits], lambda: torch.sigmoid(logits), render, lambda: None


def fft_param(height, width, seed, device=None):  # lucent places it on cuda:0 if available
    torch.manual_seed(seed)
    params, image_f = param.image(width, height)  # lucent's signature is (w, h)
    return params, image_f, identity, lambda: None  # sigmoid already keeps it in [0, 1]


def fft_spectrum_from_image(image):
    """Invert lucent's fft image pipeline so rendering starts at `image`.

    lucent renders: sigmoid(decorrelate(irfft(scale * spectrum) / 4)).
    Undo each step in reverse: logit -> inverse colour matrix -> * 4 ->
    rfft -> / scale. Pixels are clipped away from 0/1 so the logit is finite.
    """
    _, _, height, width = image.shape
    pre_sigmoid = torch.logit(image.clamp(1e-3, 1 - 1e-3))
    correlation = torch.from_numpy(color.color_correlation_normalized).to(image.device)
    pre_decorrelate = (pre_sigmoid.permute(0, 2, 3, 1) @ torch.linalg.inv(correlation.T)).permute(0, 3, 1, 2)
    spectrum = torch.fft.rfftn(pre_decorrelate * 4.0, s=(height, width), norm="ortho")
    if width % 2 == 1:  # lucent keeps one extra frequency column for odd widths
        spectrum = F.pad(spectrum, (0, 1))
    freqs = spatial.rfft2d_freqs(height, width)
    scale = torch.from_numpy(1.0 / np.maximum(freqs, 1.0 / max(width, height))).float().to(image.device)
    return torch.view_as_real(spectrum / scale)


def fft_param_from_image(image):
    params, sparse_f, render, project = fft_param(image.shape[2], image.shape[3], seed=0)
    with torch.no_grad():
        params[0].copy_(fft_spectrum_from_image(image))
    return params, sparse_f, render, project


def random_affine_theta(height, width):
    """Output->input mapping in normalised coords for one shift/scale/rotate.

    Rotation must happen in pixel space, so the normalised x/y axes (which
    have different pixel lengths for a non-square image) are rescaled around it.
    """
    angle = math.radians(random.choice(ANGLES_DEG))
    scale = random.choice(SCALES)
    shift_x = random.randint(-MAX_SHIFT_PX, MAX_SHIFT_PX) / (width / 2)
    shift_y = random.randint(-MAX_SHIFT_PX, MAX_SHIFT_PX) / (height / 2)
    aspect = height / width
    cos, sin = math.cos(angle) / scale, math.sin(angle) / scale
    return torch.tensor(
        [[[cos, -sin * aspect, shift_x], [sin / aspect, cos, shift_y]]], dtype=torch.float32
    )


def warp(tensor, theta, fill):
    grid = F.affine_grid(theta, tensor.shape, align_corners=False)
    return F.grid_sample(tensor - fill, grid, align_corners=False) + fill


def identity_theta():
    return torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])


class SpatialMapMatcher:
    def __init__(self, model, layer, channel, target, use_transforms, device):
        self.model = model
        self.channel = channel
        self.device = device
        self.target = torch.from_numpy(target)[None, None].to(device)
        self.use_transforms = use_transforms
        self.fine_tuning = False
        self.captured = {}
        getattr(model, layer).register_forward_hook(
            lambda module, inputs, output: self.captured.update(output=output)
        )

    def channel_map(self, image):
        self.model(image * 255 - 117)
        return self.captured["output"][:, self.channel : self.channel + 1]

    def sample_theta(self, image):
        if not self.use_transforms or self.fine_tuning:
            return identity_theta().to(self.device)
        return random_affine_theta(image.shape[2], image.shape[3]).to(self.device)

    def map_loss(self, image):
        theta = self.sample_theta(image)
        produced = self.channel_map(warp(image, theta, fill=0.5))
        return F.mse_loss(produced, warp(self.target, theta, fill=0.0))

    def run(self, params, sparse_f, render, project, steps, lr, l1_weight, log_every=64, trace_every=0,
            fine_tune_steps=0, tv_weight=0.0):
        """The last `fine_tune_steps` steps skip the random transforms, so the
        final fit is on the untransformed image that actually gets saved."""
        optimizer = torch.optim.Adam(params, lr=lr)
        self.trace = []
        for step in range(1, steps + 1):
            self.fine_tuning = step > steps - fine_tune_steps
            optimizer.zero_grad()
            sparse = sparse_f()
            image = render(sparse)  # one sparse tensor per step, so traced gradients reach it
            map_loss = self.map_loss(image)
            weighted_l1 = l1_weight * l1_penalty(sparse)
            weighted_tv = tv_weight * total_variation(sparse)
            if trace_every and (step % trace_every == 0 or step == 1):
                self.trace.append(trace_terms(step, sparse, map_loss, weighted_l1, weighted_tv))
            (map_loss + weighted_l1 + weighted_tv).backward()
            optimizer.step()
            project()
            if step % log_every == 0 or step == 1:
                print(f"step {step:4d}{' (no tf)' if self.fine_tuning else ''}  mse={map_loss.item():.1f}  weighted_l1={weighted_l1.item():.1f}  weighted_tv={weighted_tv.item():.1f}", flush=True)
        with torch.no_grad():
            sparse = sparse_f().detach()  # pixel param returns the leaf itself
            image = render(sparse, sample=False)
            return image.cpu(), self.channel_map(image)[0, 0].cpu().numpy(), sparse.cpu()


def trace_terms(step, image, map_loss, weighted_l1, weighted_tv):
    """Each term's gradient on the sparsified tensor, from the same forward pass.

    `image` is whatever L1 / TV act on: the image for pixel / fft, the mask
    for mask. These are the raw gradients before Adam rescales them, i.e. how
    hard each term pushes. Entries exactly at 0 get no L1 gradient, because
    d|x|/dx = sign(0) = 0 in torch.
    """
    map_grad, = torch.autograd.grad(map_loss, image, retain_graph=True)
    l1_grad, = torch.autograd.grad(weighted_l1, image, retain_graph=True)
    tv_grad = (
        torch.autograd.grad(weighted_tv, image, retain_graph=True)[0]
        if weighted_tv.requires_grad else torch.zeros_like(image)
    )
    return {
        "step": step,
        "mse": map_loss.item(),
        "weighted_l1": weighted_l1.item(),
        "mse_grad_norm": map_grad.norm().item(),
        "l1_grad_norm": l1_grad.norm().item(),
        "weighted_tv": weighted_tv.item(),
        "tv_grad_norm": tv_grad.norm().item(),
        "grad_cosine": F.cosine_similarity(map_grad.flatten(), l1_grad.flatten(), dim=0).item(),
        "grad_cosine_mse_tv": F.cosine_similarity(map_grad.flatten(), tv_grad.flatten(), dim=0).item(),
        "nonzero_pixel_frac": (image != 0).float().mean().item(),
    }


def save_map_pair(produced, target, path):
    """Side by side: target | produced, both scaled by the target's peak."""
    peak = target.max()
    pair = np.concatenate([target, np.full((target.shape[0], 1), peak), np.clip(produced, 0, peak)], axis=1)
    to_grayscale(pair).resize((pair.shape[1] * 16, pair.shape[0] * 16), resample=0).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--layer", default="mixed4e")
    parser.add_argument("--channel", type=int, default=55)
    parser.add_argument("--size", type=int, default=448, help="short side, as in activation_map.py")
    parser.add_argument("--threshold", type=float, default=200.0)
    parser.add_argument("--param", choices=["fft", "pixel", "mask", "mask-ste", "mask-gumbel"], default="fft",
                        help="mask always starts from the reference image")
    parser.add_argument("--transforms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init", choices=["noise", "reference"], default="noise",
                        help="reference: start the fft param at the reference image itself")
    parser.add_argument("--l1-weight", type=float, default=0.0)
    parser.add_argument("--fine-tune-steps", type=int, default=0,
                        help="run the last N steps without transforms")
    parser.add_argument("--trace-every", type=int, default=0,
                        help="record both terms' losses and image-gradient norms every N steps")
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("outputs/match_map"))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    device = torch.device(args.device)
    model = inceptionv1(pretrained=True).to(device).eval()
    target, (height, width) = build_target(
        model, args.layer, args.channel, args.reference, args.size, args.threshold, device
    )
    print(f"target {target.shape}: {(target > 0).sum()} of {target.size} cells >= {args.threshold:g}")

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.reference.stem}_{args.layer}_{args.channel}_thr{args.threshold:g}"
    np.save(args.out / f"{stem}_target.npy", target)
    to_grayscale(target).resize((width, height), resample=0).save(args.out / f"{stem}_target.png")

    if args.param.startswith("mask"):
        make_param = {
            "mask": mask_param_from_image,
            "mask-ste": mask_ste_param_from_image,
            "mask-gumbel": mask_gumbel_param_from_image,
        }[args.param]
        params, sparse_f, render, project = make_param(preprocess_reference(args.reference, args.size).to(device))
    elif args.init == "reference":
        make_param = fft_param_from_image if args.param == "fft" else pixel_param_from_image
        params, sparse_f, render, project = make_param(preprocess_reference(args.reference, args.size).to(device))
    else:
        make_param = fft_param if args.param == "fft" else pixel_param
        params, sparse_f, render, project = make_param(height, width, args.seed, device)
    matcher = SpatialMapMatcher(model, args.layer, args.channel, target, args.transforms, device)
    image, produced, sparse = matcher.run(
        params, sparse_f, render, project, args.steps, args.lr, args.l1_weight,
        trace_every=args.trace_every, fine_tune_steps=args.fine_tune_steps, tv_weight=args.tv_weight,
    )

    from_reference = args.init == "reference" or args.param.startswith("mask")
    regularisation = f"{args.param}{'_tf' if args.transforms else ''}{'_init-ref' if from_reference else ''}"
    run_stem = f"{stem}_{regularisation}_l1_{args.l1_weight:g}{f'_tv_{args.tv_weight:g}' if args.tv_weight else ''}{f'_lr{args.lr:g}' if args.lr != 0.05 else ''}{f'_ft{args.fine_tune_steps}' if args.fine_tune_steps else ''}"
    save_image(image, args.out / f"{run_stem}_image.png")
    np.save(args.out / f"{run_stem}_image.npy", image[0].numpy())
    if args.param.startswith("mask"):
        mask = (sparse > 0.5).float() if args.param == "mask-gumbel" else sparse  # gumbel: sparse is p
        np.save(args.out / f"{run_stem}_mask.npy", mask[0, 0].numpy())
        to_grayscale(mask[0, 0].numpy()).save(args.out / f"{run_stem}_mask.png")
        if args.param == "mask-gumbel":
            np.save(args.out / f"{run_stem}_keep_prob.npy", sparse[0, 0].numpy())
    if matcher.trace:
        (args.out / f"{run_stem}_trace.json").write_text(json.dumps(matcher.trace, indent=1))
    np.save(args.out / f"{run_stem}_produced.npy", produced)
    save_map_pair(produced, target, args.out / f"{run_stem}_maps.png")
    print(f"produced map (untransformed): max={produced.max():.1f}  "
          f"mse={np.mean((produced - target) ** 2):.1f}  saved {run_stem}")


if __name__ == "__main__":
    main()
