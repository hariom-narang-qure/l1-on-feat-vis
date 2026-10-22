# Sparse activation matching: the final process

As of 2026-10-21. Code: `src/l1_on_feat_vis/pipeline.py`. How we got here:
[experiment-log.md](experiment-log.md).

**What it does.** Given a neuron (`layer:channel` of lucent's InceptionV1) and
a set of images, it finds, for each image, a sparse version of that image
(most pixels exactly 0) that still makes the neuron produce the image's own
strong activations. It runs 20 seeds per image and keeps their average.

## Run it

```
uv run python -m l1_on_feat_vis.pipeline data/cats \
    --neuron mixed4e:55 --threshold 600 --seeds 20 --out outputs/pipeline
```

Inputs are image files or folders of images (`.jpg`, `.jpeg`, `.png`, `.webp`).
It uses the GPU when one is available (`--device` to override). Every setting
in the table below is a flag of the same name (`--l1-weight`, `--steps`, …).

## How it works, per image

1. **Load.** Resize so the short side is 448 px, keep the aspect ratio, scale
   RGB to [0, 1].
2. **Target.** Forward the image once. The network input is `x * 255 - 117`
   (the original TF InceptionV1 range). Read the neuron's post-ReLU spatial map
   (37 × 28 for a 448 × 597 image at mixed4e, one cell per 16 px). Set every
   cell below the threshold to 0. If no cell reaches the threshold, the image
   is skipped.
3. **Optimise.** Make one copy of the image per seed. Each copy is a raw RGB
   pixel tensor, initialised to the image itself. For each step and each seed:
   - Draw a random affine: shift up to ±12 px, scale 0.9–1.1, rotate ±10°.
     Warp the image with it (filling with grey) and warp the target map with
     the same normalised affine (filling with 0), so the cells stay aligned.
   - Loss = `MSE(map(warped image), warped target) + l1 · mean|x| + tv · TV(x)`.
     TV is the mean absolute difference between horizontal neighbours plus
     vertical neighbours.
   - Adam step, then clamp pixels to [0, 1].
   - The last 96 steps skip the affine, so the final fit is on the saved image.
4. **Average.** Average the 20 optimised images pixel by pixel.

All seeds run as one batch. The per-seed losses are summed, and Adam updates
each element on its own, so the batch equals 20 separate runs. Seed *s* draws
its transforms from `random.Random(s)`.

## Why each part is there

| Part | Without it |
|---|---|
| Start from the image | From noise, the optimiser invents patterns; from the image it only has to remove things |
| Raw pixels, clamped | lucent's FFT param uses a sigmoid; under L1, saturated pixels get stuck as white blobs |
| L1 on pixels | No pressure to go to 0 |
| TV on pixels | L1 alone leaves isolated, single-pixel coloured specks |
| Random affine on image and target | Pure colour-noise solutions that exactly match the map |
| Transform-free last 96 steps | The map fit jumps step to step; final MSE ~50× higher |
| lr 0.01 | At 0.05 the fit oscillates at the end |
| 20 seeds, averaged | Each seed reaches a near-perfect map fit partly through seed-specific pixel edits that suppress neighbouring cells; averaging cancels those and keeps the shared, real image structure |

## Settings

| Setting | Default | Flag |
|---|---|---|
| Neuron | `mixed4e:55` | `--neuron` |
| Threshold | 600 | `--threshold` |
| Short side | 448 px | `--size` |
| L1 weight | 3e5 | `--l1-weight` |
| TV weight | 1e5 | `--tv-weight` |
| Learning rate | 0.01 | `--lr` |
| Steps | 768 | `--steps` |
| Transform-free steps at the end | 96 | `--fine-tune-steps` |
| Seeds | 20 | `--seeds` |

The weights are large because L1 and TV are means over all pixels. They were
tuned for `mixed4e:55`, whose peak on the cat photo is ~985; another neuron
with a different activation scale needs them re-tuned.

## Output

```
<out>/
  <image>.png                      seed-averaged image — the only files at the top level
  intermediates/
    run_summary.json               config + per-image results, including skipped images
    <image>/
      input.png                    the resized input
      activation_map.npy           the neuron's full map on the input
      target.npy, target.png       the thresholded map
      seeds/seed_XX.png            each seed's final image
      seeds/seed_XX_maps.png       target | that seed's map
      mean_image.npy               the averaged image, float
      mean_image_maps.png          target | the averaged image's map
      std_across_seeds.png         where seeds disagree (per-pixel std, max over RGB)
      summary.json                 per-seed map MSE, peak, sparsity; mean-image stats
```

## Reading the results

- **Per-seed map MSE** should be near 0 (0.5–8.5 on the original cat). It shows
  the optimisation worked.
- **Mean-image map MSE** is much higher (~1900 on the original cat). That is
  expected: the averaged image is the real content, and real content drives the
  cells around the core too. See "The halo is real" in the experiment log.
- **`std_across_seeds.png`** shows which parts are stable (dark) and which are
  seed-specific (bright).

## Known limits

- Flecks appear in regions far from the target, often at the same places in
  every seed.
- The weights are tuned for one neuron and image scale (see above).
- A threshold is absolute, so images whose peak is below it are skipped.
