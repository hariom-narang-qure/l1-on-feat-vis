# Experiment log: sparse images that reproduce a neuron's spatial activation map

As of 2026-10-21.

**Goal.** Find the sparsest image (most pixels exactly 0) that still makes one
InceptionV1 channel produce a given spatial activation map. Model: lucent's
InceptionV1 (`inception5h`, TF port, input `x * 255 - 117`). Neuron:
`mixed4e:55`, a cat-face channel. Reference image: a cat photo (`cat.jpg`,
resized to a short side of 448 px, giving a 37 × 28 map at mixed4e, stride 16).

**Final result.** Optimise raw pixels starting from the photo, matching a
thresholded map (cells < 600 set to 0) with MSE, plus L1 and TV on the image,
lucent-style transforms that are mirrored onto the target, and a transform-free
fine-tune. Average 8–20 seeds. Each seed fits the map almost exactly
(MSE 0.5–8.5); the seed average is a clean, sparse cat face.

## Chain of experiments

Weights below are for the loss `MSE(map, target) + l1 · mean|x| + tv · TV(x)`,
where `mean` is over all pixels (so weights are large).

| # | What changed | Result | Why / what we learned |
|---|---|---|---|
| 1 | Lucent feature vis of `mixed4e:55` (FFT param, standard transforms, 512 steps) | Tiled cat faces everywhere | Confirms it is a cat-face channel |
| 2 | Raw pixels in [0, 1], L1 on the image, normalisation kept in the graph, maximise the channel | L1 = 10 no visible effect; wonky borders, high-contrast colour | Raw pixels have no smoothness prior; edge pixels get fewer gradients under jitter |
| 3 | Activation map of the cat photo, flattened + sorted | Peak 985, median 4.6, 46% exact zeros; top 1% of cells hold 15% of total | Chose threshold 200, later 600 (13 core cells); natural break at ~440 |
| 4 | Objective → MSE between produced map and thresholded target; raw pixels from noise, no transforms | MSE → 0.0, image is pure colour noise | Adversarial solution: exact activations, no visible structure |
| 5 | Lucent FFT + decorrelation param, jitter/scale/rotate applied to image **and** target with one affine | Stylised cat face at the hotspot, grey background, MSE 182 | Standard regularisers restore structure; aligned transforms keep the per-cell MSE meaningful |
| 6 | Start from the cat photo (inverted lucent's FFT pipeline exactly), FFT + L1 1e4–1e6 | Sparse face, but solid white blobs at 1e5+ | L1 acts after the sigmoid; saturated pixels get no gradient and get stuck |
| 7 | Raw pixels from the cat (clamped, no sigmoid), L1 1e6–1e8 | Blobs gone; 76–98% exact zeros; specks of coloured noise | No saturation, but L1 favours isolated bright specks |
| 8 | Logged both terms' gradients on the image every 32 steps | Weighted L1 value 20–40× MSE, but MSE gradient 3–9× L1's; cosine ≈ 0 | L1 gradient is a constant `w / N` per lit pixel; the two terms act on mostly different pixels |
| 9 | Threshold 600, last 64 steps without transforms, L1 3e5 | Map MSE 1972 → 38, peak back to 993 | Transforms make the MSE jump; fitting the untransformed image at the end fixes it |
| 10 | Add TV, lr 0.05 → 0.01, 768 steps (last 96 transform-free) | TV 1e5: face clearly visible, map MSE 0.5, 73% exact zeros | TV turns specks into connected strokes; lower lr removes end-of-run oscillation |
| 11 | Soft mask `m · cat`, L1 + TV on m | Mask settles on partial values (0.25–0.7), map MSE 185–889 | Nothing pushes m to 0 or 1: 0.3 costs less L1 than 1.0 and keeps most of the activation |
| 12 | Straight-through binary mask (forward hard 0/1, backward sigmoid gradient) | Exactly binary, true colours, but salt-and-pepper scatter; MSE 222–1049 | Biased gradient; isolated pixels flicker instead of clearing |
| 13 | Gumbel-sigmoid mask (straight-through, τ = 0.5), L1 + TV on keep probabilities | Clean, connected cut-outs of the real photo; MSE 755–1691 | Noise drives logits to confident 0 / 1; MSE floor explained below |
| 14 | Experiment 10's config, 8 seeds, averaged | Every seed MSE 0.5–8.5; mean image is a clean face, but its map has MSE 1878 | Each seed suppresses the halo with its own pixel edits, which average out |

## Key findings

- **The halo is real.** Map cells are 16 px apart but each sees well over
  100 px, so a real cat face drives the core cells *and* their neighbours
  (the 200–600 halo). Selecting real pixels (masks, exp. 11–13) can never give
  a 13-cell spike with nothing around it.
- **Low MSE with free pixels means adversarial edits.** Raw-pixel runs reach
  MSE ≈ 0 by adding seed-specific patterns that suppress the neighbouring cells.
  Seed averaging separates the two parts: the shared cat-face structure stays,
  the suppression averages out (exp. 14).
- **Gradient balance, not loss values, decides the outcome.** Adam normalises
  per pixel, so whichever term dominates a pixel's gradient decides it:
  L1 wins on the background, MSE wins near the face.
- **CPU runs need few threads.** torch's default 56 threads was ~50× slower per
  step than 8 on this small model.

## Final pipeline

Experiment 10's configuration plus seed averaging, in
`src/l1_on_feat_vis/pipeline.py`. How it works and how to run it:
[final-process.md](final-process.md).

## Results on 10 cat images

- **Run settings:** 448 px, 768 steps (last 96 without transforms), 20 seeds.
- **Threshold:** 0.7 × each image's own peak.
  - An absolute 600 skipped 3 of the first 5 photos, whose peaks were 331–599.
- **Per-seed fit:** map MSE 0.6–7 on every image.
- **Sparsity:** 60–76% of pixels exactly 0, and mean pixel value 0.001–0.05.
- **What remains:** most results keep only the face.
  - The two dark cats (`cat_05`, `cat_08`) nearly vanish, because their fur is already near 0.
  - A green tint remains in the kept fur.
- **A faster run did not hold up:** 336 px, 384 steps and 10 seeds left 3–50× more of each image. It is kept in `outputs/pipeline_mixed4e_55_fast336/`.
- **Outputs:** `outputs/pipeline_mixed4e_55/`, with an overview at `intermediates/overview.png`.

## Open questions

- Should the target ignore cells outside the core instead of forcing them to 0,
  so real-pixel selection can match it?
- Can the flecks at the bottom of the image be removed without losing the fit?
