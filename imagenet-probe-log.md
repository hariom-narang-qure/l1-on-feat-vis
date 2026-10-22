# Random conv nets on Imagenette / Imagewoof: does depth or more nets give semantic features? (2026-10-22)

Question: the fissure work found random conv nets far behind pretrained ones on a thin-line target. Do they
do better on a semantic target (object class), with deeper nets or with more nets, in a low-data regime?
Prediction going in: deeper is better, and classification is easier than fissure lines.

**Short answer: neither prediction holds for random nets.**

- **More nets help steadily**, but slowly: about +2.4 points per doubling, from 1 to 40 nets.
- **Depth helps only up to 4–6 blocks.** After that accuracy falls, and features from deeper blocks get
  steadily worse, because plain random ReLU stacks collapse all images towards one direction. Residual wiring
  on the same weights slows the collapse but does not turn depth into a gain.
- **Classification is not easier for random features.**
  - Imagenette, at 100 shots: 68% for the best random setup, against 99% for DINOv2.
  - Imagewoof, the fine-grained version: 35% against 96%, with 10% chance.
  - An out-of-domain pretrained net (RAD-DINO, chest X-rays) beats every random setup.

## Setup

| Item | Choice |
|---|---|
| Data | fastai Imagenette and Imagewoof (320 px releases), 10 classes each; short side resized to 224, centre crop 224 (`imagenet_probe/data.py`) |
| Low-data regime | each probe trains on **10, 25 or 100 images per class**, drawn from a fixed pool of 200 per class; scored on the **full val split** (3925 / 3929 images); 3 replicates |
| Random nets | `fissure_probe.models.random_convs.random_net`: conv 3×3 → ReLU, He-normal init, zero bias, strides 1, 2, 2, 2, 2 then 1 (224 → 14 px); ImageNet mean / std input |
| Controlled depth | every net is built at depth 40; the depth-D net is its first D blocks (the same seed gives the same weights), so depths differ only by the blocks they add |
| Features | per block and channel, the global mean and the global max over the image |
| Replicate r | nets with seeds 40r … 40r+N−1 (nested within a replicate as N grows) and its own training draw |
| Residual control | the same conv weights, wired as pre-activation residual steps `x + 0.5·conv(relu(x))` (rescaled) after the 5 downsampling blocks (`features.py`) |
| References | 16×16 pixels; DINOv2-S/B/L (CLS + mean patch); RAD-DINO; ResNet-50 a1_in1k (trained on IN-1k labels, so a ceiling) |

Readouts (`imagenet_probe/probes.py`); every choice is made on the training subset only:

- **linear:** ridge classifier on one-hot targets over all features, kernel form, ridge chosen by 5-fold CV.
- **select64 / select256:** the fissure-style unit screen, then the linear probe on the kept units. Units are
  ranked by fold-min ANOVA F and kept only if \|r\| ≤ 0.9 to every unit already kept.
- **mlp:** a trained head, 512 hidden units with dropout 0.5; weight decay chosen by CV.

Grid: 42 rows × 2 datasets × 3 training sizes × 3 replicates = 756 results (`imagenet_probe/run.py`). The rows:

- **nets:** depth 6 and depth 20, N = 1 to 40;
- **depth:** N = 10, depth 1 to 40, all blocks or the last block only;
- **width:** 16 against 64 channels;
- **residual:** depth 10, 20 and 40;
- **references.**

## Results (linear readout, val top-1 %, mean ± sd over 3 replicates)

### Number of nets: steady gain, about +2.4 points per doubling (depth 6, width 64, all blocks)

| Nets | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| 1 | 34.3 ± 0.9 | 44.6 ± 0.5 | 56.4 ± 0.3 | 18.2 ± 0.1 | 22.4 ± 0.8 | 27.9 ± 0.8 |
| 2 | 37.8 ± 1.2 | 46.7 ± 0.7 | 60.0 ± 0.8 | 18.1 ± 1.7 | 24.3 ± 0.7 | 30.2 ± 0.9 |
| 5 | 41.2 ± 1.1 | 51.2 ± 0.8 | 62.6 ± 1.4 | 20.1 ± 0.9 | 25.1 ± 0.4 | 31.2 ± 0.8 |
| 10 | 40.1 ± 1.6 | 53.0 ± 1.0 | 64.7 ± 0.2 | 18.0 ± 1.5 | 25.9 ± 0.3 | 32.7 ± 0.4 |
| 20 | 43.2 ± 1.5 | 54.7 ± 0.2 | 66.8 ± 0.3 | 18.7 ± 2.3 | 26.9 ± 0.6 | 33.8 ± 0.7 |
| 40 | 43.2 ± 1.4 | 55.9 ± 0.4 | **68.2 ± 0.6** | 18.7 ± 2.3 | 27.2 ± 0.5 | **34.6 ± 0.5** |

- At depth 20 the curve has the same shape, about 3 points lower (Imagenette 100: 52.8 → 65.4).
- At 10 images per class on Imagewoof, nothing moves: every all-blocks random setup sits at 17–21%, against 10% chance.

### Depth (N = 10, width 64): best at 4–6 blocks; deeper blocks are worse on their own

| Depth | all blocks, Imagenette 10 | all, Imagenette 100 | all, Imagewoof 100 | last block only, Imagenette 100 | last only, Imagewoof 100 |
|---|---|---|---|---|---|
| 1 | 33.6 ± 0.9 | 54.7 ± 0.3 | 26.6 ± 0.2 | 54.7 ± 0.3 | 26.6 ± 0.2 |
| 2 | 38.3 ± 2.4 | 60.7 ± 0.4 | 29.2 ± 0.4 | 59.7 ± 0.6 | 29.5 ± 0.4 |
| 3 | 41.2 ± 0.8 | 62.5 ± 0.6 | 30.8 ± 0.3 | 59.9 ± 0.8 | 29.2 ± 0.2 |
| 4 | 41.7 ± 1.2 | 63.7 ± 0.8 | 31.6 ± 0.4 | 57.3 ± 0.5 | 28.9 ± 0.5 |
| 6 | 40.1 ± 1.6 | **64.7 ± 0.2** | **32.7 ± 0.4** | 49.1 ± 0.6 | 26.5 ± 0.4 |
| 10 | 39.0 ± 1.6 | 64.3 ± 0.3 | 32.4 ± 0.6 | 45.3 ± 0.2 | 25.0 ± 0.8 |
| 20 | 36.4 ± 1.5 | 62.3 ± 0.4 | 32.5 ± 0.6 | 37.9 ± 0.9 | 23.2 ± 0.8 |
| 40 | 35.4 ± 1.8 | 60.1 ± 0.6 | 31.3 ± 0.8 | 30.3 ± 1.3 | 20.9 ± 0.4 |

- **Last block only.** These rows hold the feature count fixed at 128 per net. The best block is block 2–3,
  and block 40 is only just above pixels on Imagenette (30.3 against 33.5).
- **All blocks.** Deep blocks add little and dilute the probe. The selected units say the same: for depth 20,
  72% of the units kept by select256 come from blocks 1–6.

**Why depth fails here: collapse.** The table below is the depth diagnostic for Imagenette, 10 nets.
Effective dimensions are computed on the standardised features (the participation ratio of the correlation
eigenvalues).

| Block | plain: mean cosine between images | plain: effective dims of 128 | residual: mean cosine | residual: effective dims |
|---|---|---|---|---|
| 1 | 0.969 | 6.0 | 0.969 | 6.0 |
| 4 | 0.969 | 9.0 | 0.969 | 9.0 |
| 6 | 0.969 | 6.6 | 0.968 | 5.6 |
| 10 | 0.980 | 4.1 | 0.972 | 4.5 |
| 20 | 0.989 | 2.4 | 0.977 | 3.7 |
| 40 | 0.994 | 1.7 | 0.982 | 2.8 |

Plain ReLU stacks at init send every input towards one direction as depth grows (Poole et al. 2016;
Schoenholz et al. 2017), and that's what happens here.

**The residual control** uses the same conv weights. Its residual steps start after block 5, so depth ≤ 6 is
identical to plain.

| Depth | Readout input | Plain, Imagenette 100 | Residual, Imagenette 100 | Plain, Imagewoof 100 | Residual, Imagewoof 100 |
|---|---|---|---|---|---|
| 10 | last block | 45.3 ± 0.2 | 48.3 ± 1.5 | 25.0 ± 0.8 | 26.3 ± 0.6 |
| 20 | last block | 37.9 ± 0.9 | 47.1 ± 0.8 | 23.2 ± 0.8 | 26.4 ± 0.1 |
| 40 | last block | 30.3 ± 1.3 | **44.2 ± 1.5** | 20.9 ± 0.4 | 25.2 ± 0.7 |
| 10 | all blocks | 64.3 ± 0.3 | 63.7 ± 0.7 | 32.4 ± 0.6 | 32.0 ± 0.6 |
| 20 | all blocks | 62.3 ± 0.4 | 63.2 ± 0.7 | 32.5 ± 0.6 | 32.2 ± 0.3 |
| 40 | all blocks | 60.1 ± 0.6 | 61.4 ± 1.3 | 31.3 ± 0.8 | 31.2 ± 0.3 |

- **Last block.** Better propagation recovers most of the loss (block 40: 30.3 → 44.2).
- **All blocks.** Even well-propagated deep blocks are still no better than block 4–6, and no residual row
  beats plain depth 6 (64.7 / 32.7).
- **Conclusion.** Composing more random layers does not build more useful (semantic) features. The gain from
  depth stops once the receptive field and the multi-scale pooling saturate, at about 4–6 blocks.

A first residual version, post-activation `x + relu(conv(x))`, collapsed faster than plain: 1.1 effective
dimensions at block 40. Its branch only ever adds non-negative values, so it was replaced before any probe ran.

### More nets vs more depth vs more width, at matched feature counts (Imagenette 100 / Imagewoof 100)

| Row | Features | Imagenette 100 | Imagewoof 100 |
|---|---|---|---|
| w64, depth 6, N10 | 7,680 | 64.7 ± 0.2 | 32.7 ± 0.4 |
| w16, depth 6, N40 | 7,680 | 65.4 ± 0.4 | 32.9 ± 0.5 |
| w16, depth 6, N10 | 1,920 | 62.4 ± 0.3 | 31.2 ± 0.6 |
| w64, depth 20, N10 | 25,600 | 62.3 ± 0.4 | 32.5 ± 0.6 |
| w64, depth 6, N40 | 30,720 | **68.2 ± 0.6** | **34.6 ± 0.5** |

- At equal feature count, many narrow nets match a few wide ones.
- Spending the features on more nets beats spending them on depth: 68.2 against 62.3, with 20% more features.

### References: random features stay far from pretrained ones, especially on fine-grained classes

| Features | Imagenette 10 | Imagenette 100 | Imagewoof 10 | Imagewoof 100 |
|---|---|---|---|---|
| pixels 16×16 | 22.8 ± 1.1 | 33.5 ± 0.4 | 16.9 ± 0.8 | 22.4 ± 0.5 |
| best random (w64, depth 6, N40) | 43.2 ± 1.4 | 68.2 ± 0.6 | 18.7 ± 2.3 | 34.6 ± 0.5 |
| RAD-DINO (chest X-ray pretraining) | 47.9 ± 1.5 | 70.1 ± 0.3 | 24.1 ± 0.8 | 41.4 ± 0.7 |
| DINOv2-S | 98.1 ± 0.1 | 99.0 ± 0.2 | 91.0 ± 0.1 | 93.9 ± 0.3 |
| DINOv2-B | 99.2 ± 0.1 | 99.4 ± 0.1 | 93.0 ± 0.4 | 95.2 ± 0.1 |
| DINOv2-L | 99.5 ± 0.1 | 99.7 ± 0.0 | 95.5 ± 0.2 | 96.2 ± 0.2 |
| ResNet-50 IN-1k (label ceiling) | 97.2 ± 0.9 | 99.2 ± 0.2 | 93.9 ± 0.3 | 96.2 ± 0.1 |

**The share of the gap above chance that the best random setup closes**, (random − 10) / (DINOv2-L − 10):

- Imagenette: 65% at 100 shots, 37% at 10 shots;
- Imagewoof: 28% at 100 shots, 10% at 10 shots;
- fissures, for comparison: ~50%, from tolerant Dice 0.35–0.44 against 0.77.

So classification is not easier than fissures for random features. Imagenette classes partly differ by
colour, texture and scene statistics, which random filters do capture. Dog breeds need semantic features that
random filters don't produce, and without data to fit them the 10-shot rows stay at chance + 8 points.

### Readouts: the unit screen hurts random features; the MLP head ≈ the linear probe

| Row | Readout | Imagenette 10 | Imagenette 100 | Imagewoof 10 | Imagewoof 100 |
|---|---|---|---|---|---|
| w64, depth 6, N10 | linear | 40.1 ± 1.6 | 64.7 ± 0.2 | 18.0 ± 1.5 | 32.7 ± 0.4 |
| | select64 | 30.4 ± 1.1 | 46.7 ± 1.6 | 15.7 ± 1.5 | 25.4 ± 0.7 |
| | select256 | 35.1 ± 1.5 | 53.7 ± 1.1 | 18.9 ± 1.3 | 28.0 ± 0.6 |
| | mlp | 33.6 ± 2.5 | 65.1 ± 0.7 | 17.5 ± 0.6 | 33.6 ± 0.2 |
| DINOv2-B | linear | 99.2 ± 0.1 | 99.4 ± 0.1 | 93.0 ± 0.4 | 95.2 ± 0.1 |
| | select64 | 97.4 ± 0.5 | 98.1 ± 0.1 | 92.4 ± 0.4 | 93.8 ± 0.2 |
| | select256 | 98.7 ± 0.4 | 99.2 ± 0.0 | 93.6 ± 0.1 | 94.9 ± 0.1 |
| | mlp | 99.2 ± 0.1 | 99.5 ± 0.0 | 93.3 ± 0.1 | 95.5 ± 0.2 |

- **Random features: the screen loses 11–18 points on Imagenette and 5–7 on Imagewoof at 100 shots.** Their class information is spread thinly over many
  weak units, which a per-unit screen on ≤ 1000 images can't rank.
- **DINOv2: the screen loses almost nothing.** Single DINOv2 units are already class-selective, so 256 units
  lose ≤ 0.5 points.
- **This matches the fissure finding.** Selecting neurons pays off when individual units carry the concept,
  which pretraining provides and random init does not.
- **MLP head.** It matches the linear probe at 100 shots and is worse at 10 shots, so the readout is not the
  bottleneck.

## Conclusions

1. **More random nets give a reliable but slow gain**, about +2.4 points per doubling at 100 shots. Read from
   these 6 points as a log-linear trend, matching DINOv2 on Imagenette would take far more than 1000 nets.
   With 10 shots on fine-grained classes, more nets give nothing.
2. **Deeper random nets are not better.** Accuracy peaks at 4–6 blocks.
   - **Plain nets:** deeper blocks collapse (effective dimensions 9 → 1.7).
   - **Residual nets:** they fix most of the collapse but still add no useful features beyond block 6.
   - The depth that helped pretrained nets comes from training, not from architecture.
3. **Semantic classification is harder than thin lines for random features.** They close 65% of the gap above
   chance on Imagenette but 28% on Imagewoof. Out-of-domain pretraining (RAD-DINO) already beats every random
   setup.
4. **Selecting units helps only when units are individually meaningful** (pretrained). For random features,
   the gradient-trained readout over all units is clearly better.

## Reproduce

Run everything as `python -m l1_on_feat_vis.imagenet_probe.<module>`. Data and features are under
`/realdata/hariom.narang/imagenet_subsets/`; results are in `outputs/imagenet_probe/results_*.jsonl`.

| Step | Command | Time |
|---|---|---|
| decode | `data` | 1 min |
| random features | `features random --width 64 --seeds 0-119`; also `--width 16`, and `--arch res` | ~3 s per net |
| reference features | `features refs` | 5 min |
| grid | `run --worker I --workers 4` (2 per GPU) | ~45 min |
| tables | `report > outputs/imagenet_probe/tables.md` | 1 min |

## Appendix: full tables (every readout, every row)

Generated by `report` (also written to `outputs/imagenet_probe/tables.md`); the headline tables above are copied from it.

## Readout: linear


| nets (w64, depth 6, all blocks) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| N1 | 34.3 ± 0.9 | 44.6 ± 0.5 | 56.4 ± 0.3 | 18.2 ± 0.1 | 22.4 ± 0.8 | 27.9 ± 0.8 |
| N2 | 37.8 ± 1.2 | 46.7 ± 0.7 | 60.0 ± 0.8 | 18.1 ± 1.7 | 24.3 ± 0.7 | 30.2 ± 0.9 |
| N5 | 41.2 ± 1.1 | 51.2 ± 0.8 | 62.6 ± 1.4 | 20.1 ± 0.9 | 25.1 ± 0.4 | 31.2 ± 0.8 |
| N10 | 40.1 ± 1.6 | 53.0 ± 1.0 | 64.7 ± 0.2 | 18.0 ± 1.5 | 25.9 ± 0.3 | 32.7 ± 0.4 |
| N20 | 43.2 ± 1.5 | 54.7 ± 0.2 | 66.8 ± 0.3 | 18.7 ± 2.3 | 26.9 ± 0.6 | 33.8 ± 0.7 |
| N40 | 43.2 ± 1.4 | 55.9 ± 0.4 | 68.2 ± 0.6 | 18.7 ± 2.3 | 27.2 ± 0.5 | 34.6 ± 0.5 |

| nets (w64, depth 20, all blocks) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| N1 | 33.2 ± 0.6 | 41.0 ± 1.6 | 52.8 ± 1.5 | 18.0 ± 1.6 | 21.5 ± 0.5 | 27.6 ± 0.4 |
| N2 | 31.0 ± 2.7 | 45.2 ± 1.0 | 55.6 ± 1.7 | 19.8 ± 0.8 | 23.5 ± 0.4 | 29.3 ± 1.8 |
| N5 | 33.7 ± 4.9 | 48.2 ± 1.3 | 59.9 ± 0.9 | 20.3 ± 1.3 | 24.4 ± 0.4 | 30.7 ± 1.0 |
| N10 | 36.4 ± 1.5 | 49.5 ± 0.9 | 62.3 ± 0.4 | 20.5 ± 1.4 | 24.8 ± 0.4 | 32.5 ± 0.6 |
| N20 | 36.7 ± 1.5 | 50.2 ± 0.9 | 63.8 ± 0.4 | 20.6 ± 1.2 | 25.7 ± 0.2 | 32.6 ± 0.3 |
| N40 | 35.6 ± 0.8 | 50.9 ± 1.3 | 65.4 ± 0.2 | 19.5 ± 2.7 | 26.1 ± 0.3 | 33.9 ± 0.4 |

| depth (w64, N10, all blocks) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| d1 | 33.6 ± 0.9 | 44.4 ± 1.8 | 54.7 ± 0.3 | 17.2 ± 0.3 | 21.2 ± 0.5 | 26.6 ± 0.2 |
| d2 | 38.3 ± 2.4 | 50.6 ± 0.6 | 60.7 ± 0.4 | 18.4 ± 0.8 | 23.5 ± 2.1 | 29.2 ± 0.4 |
| d3 | 41.2 ± 0.8 | 52.5 ± 0.3 | 62.5 ± 0.6 | 18.8 ± 1.0 | 25.2 ± 0.7 | 30.8 ± 0.3 |
| d4 | 41.7 ± 1.2 | 53.2 ± 0.5 | 63.7 ± 0.8 | 18.3 ± 2.1 | 25.4 ± 0.9 | 31.6 ± 0.4 |
| d6 | 40.1 ± 1.6 | 53.0 ± 1.0 | 64.7 ± 0.2 | 18.0 ± 1.5 | 25.9 ± 0.3 | 32.7 ± 0.4 |
| d10 | 39.0 ± 1.6 | 51.9 ± 0.6 | 64.3 ± 0.3 | 17.8 ± 2.8 | 25.6 ± 0.8 | 32.4 ± 0.6 |
| d20 | 36.4 ± 1.5 | 49.5 ± 0.9 | 62.3 ± 0.4 | 20.5 ± 1.4 | 24.8 ± 0.4 | 32.5 ± 0.6 |
| d40 | 35.4 ± 1.8 | 46.8 ± 0.9 | 60.1 ± 0.6 | 18.9 ± 1.5 | 24.3 ± 0.4 | 31.3 ± 0.8 |

| depth (w64, N10, last blocks) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| d1 | 33.6 ± 0.9 | 44.4 ± 1.8 | 54.7 ± 0.3 | 17.2 ± 0.3 | 21.2 ± 0.5 | 26.6 ± 0.2 |
| d2 | 37.8 ± 0.6 | 48.5 ± 0.5 | 59.7 ± 0.6 | 18.1 ± 0.8 | 23.2 ± 1.2 | 29.5 ± 0.4 |
| d3 | 38.0 ± 2.0 | 46.4 ± 0.6 | 59.9 ± 0.8 | 17.7 ± 0.4 | 22.8 ± 0.7 | 29.2 ± 0.2 |
| d4 | 36.6 ± 2.3 | 44.3 ± 1.9 | 57.3 ± 0.5 | 17.6 ± 1.2 | 23.5 ± 0.5 | 28.9 ± 0.5 |
| d6 | 32.3 ± 0.4 | 39.9 ± 0.2 | 49.1 ± 0.6 | 16.5 ± 1.0 | 22.3 ± 0.7 | 26.5 ± 0.4 |
| d10 | 27.1 ± 1.7 | 36.2 ± 1.5 | 45.3 ± 0.2 | 17.9 ± 0.9 | 21.4 ± 1.0 | 25.0 ± 0.8 |
| d20 | 24.3 ± 1.0 | 29.8 ± 1.0 | 37.9 ± 0.9 | 16.9 ± 1.7 | 20.5 ± 0.9 | 23.2 ± 0.8 |
| d40 | 21.7 ± 1.2 | 24.8 ± 2.4 | 30.3 ± 1.3 | 15.7 ± 1.9 | 19.0 ± 0.5 | 20.9 ± 0.4 |

| depth, residual (w64, N10) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| w64 d6 all | 40.1 ± 1.6 | 53.0 ± 1.0 | 64.7 ± 0.2 | 18.0 ± 1.5 | 25.9 ± 0.3 | 32.7 ± 0.4 |
| w64 d10 all | 39.0 ± 1.6 | 51.9 ± 0.6 | 64.3 ± 0.3 | 17.8 ± 2.8 | 25.6 ± 0.8 | 32.4 ± 0.6 |
| res w64 d10 all | 41.5 ± 1.2 | 51.8 ± 0.9 | 63.7 ± 0.7 | 18.1 ± 1.6 | 25.4 ± 0.5 | 32.0 ± 0.6 |
| w64 d20 all | 36.4 ± 1.5 | 49.5 ± 0.9 | 62.3 ± 0.4 | 20.5 ± 1.4 | 24.8 ± 0.4 | 32.5 ± 0.6 |
| res w64 d20 all | 39.7 ± 2.2 | 50.7 ± 1.2 | 63.2 ± 0.7 | 17.2 ± 1.1 | 24.2 ± 0.8 | 32.2 ± 0.3 |
| w64 d40 all | 35.4 ± 1.8 | 46.8 ± 0.9 | 60.1 ± 0.6 | 18.9 ± 1.5 | 24.3 ± 0.4 | 31.3 ± 0.8 |
| res w64 d40 all | 36.2 ± 3.1 | 48.2 ± 0.8 | 61.4 ± 1.3 | 20.2 ± 0.9 | 24.6 ± 0.6 | 31.2 ± 0.3 |
| w64 d6 last | 32.3 ± 0.4 | 39.9 ± 0.2 | 49.1 ± 0.6 | 16.5 ± 1.0 | 22.3 ± 0.7 | 26.5 ± 0.4 |
| w64 d10 last | 27.1 ± 1.7 | 36.2 ± 1.5 | 45.3 ± 0.2 | 17.9 ± 0.9 | 21.4 ± 1.0 | 25.0 ± 0.8 |
| res w64 d10 last | 31.5 ± 0.9 | 39.0 ± 0.4 | 48.3 ± 1.5 | 16.7 ± 0.4 | 22.5 ± 0.7 | 26.3 ± 0.6 |
| w64 d20 last | 24.3 ± 1.0 | 29.8 ± 1.0 | 37.9 ± 0.9 | 16.9 ± 1.7 | 20.5 ± 0.9 | 23.2 ± 0.8 |
| res w64 d20 last | 30.3 ± 0.7 | 37.4 ± 0.4 | 47.1 ± 0.8 | 17.4 ± 1.2 | 21.8 ± 0.8 | 26.4 ± 0.1 |
| w64 d40 last | 21.7 ± 1.2 | 24.8 ± 2.4 | 30.3 ± 1.3 | 15.7 ± 1.9 | 19.0 ± 0.5 | 20.9 ± 0.4 |
| res w64 d40 last | 27.3 ± 1.9 | 35.0 ± 1.2 | 44.2 ± 1.5 | 17.7 ± 1.2 | 21.7 ± 0.1 | 25.2 ± 0.7 |

| width | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| w64 d6 N10 all | 40.1 ± 1.6 | 53.0 ± 1.0 | 64.7 ± 0.2 | 18.0 ± 1.5 | 25.9 ± 0.3 | 32.7 ± 0.4 |
| w16 d6 N10 all | 40.3 ± 0.8 | 50.6 ± 1.0 | 62.4 ± 0.3 | 18.4 ± 2.0 | 24.6 ± 0.2 | 31.2 ± 0.6 |
| w16 d6 N40 all | 43.1 ± 1.4 | 54.5 ± 1.1 | 65.4 ± 0.4 | 18.7 ± 2.4 | 26.4 ± 0.4 | 32.9 ± 0.5 |
| w64 d6 N40 all | 43.2 ± 1.4 | 55.9 ± 0.4 | 68.2 ± 0.6 | 18.7 ± 2.3 | 27.2 ± 0.5 | 34.6 ± 0.5 |
| w64 d20 N10 all | 36.4 ± 1.5 | 49.5 ± 0.9 | 62.3 ± 0.4 | 20.5 ± 1.4 | 24.8 ± 0.4 | 32.5 ± 0.6 |
| w16 d20 N10 all | 34.0 ± 4.2 | 48.1 ± 0.1 | 59.6 ± 0.5 | 20.1 ± 1.1 | 24.3 ± 0.6 | 31.2 ± 0.8 |
| w16 d20 N40 all | 38.4 ± 2.8 | 52.1 ± 0.7 | 64.3 ± 0.2 | 19.8 ± 0.9 | 26.0 ± 0.2 | 32.2 ± 0.1 |
| w64 d20 N40 all | 35.6 ± 0.8 | 50.9 ± 1.3 | 65.4 ± 0.2 | 19.5 ± 2.7 | 26.1 ± 0.3 | 33.9 ± 0.4 |

| reference | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| pixels | 22.8 ± 1.1 | 28.3 ± 1.2 | 33.5 ± 0.4 | 16.9 ± 0.8 | 18.9 ± 0.3 | 22.4 ± 0.5 |
| dinov2_s | 98.1 ± 0.1 | 98.4 ± 0.1 | 99.0 ± 0.2 | 91.0 ± 0.1 | 92.0 ± 0.3 | 93.9 ± 0.3 |
| dinov2_b | 99.2 ± 0.1 | 99.3 ± 0.1 | 99.4 ± 0.1 | 93.0 ± 0.4 | 94.3 ± 0.4 | 95.2 ± 0.1 |
| dinov2_l | 99.5 ± 0.1 | 99.4 ± 0.0 | 99.7 ± 0.0 | 95.5 ± 0.2 | 95.9 ± 0.3 | 96.2 ± 0.2 |
| rad_dino | 47.9 ± 1.5 | 60.5 ± 1.0 | 70.1 ± 0.3 | 24.1 ± 0.8 | 29.5 ± 0.9 | 41.4 ± 0.7 |
| resnet50 | 97.2 ± 0.9 | 99.0 ± 0.0 | 99.2 ± 0.2 | 93.9 ± 0.3 | 95.4 ± 0.1 | 96.2 ± 0.1 |

### Readout: select64


| nets (w64, depth 6, all blocks) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| N1 | 30.5 ± 1.1 | 34.9 ± 2.3 | 45.6 ± 3.1 | 16.9 ± 0.5 | 20.7 ± 0.8 | 23.4 ± 0.3 |
| N2 | 28.6 ± 2.6 | 37.2 ± 0.8 | 46.0 ± 2.6 | 16.1 ± 0.9 | 20.8 ± 0.6 | 24.8 ± 0.5 |
| N5 | 31.2 ± 1.1 | 36.1 ± 0.2 | 46.1 ± 1.6 | 17.4 ± 0.5 | 20.8 ± 0.7 | 24.8 ± 0.7 |
| N10 | 30.4 ± 1.1 | 38.0 ± 0.5 | 46.7 ± 1.6 | 15.7 ± 1.5 | 21.5 ± 0.7 | 25.4 ± 0.7 |
| N20 | 29.3 ± 2.2 | 37.9 ± 1.3 | 48.2 ± 1.3 | 17.0 ± 0.3 | 20.7 ± 0.5 | 24.8 ± 0.8 |
| N40 | 27.8 ± 2.7 | 37.9 ± 0.7 | 48.5 ± 1.8 | 17.3 ± 0.5 | 21.2 ± 1.1 | 25.0 ± 0.2 |

| nets (w64, depth 20, all blocks) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| N1 | 28.2 ± 2.0 | 35.0 ± 1.1 | 43.6 ± 2.1 | 15.7 ± 0.1 | 20.6 ± 0.9 | 23.8 ± 0.3 |
| N2 | 28.6 ± 1.8 | 34.8 ± 2.5 | 45.3 ± 2.4 | 16.4 ± 0.8 | 21.1 ± 0.5 | 24.9 ± 0.8 |
| N5 | 29.0 ± 0.9 | 35.7 ± 0.3 | 44.6 ± 1.8 | 16.6 ± 0.9 | 20.4 ± 0.8 | 25.8 ± 0.4 |
| N10 | 28.6 ± 2.0 | 36.8 ± 0.5 | 45.9 ± 1.4 | 17.3 ± 1.6 | 21.1 ± 0.8 | 25.8 ± 0.2 |
| N20 | 28.1 ± 2.8 | 37.4 ± 1.6 | 47.8 ± 1.6 | 16.9 ± 1.2 | 22.3 ± 0.1 | 26.2 ± 0.7 |
| N40 | 30.5 ± 1.9 | 37.1 ± 0.9 | 47.9 ± 0.9 | 17.5 ± 1.0 | 21.7 ± 0.8 | 25.0 ± 0.6 |

| depth (w64, N10, all blocks) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| d1 | 30.8 ± 2.0 | 39.0 ± 0.8 | 47.3 ± 0.6 | 15.4 ± 1.0 | 20.4 ± 0.4 | 24.3 ± 0.8 |
| d2 | 31.0 ± 1.9 | 41.8 ± 0.5 | 50.7 ± 1.2 | 16.6 ± 0.3 | 21.5 ± 0.4 | 25.4 ± 0.5 |
| d3 | 31.9 ± 1.5 | 38.1 ± 2.1 | 48.5 ± 0.4 | 16.3 ± 0.9 | 21.3 ± 0.1 | 25.3 ± 0.3 |
| d4 | 32.0 ± 1.7 | 36.5 ± 0.8 | 47.7 ± 1.1 | 16.4 ± 1.0 | 21.0 ± 0.8 | 25.8 ± 0.6 |
| d6 | 30.4 ± 1.1 | 38.0 ± 0.5 | 46.7 ± 1.6 | 15.7 ± 1.5 | 21.5 ± 0.7 | 25.4 ± 0.7 |
| d10 | 30.2 ± 2.5 | 36.8 ± 0.8 | 45.8 ± 1.2 | 16.5 ± 1.2 | 21.1 ± 0.8 | 25.4 ± 0.3 |
| d20 | 28.6 ± 2.0 | 36.8 ± 0.5 | 45.9 ± 1.4 | 17.3 ± 1.6 | 21.1 ± 0.8 | 25.8 ± 0.2 |
| d40 | 29.4 ± 0.9 | 36.8 ± 0.5 | 45.9 ± 1.4 | 16.2 ± 1.7 | 21.1 ± 0.9 | 25.6 ± 0.6 |

| depth (w64, N10, last blocks) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| d1 | 30.8 ± 2.0 | 39.0 ± 0.8 | 47.3 ± 0.6 | 15.4 ± 1.0 | 20.4 ± 0.4 | 24.3 ± 0.8 |
| d2 | 29.4 ± 1.9 | 39.1 ± 0.2 | 48.9 ± 1.1 | 16.5 ± 0.4 | 20.1 ± 0.7 | 25.2 ± 0.3 |
| d3 | 30.4 ± 1.0 | 37.6 ± 0.3 | 46.5 ± 1.7 | 16.4 ± 1.0 | 20.5 ± 0.3 | 25.6 ± 0.4 |
| d4 | 30.3 ± 1.5 | 36.3 ± 1.2 | 44.2 ± 0.3 | 17.0 ± 0.6 | 20.4 ± 0.4 | 23.5 ± 0.8 |
| d6 | 26.9 ± 0.4 | 30.7 ± 1.8 | 40.5 ± 1.9 | 16.3 ± 0.4 | 20.9 ± 1.2 | 23.4 ± 0.6 |
| d10 | 24.4 ± 1.0 | 30.6 ± 0.7 | 38.7 ± 1.6 | 15.4 ± 0.5 | 19.0 ± 0.3 | 23.1 ± 0.2 |
| d20 | 20.4 ± 0.6 | 24.1 ± 1.5 | 31.2 ± 0.9 | 15.1 ± 1.0 | 16.4 ± 0.5 | 19.4 ± 0.7 |
| d40 | 17.8 ± 0.7 | 20.9 ± 1.3 | 23.6 ± 2.6 | 13.9 ± 1.0 | 15.8 ± 1.0 | 18.2 ± 0.2 |

| depth, residual (w64, N10) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| w64 d6 all | 30.4 ± 1.1 | 38.0 ± 0.5 | 46.7 ± 1.6 | 15.7 ± 1.5 | 21.5 ± 0.7 | 25.4 ± 0.7 |
| w64 d10 all | 30.2 ± 2.5 | 36.8 ± 0.8 | 45.8 ± 1.2 | 16.5 ± 1.2 | 21.1 ± 0.8 | 25.4 ± 0.3 |
| res w64 d10 all | 28.7 ± 0.8 | 36.9 ± 0.5 | 46.0 ± 0.7 | 16.0 ± 1.1 | 21.1 ± 0.3 | 25.7 ± 0.8 |
| w64 d20 all | 28.6 ± 2.0 | 36.8 ± 0.5 | 45.9 ± 1.4 | 17.3 ± 1.6 | 21.1 ± 0.8 | 25.8 ± 0.2 |
| res w64 d20 all | 29.0 ± 2.0 | 37.7 ± 1.1 | 45.1 ± 1.0 | 17.0 ± 0.7 | 22.1 ± 1.0 | 25.3 ± 1.2 |
| w64 d40 all | 29.4 ± 0.9 | 36.8 ± 0.5 | 45.9 ± 1.4 | 16.2 ± 1.7 | 21.1 ± 0.9 | 25.6 ± 0.6 |
| res w64 d40 all | 29.6 ± 1.0 | 35.9 ± 0.4 | 44.3 ± 1.1 | 17.4 ± 0.2 | 22.3 ± 0.3 | 25.3 ± 0.6 |
| w64 d6 last | 26.9 ± 0.4 | 30.7 ± 1.8 | 40.5 ± 1.9 | 16.3 ± 0.4 | 20.9 ± 1.2 | 23.4 ± 0.6 |
| w64 d10 last | 24.4 ± 1.0 | 30.6 ± 0.7 | 38.7 ± 1.6 | 15.4 ± 0.5 | 19.0 ± 0.3 | 23.1 ± 0.2 |
| res w64 d10 last | 26.0 ± 0.8 | 33.7 ± 1.2 | 41.6 ± 1.4 | 16.5 ± 0.3 | 21.1 ± 0.2 | 23.2 ± 0.1 |
| w64 d20 last | 20.4 ± 0.6 | 24.1 ± 1.5 | 31.2 ± 0.9 | 15.1 ± 1.0 | 16.4 ± 0.5 | 19.4 ± 0.7 |
| res w64 d20 last | 26.4 ± 0.6 | 34.3 ± 0.2 | 41.2 ± 0.7 | 15.6 ± 0.4 | 21.3 ± 0.6 | 23.4 ± 0.4 |
| w64 d40 last | 17.8 ± 0.7 | 20.9 ± 1.3 | 23.6 ± 2.6 | 13.9 ± 1.0 | 15.8 ± 1.0 | 18.2 ± 0.2 |
| res w64 d40 last | 24.3 ± 1.1 | 31.9 ± 1.3 | 39.2 ± 1.3 | 16.3 ± 0.9 | 21.0 ± 0.7 | 23.1 ± 0.2 |

| width | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| w64 d6 N10 all | 30.4 ± 1.1 | 38.0 ± 0.5 | 46.7 ± 1.6 | 15.7 ± 1.5 | 21.5 ± 0.7 | 25.4 ± 0.7 |
| w16 d6 N10 all | 30.6 ± 1.5 | 38.1 ± 1.8 | 46.6 ± 0.7 | 16.2 ± 1.1 | 21.8 ± 0.1 | 24.7 ± 0.8 |
| w16 d6 N40 all | 30.0 ± 2.3 | 38.2 ± 0.5 | 47.0 ± 0.5 | 16.8 ± 0.8 | 21.5 ± 0.7 | 24.3 ± 0.8 |
| w64 d6 N40 all | 27.8 ± 2.7 | 37.9 ± 0.7 | 48.5 ± 1.8 | 17.3 ± 0.5 | 21.2 ± 1.1 | 25.0 ± 0.2 |
| w64 d20 N10 all | 28.6 ± 2.0 | 36.8 ± 0.5 | 45.9 ± 1.4 | 17.3 ± 1.6 | 21.1 ± 0.8 | 25.8 ± 0.2 |
| w16 d20 N10 all | 28.3 ± 3.5 | 36.9 ± 0.7 | 45.5 ± 0.4 | 15.8 ± 0.5 | 21.0 ± 0.6 | 24.0 ± 1.0 |
| w16 d20 N40 all | 29.5 ± 1.4 | 38.2 ± 0.7 | 47.3 ± 1.1 | 16.4 ± 0.1 | 22.0 ± 0.4 | 24.4 ± 1.6 |
| w64 d20 N40 all | 30.5 ± 1.9 | 37.1 ± 0.9 | 47.9 ± 0.9 | 17.5 ± 1.0 | 21.7 ± 0.8 | 25.0 ± 0.6 |

| reference | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| pixels | 22.0 ± 1.7 | 22.8 ± 0.9 | 26.0 ± 0.9 | 16.5 ± 0.4 | 18.8 ± 1.0 | 21.3 ± 1.0 |
| dinov2_s | 95.9 ± 0.5 | 95.8 ± 0.1 | 96.9 ± 0.2 | 87.5 ± 1.0 | 90.0 ± 0.3 | 91.1 ± 0.4 |
| dinov2_b | 97.4 ± 0.5 | 98.0 ± 0.1 | 98.1 ± 0.1 | 92.4 ± 0.4 | 92.9 ± 0.3 | 93.8 ± 0.2 |
| dinov2_l | 97.3 ± 0.6 | 98.4 ± 0.2 | 98.9 ± 0.0 | 93.1 ± 0.9 | 94.2 ± 0.3 | 94.9 ± 0.1 |
| rad_dino | 44.0 ± 1.7 | 50.3 ± 1.0 | 56.3 ± 0.8 | 22.5 ± 0.9 | 25.1 ± 0.8 | 31.9 ± 0.4 |
| resnet50 | 96.2 ± 0.7 | 98.3 ± 0.3 | 98.9 ± 0.1 | 93.7 ± 0.4 | 94.5 ± 0.7 | 94.9 ± 0.3 |

### Readout: select256


| nets (w64, depth 6, all blocks) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| N1 | 33.5 ± 2.2 | 42.8 ± 1.2 | 53.2 ± 0.8 | 17.3 ± 0.7 | 22.1 ± 0.3 | 26.2 ± 0.1 |
| N2 | 34.8 ± 2.0 | 43.6 ± 2.3 | 53.6 ± 2.0 | 18.3 ± 0.9 | 23.0 ± 0.7 | 25.6 ± 0.7 |
| N5 | 34.2 ± 3.0 | 44.0 ± 1.7 | 53.8 ± 2.0 | 18.3 ± 0.8 | 22.8 ± 1.0 | 27.4 ± 0.4 |
| N10 | 35.1 ± 1.5 | 41.0 ± 3.1 | 53.7 ± 1.1 | 18.9 ± 1.3 | 23.3 ± 0.1 | 28.0 ± 0.6 |
| N20 | 33.8 ± 0.7 | 40.2 ± 1.9 | 53.6 ± 1.3 | 18.0 ± 0.1 | 23.0 ± 0.3 | 27.4 ± 1.4 |
| N40 | 35.1 ± 0.8 | 42.5 ± 0.8 | 54.4 ± 1.4 | 17.9 ± 1.3 | 23.1 ± 1.2 | 29.2 ± 0.2 |

| nets (w64, depth 20, all blocks) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| N1 | 30.3 ± 2.9 | 40.9 ± 2.4 | 51.3 ± 2.3 | 16.7 ± 0.8 | 21.5 ± 0.6 | 25.2 ± 0.7 |
| N2 | 31.7 ± 3.8 | 41.1 ± 2.8 | 53.2 ± 2.5 | 17.8 ± 0.3 | 21.6 ± 1.7 | 26.7 ± 0.5 |
| N5 | 32.4 ± 1.2 | 42.7 ± 2.2 | 53.1 ± 2.2 | 17.8 ± 1.4 | 23.5 ± 0.6 | 28.8 ± 0.6 |
| N10 | 33.1 ± 1.9 | 39.9 ± 0.5 | 53.2 ± 1.6 | 17.9 ± 0.7 | 24.0 ± 0.4 | 28.8 ± 0.5 |
| N20 | 34.3 ± 0.5 | 41.1 ± 1.2 | 53.7 ± 1.0 | 17.7 ± 1.2 | 23.6 ± 0.6 | 29.4 ± 0.5 |
| N40 | 33.8 ± 0.3 | 42.6 ± 1.1 | 53.9 ± 1.0 | 18.6 ± 0.6 | 22.6 ± 1.0 | 29.9 ± 0.3 |

| depth (w64, N10, all blocks) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| d1 | 32.3 ± 2.1 | 41.9 ± 0.8 | 50.5 ± 1.1 | 16.1 ± 1.1 | 21.2 ± 1.4 | 25.2 ± 0.2 |
| d2 | 34.4 ± 1.4 | 45.2 ± 1.2 | 55.3 ± 0.2 | 17.1 ± 0.2 | 22.5 ± 0.8 | 28.1 ± 0.7 |
| d3 | 35.8 ± 1.6 | 45.6 ± 1.9 | 54.7 ± 0.7 | 17.7 ± 0.3 | 23.6 ± 0.1 | 27.8 ± 0.4 |
| d4 | 35.7 ± 0.1 | 45.0 ± 1.2 | 54.3 ± 1.6 | 18.8 ± 0.6 | 23.3 ± 0.9 | 27.8 ± 0.1 |
| d6 | 35.1 ± 1.5 | 41.0 ± 3.1 | 53.7 ± 1.1 | 18.9 ± 1.3 | 23.3 ± 0.1 | 28.0 ± 0.6 |
| d10 | 34.9 ± 1.1 | 41.1 ± 1.5 | 53.2 ± 1.2 | 18.3 ± 0.7 | 23.5 ± 0.2 | 28.6 ± 0.8 |
| d20 | 33.1 ± 1.9 | 39.9 ± 0.5 | 53.2 ± 1.6 | 17.9 ± 0.7 | 24.0 ± 0.4 | 28.8 ± 0.5 |
| d40 | 32.6 ± 2.1 | 40.1 ± 0.6 | 53.1 ± 1.6 | 16.9 ± 0.4 | 23.0 ± 0.6 | 28.1 ± 0.7 |

| depth (w64, N10, last blocks) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| d1 | 32.3 ± 2.1 | 41.9 ± 0.8 | 50.5 ± 1.1 | 16.1 ± 1.1 | 21.2 ± 1.4 | 25.2 ± 0.2 |
| d2 | 34.0 ± 3.0 | 44.6 ± 0.8 | 55.0 ± 0.6 | 17.6 ± 0.6 | 22.6 ± 0.4 | 27.2 ± 0.3 |
| d3 | 33.8 ± 2.0 | 43.9 ± 0.9 | 53.3 ± 0.8 | 16.0 ± 0.3 | 22.2 ± 0.7 | 26.7 ± 0.8 |
| d4 | 33.8 ± 2.4 | 41.2 ± 2.4 | 50.0 ± 1.0 | 17.2 ± 0.8 | 22.2 ± 0.6 | 25.1 ± 0.5 |
| d6 | 31.1 ± 1.4 | 37.1 ± 0.7 | 45.2 ± 1.2 | 16.2 ± 0.3 | 21.4 ± 0.4 | 24.5 ± 1.7 |
| d10 | 28.3 ± 0.6 | 32.9 ± 2.2 | 42.6 ± 0.8 | 16.0 ± 1.2 | 20.7 ± 0.5 | 23.5 ± 0.7 |
| d20 | 23.8 ± 0.3 | 29.1 ± 0.8 | 35.5 ± 1.0 | 15.8 ± 1.3 | 18.5 ± 0.6 | 21.8 ± 0.3 |
| d40 | 21.0 ± 1.4 | 24.0 ± 1.3 | 28.0 ± 1.7 | 15.3 ± 1.0 | 18.2 ± 0.5 | 20.2 ± 0.6 |

| depth, residual (w64, N10) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| w64 d6 all | 35.1 ± 1.5 | 41.0 ± 3.1 | 53.7 ± 1.1 | 18.9 ± 1.3 | 23.3 ± 0.1 | 28.0 ± 0.6 |
| w64 d10 all | 34.9 ± 1.1 | 41.1 ± 1.5 | 53.2 ± 1.2 | 18.3 ± 0.7 | 23.5 ± 0.2 | 28.6 ± 0.8 |
| res w64 d10 all | 34.4 ± 1.7 | 39.5 ± 2.9 | 52.7 ± 1.9 | 18.6 ± 0.7 | 23.3 ± 0.6 | 28.1 ± 0.2 |
| w64 d20 all | 33.1 ± 1.9 | 39.9 ± 0.5 | 53.2 ± 1.6 | 17.9 ± 0.7 | 24.0 ± 0.4 | 28.8 ± 0.5 |
| res w64 d20 all | 33.0 ± 1.8 | 40.2 ± 0.5 | 52.2 ± 1.0 | 17.7 ± 0.2 | 23.1 ± 0.6 | 28.6 ± 0.6 |
| w64 d40 all | 32.6 ± 2.1 | 40.1 ± 0.6 | 53.1 ± 1.6 | 16.9 ± 0.4 | 23.0 ± 0.6 | 28.1 ± 0.7 |
| res w64 d40 all | 32.1 ± 2.1 | 40.5 ± 0.9 | 51.5 ± 1.3 | 17.8 ± 0.3 | 23.6 ± 0.8 | 28.2 ± 0.1 |
| w64 d6 last | 31.1 ± 1.4 | 37.1 ± 0.7 | 45.2 ± 1.2 | 16.2 ± 0.3 | 21.4 ± 0.4 | 24.5 ± 1.7 |
| w64 d10 last | 28.3 ± 0.6 | 32.9 ± 2.2 | 42.6 ± 0.8 | 16.0 ± 1.2 | 20.7 ± 0.5 | 23.5 ± 0.7 |
| res w64 d10 last | 29.5 ± 0.7 | 36.0 ± 0.6 | 44.3 ± 1.4 | 16.9 ± 0.7 | 21.0 ± 0.5 | 24.4 ± 0.1 |
| w64 d20 last | 23.8 ± 0.3 | 29.1 ± 0.8 | 35.5 ± 1.0 | 15.8 ± 1.3 | 18.5 ± 0.6 | 21.8 ± 0.3 |
| res w64 d20 last | 29.6 ± 0.8 | 34.9 ± 0.8 | 44.1 ± 1.2 | 16.9 ± 0.9 | 21.5 ± 0.6 | 24.6 ± 0.4 |
| w64 d40 last | 21.0 ± 1.4 | 24.0 ± 1.3 | 28.0 ± 1.7 | 15.3 ± 1.0 | 18.2 ± 0.5 | 20.2 ± 0.6 |
| res w64 d40 last | 27.1 ± 1.7 | 33.9 ± 1.0 | 41.9 ± 0.9 | 17.4 ± 0.7 | 21.8 ± 0.5 | 24.2 ± 1.1 |

| width | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| w64 d6 N10 all | 35.1 ± 1.5 | 41.0 ± 3.1 | 53.7 ± 1.1 | 18.9 ± 1.3 | 23.3 ± 0.1 | 28.0 ± 0.6 |
| w16 d6 N10 all | 34.4 ± 1.4 | 43.7 ± 2.6 | 55.8 ± 0.5 | 18.1 ± 0.3 | 23.0 ± 0.7 | 26.7 ± 0.7 |
| w16 d6 N40 all | 34.0 ± 0.7 | 43.8 ± 2.0 | 54.8 ± 0.9 | 17.6 ± 0.4 | 22.6 ± 0.9 | 27.8 ± 1.0 |
| w64 d6 N40 all | 35.1 ± 0.8 | 42.5 ± 0.8 | 54.4 ± 1.4 | 17.9 ± 1.3 | 23.1 ± 1.2 | 29.2 ± 0.2 |
| w64 d20 N10 all | 33.1 ± 1.9 | 39.9 ± 0.5 | 53.2 ± 1.6 | 17.9 ± 0.7 | 24.0 ± 0.4 | 28.8 ± 0.5 |
| w16 d20 N10 all | 34.3 ± 1.0 | 42.9 ± 3.3 | 53.4 ± 2.0 | 17.4 ± 0.2 | 22.1 ± 0.7 | 27.2 ± 0.7 |
| w16 d20 N40 all | 32.8 ± 1.1 | 42.7 ± 0.6 | 53.5 ± 1.0 | 17.6 ± 1.2 | 23.4 ± 0.2 | 28.1 ± 0.3 |
| w64 d20 N40 all | 33.8 ± 0.3 | 42.6 ± 1.1 | 53.9 ± 1.0 | 18.6 ± 0.6 | 22.6 ± 1.0 | 29.9 ± 0.3 |

| reference | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| pixels | 23.7 ± 1.1 | 26.4 ± 1.6 | 29.2 ± 0.6 | 17.1 ± 1.1 | 19.0 ± 0.7 | 23.0 ± 0.4 |
| dinov2_s | 98.0 ± 0.0 | 98.3 ± 0.2 | 98.6 ± 0.1 | 91.3 ± 0.3 | 91.9 ± 0.5 | 93.4 ± 0.3 |
| dinov2_b | 98.7 ± 0.4 | 99.0 ± 0.0 | 99.2 ± 0.0 | 93.6 ± 0.1 | 94.2 ± 0.2 | 94.9 ± 0.1 |
| dinov2_l | 99.0 ± 0.1 | 99.3 ± 0.0 | 99.3 ± 0.1 | 95.2 ± 0.4 | 95.3 ± 0.2 | 95.9 ± 0.1 |
| rad_dino | 44.5 ± 1.8 | 56.7 ± 1.2 | 65.8 ± 0.4 | 23.9 ± 1.0 | 27.8 ± 1.0 | 37.1 ± 0.5 |
| resnet50 | 98.4 ± 0.1 | 98.7 ± 0.1 | 99.4 ± 0.1 | 95.2 ± 0.4 | 95.7 ± 0.2 | 96.2 ± 0.0 |

### Readout: mlp


| nets (w64, depth 6, all blocks) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| N1 | 34.1 ± 1.3 | 44.8 ± 1.0 | 57.2 ± 0.6 | 18.2 ± 1.1 | 22.7 ± 0.4 | 28.4 ± 0.8 |
| N2 | 34.9 ± 1.4 | 46.5 ± 0.6 | 60.9 ± 0.7 | 18.8 ± 1.0 | 24.0 ± 0.1 | 30.7 ± 0.3 |
| N5 | 34.6 ± 3.2 | 48.2 ± 0.5 | 63.6 ± 0.8 | 18.3 ± 1.1 | 24.2 ± 0.3 | 32.7 ± 0.1 |
| N10 | 33.6 ± 2.5 | 48.4 ± 0.6 | 65.1 ± 0.7 | 17.5 ± 0.6 | 23.9 ± 0.3 | 33.6 ± 0.2 |
| N20 | 33.4 ± 1.4 | 46.0 ± 2.3 | 65.5 ± 0.6 | 17.7 ± 0.9 | 23.9 ± 0.8 | 34.1 ± 0.3 |
| N40 | 33.2 ± 2.2 | 45.7 ± 1.1 | 65.7 ± 0.0 | 18.3 ± 0.6 | 23.6 ± 0.6 | 34.4 ± 0.4 |

| nets (w64, depth 20, all blocks) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| N1 | 31.3 ± 1.5 | 41.1 ± 0.9 | 53.4 ± 1.4 | 17.4 ± 0.3 | 21.9 ± 0.6 | 27.5 ± 0.5 |
| N2 | 30.8 ± 2.2 | 42.3 ± 1.8 | 56.9 ± 1.6 | 18.1 ± 1.3 | 22.8 ± 0.0 | 29.9 ± 0.9 |
| N5 | 29.8 ± 2.0 | 43.2 ± 1.0 | 58.8 ± 1.1 | 18.1 ± 0.8 | 22.9 ± 0.4 | 31.3 ± 0.7 |
| N10 | 30.1 ± 2.9 | 42.0 ± 0.3 | 59.5 ± 0.8 | 17.8 ± 0.1 | 22.8 ± 0.6 | 31.3 ± 0.4 |
| N20 | 30.5 ± 1.3 | 42.5 ± 1.0 | 59.5 ± 0.2 | 18.3 ± 0.9 | 22.7 ± 0.5 | 31.8 ± 0.8 |
| N40 | 30.3 ± 0.6 | 40.9 ± 1.9 | 58.8 ± 0.6 | 17.7 ± 0.6 | 22.6 ± 0.4 | 32.0 ± 0.6 |

| depth (w64, N10, all blocks) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| d1 | 32.6 ± 1.8 | 42.9 ± 0.8 | 55.9 ± 0.3 | 16.7 ± 0.9 | 21.4 ± 0.6 | 27.7 ± 0.5 |
| d2 | 32.9 ± 1.8 | 46.8 ± 0.8 | 60.8 ± 0.3 | 17.9 ± 0.8 | 23.2 ± 0.4 | 30.7 ± 0.3 |
| d3 | 33.2 ± 3.1 | 48.0 ± 0.8 | 63.6 ± 0.2 | 17.9 ± 0.6 | 24.2 ± 0.9 | 32.2 ± 0.4 |
| d4 | 34.5 ± 3.3 | 49.0 ± 0.5 | 64.9 ± 0.7 | 16.8 ± 0.1 | 24.2 ± 0.5 | 32.7 ± 0.2 |
| d6 | 33.6 ± 2.5 | 48.4 ± 0.6 | 65.1 ± 0.7 | 17.5 ± 0.6 | 23.9 ± 0.3 | 33.6 ± 0.2 |
| d10 | 31.2 ± 1.6 | 44.9 ± 1.2 | 63.6 ± 0.2 | 17.7 ± 0.2 | 24.2 ± 0.6 | 32.8 ± 0.3 |
| d20 | 30.1 ± 2.9 | 42.0 ± 0.3 | 59.5 ± 0.8 | 17.8 ± 0.1 | 22.8 ± 0.6 | 31.3 ± 0.4 |
| d40 | 27.0 ± 2.3 | 37.2 ± 0.7 | 55.1 ± 1.2 | 17.4 ± 1.3 | 21.9 ± 1.7 | 30.1 ± 0.3 |

| depth (w64, N10, last blocks) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| d1 | 32.6 ± 1.8 | 42.9 ± 0.8 | 55.9 ± 0.3 | 16.7 ± 0.9 | 21.4 ± 0.6 | 27.7 ± 0.5 |
| d2 | 35.1 ± 1.8 | 46.6 ± 0.9 | 59.7 ± 0.9 | 17.4 ± 0.9 | 23.6 ± 0.3 | 30.4 ± 0.2 |
| d3 | 35.1 ± 1.2 | 46.3 ± 0.4 | 60.1 ± 1.0 | 17.8 ± 0.7 | 23.1 ± 0.2 | 30.0 ± 0.7 |
| d4 | 34.5 ± 0.9 | 45.5 ± 0.5 | 58.1 ± 1.0 | 18.4 ± 0.5 | 23.6 ± 0.5 | 29.7 ± 0.6 |
| d6 | 31.7 ± 0.4 | 40.7 ± 0.3 | 50.5 ± 0.8 | 17.2 ± 0.4 | 22.2 ± 0.6 | 26.9 ± 0.9 |
| d10 | 28.4 ± 1.3 | 37.0 ± 0.7 | 45.5 ± 0.6 | 17.4 ± 1.1 | 21.1 ± 0.2 | 24.6 ± 1.0 |
| d20 | 25.0 ± 1.3 | 30.9 ± 0.4 | 37.8 ± 0.7 | 16.9 ± 1.6 | 19.5 ± 0.7 | 22.0 ± 0.6 |
| d40 | 21.7 ± 0.5 | 25.7 ± 1.7 | 30.2 ± 1.9 | 15.3 ± 1.4 | 18.1 ± 1.0 | 19.4 ± 0.7 |

| depth, residual (w64, N10) | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| w64 d6 all | 33.6 ± 2.5 | 48.4 ± 0.6 | 65.1 ± 0.7 | 17.5 ± 0.6 | 23.9 ± 0.3 | 33.6 ± 0.2 |
| w64 d10 all | 31.2 ± 1.6 | 44.9 ± 1.2 | 63.6 ± 0.2 | 17.7 ± 0.2 | 24.2 ± 0.6 | 32.8 ± 0.3 |
| res w64 d10 all | 32.4 ± 1.1 | 44.6 ± 1.0 | 63.5 ± 0.6 | 16.9 ± 1.3 | 23.4 ± 0.8 | 32.8 ± 0.1 |
| w64 d20 all | 30.1 ± 2.9 | 42.0 ± 0.3 | 59.5 ± 0.8 | 17.8 ± 0.1 | 22.8 ± 0.6 | 31.3 ± 0.4 |
| res w64 d20 all | 30.9 ± 1.6 | 41.5 ± 0.5 | 61.6 ± 0.7 | 17.3 ± 1.0 | 22.6 ± 0.6 | 32.0 ± 0.7 |
| w64 d40 all | 27.0 ± 2.3 | 37.2 ± 0.7 | 55.1 ± 1.2 | 17.4 ± 1.3 | 21.9 ± 1.7 | 30.1 ± 0.3 |
| res w64 d40 all | 28.9 ± 1.5 | 41.3 ± 0.3 | 58.4 ± 1.5 | 17.1 ± 0.9 | 21.8 ± 1.2 | 31.1 ± 0.6 |
| w64 d6 last | 31.7 ± 0.4 | 40.7 ± 0.3 | 50.5 ± 0.8 | 17.2 ± 0.4 | 22.2 ± 0.6 | 26.9 ± 0.9 |
| w64 d10 last | 28.4 ± 1.3 | 37.0 ± 0.7 | 45.5 ± 0.6 | 17.4 ± 1.1 | 21.1 ± 0.2 | 24.6 ± 1.0 |
| res w64 d10 last | 30.6 ± 1.0 | 39.8 ± 0.4 | 49.6 ± 1.3 | 17.8 ± 0.8 | 22.1 ± 0.6 | 26.4 ± 0.2 |
| w64 d20 last | 25.0 ± 1.3 | 30.9 ± 0.4 | 37.8 ± 0.7 | 16.9 ± 1.6 | 19.5 ± 0.7 | 22.0 ± 0.6 |
| res w64 d20 last | 29.7 ± 0.7 | 38.3 ± 0.4 | 48.1 ± 0.8 | 17.6 ± 0.9 | 21.7 ± 0.5 | 25.2 ± 0.9 |
| w64 d40 last | 21.7 ± 0.5 | 25.7 ± 1.7 | 30.2 ± 1.9 | 15.3 ± 1.4 | 18.1 ± 1.0 | 19.4 ± 0.7 |
| res w64 d40 last | 28.0 ± 0.6 | 36.5 ± 0.1 | 45.6 ± 1.5 | 17.3 ± 1.0 | 21.5 ± 0.5 | 24.8 ± 0.5 |

| width | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| w64 d6 N10 all | 33.6 ± 2.5 | 48.4 ± 0.6 | 65.1 ± 0.7 | 17.5 ± 0.6 | 23.9 ± 0.3 | 33.6 ± 0.2 |
| w16 d6 N10 all | 37.2 ± 1.9 | 49.8 ± 0.9 | 62.7 ± 0.4 | 18.7 ± 0.8 | 24.4 ± 0.5 | 31.3 ± 0.7 |
| w16 d6 N40 all | 35.6 ± 2.6 | 50.1 ± 0.4 | 66.3 ± 0.3 | 18.5 ± 0.7 | 24.8 ± 0.4 | 33.8 ± 0.4 |
| w64 d6 N40 all | 33.2 ± 2.2 | 45.7 ± 1.1 | 65.7 ± 0.0 | 18.3 ± 0.6 | 23.6 ± 0.6 | 34.4 ± 0.4 |
| w64 d20 N10 all | 30.1 ± 2.9 | 42.0 ± 0.3 | 59.5 ± 0.8 | 17.8 ± 0.1 | 22.8 ± 0.6 | 31.3 ± 0.4 |
| w16 d20 N10 all | 33.5 ± 2.8 | 46.3 ± 0.6 | 60.0 ± 0.3 | 17.8 ± 0.5 | 24.0 ± 0.3 | 31.1 ± 0.8 |
| w16 d20 N40 all | 31.7 ± 2.7 | 44.7 ± 1.6 | 61.9 ± 0.9 | 17.9 ± 0.2 | 23.0 ± 0.8 | 32.2 ± 0.3 |
| w64 d20 N40 all | 30.3 ± 0.6 | 40.9 ± 1.9 | 58.8 ± 0.6 | 17.7 ± 0.6 | 22.6 ± 0.4 | 32.0 ± 0.6 |

| reference | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|
| pixels | 25.2 ± 1.2 | 30.3 ± 1.2 | 36.9 ± 0.4 | 16.9 ± 1.4 | 19.0 ± 0.6 | 21.9 ± 0.5 |
| dinov2_s | 97.9 ± 0.1 | 98.3 ± 0.1 | 99.1 ± 0.2 | 90.5 ± 0.4 | 91.8 ± 0.2 | 94.1 ± 0.3 |
| dinov2_b | 99.2 ± 0.1 | 99.3 ± 0.1 | 99.5 ± 0.0 | 93.3 ± 0.1 | 94.3 ± 0.3 | 95.5 ± 0.2 |
| dinov2_l | 99.3 ± 0.0 | 99.3 ± 0.1 | 99.7 ± 0.0 | 95.1 ± 0.4 | 95.5 ± 0.1 | 96.1 ± 0.1 |
| rad_dino | 47.1 ± 1.2 | 58.5 ± 0.6 | 69.2 ± 0.2 | 23.8 ± 0.9 | 29.7 ± 0.5 | 40.5 ± 0.8 |
| resnet50 | 96.5 ± 0.5 | 98.2 ± 0.1 | 99.2 ± 0.1 | 92.6 ± 0.2 | 94.5 ± 0.3 | 95.7 ± 0.0 |

### All readouts side by side (selected rows)


| row | readout | Imagenette 10 | Imagenette 25 | Imagenette 100 | Imagewoof 10 | Imagewoof 25 | Imagewoof 100 |
|---|---|---|---|---|---|---|---|
| w64 d6 N10 all | linear | 40.1 ± 1.6 | 53.0 ± 1.0 | 64.7 ± 0.2 | 18.0 ± 1.5 | 25.9 ± 0.3 | 32.7 ± 0.4 |
| w64 d6 N10 all | select64 | 30.4 ± 1.1 | 38.0 ± 0.5 | 46.7 ± 1.6 | 15.7 ± 1.5 | 21.5 ± 0.7 | 25.4 ± 0.7 |
| w64 d6 N10 all | select256 | 35.1 ± 1.5 | 41.0 ± 3.1 | 53.7 ± 1.1 | 18.9 ± 1.3 | 23.3 ± 0.1 | 28.0 ± 0.6 |
| w64 d6 N10 all | mlp | 33.6 ± 2.5 | 48.4 ± 0.6 | 65.1 ± 0.7 | 17.5 ± 0.6 | 23.9 ± 0.3 | 33.6 ± 0.2 |
| w64 d20 N10 all | linear | 36.4 ± 1.5 | 49.5 ± 0.9 | 62.3 ± 0.4 | 20.5 ± 1.4 | 24.8 ± 0.4 | 32.5 ± 0.6 |
| w64 d20 N10 all | select64 | 28.6 ± 2.0 | 36.8 ± 0.5 | 45.9 ± 1.4 | 17.3 ± 1.6 | 21.1 ± 0.8 | 25.8 ± 0.2 |
| w64 d20 N10 all | select256 | 33.1 ± 1.9 | 39.9 ± 0.5 | 53.2 ± 1.6 | 17.9 ± 0.7 | 24.0 ± 0.4 | 28.8 ± 0.5 |
| w64 d20 N10 all | mlp | 30.1 ± 2.9 | 42.0 ± 0.3 | 59.5 ± 0.8 | 17.8 ± 0.1 | 22.8 ± 0.6 | 31.3 ± 0.4 |
| w64 d20 N40 all | linear | 35.6 ± 0.8 | 50.9 ± 1.3 | 65.4 ± 0.2 | 19.5 ± 2.7 | 26.1 ± 0.3 | 33.9 ± 0.4 |
| w64 d20 N40 all | select64 | 30.5 ± 1.9 | 37.1 ± 0.9 | 47.9 ± 0.9 | 17.5 ± 1.0 | 21.7 ± 0.8 | 25.0 ± 0.6 |
| w64 d20 N40 all | select256 | 33.8 ± 0.3 | 42.6 ± 1.1 | 53.9 ± 1.0 | 18.6 ± 0.6 | 22.6 ± 1.0 | 29.9 ± 0.3 |
| w64 d20 N40 all | mlp | 30.3 ± 0.6 | 40.9 ± 1.9 | 58.8 ± 0.6 | 17.7 ± 0.6 | 22.6 ± 0.4 | 32.0 ± 0.6 |
| w64 d6 N40 all | linear | 43.2 ± 1.4 | 55.9 ± 0.4 | 68.2 ± 0.6 | 18.7 ± 2.3 | 27.2 ± 0.5 | 34.6 ± 0.5 |
| w64 d6 N40 all | select64 | 27.8 ± 2.7 | 37.9 ± 0.7 | 48.5 ± 1.8 | 17.3 ± 0.5 | 21.2 ± 1.1 | 25.0 ± 0.2 |
| w64 d6 N40 all | select256 | 35.1 ± 0.8 | 42.5 ± 0.8 | 54.4 ± 1.4 | 17.9 ± 1.3 | 23.1 ± 1.2 | 29.2 ± 0.2 |
| w64 d6 N40 all | mlp | 33.2 ± 2.2 | 45.7 ± 1.1 | 65.7 ± 0.0 | 18.3 ± 0.6 | 23.6 ± 0.6 | 34.4 ± 0.4 |
| dinov2_b | linear | 99.2 ± 0.1 | 99.3 ± 0.1 | 99.4 ± 0.1 | 93.0 ± 0.4 | 94.3 ± 0.4 | 95.2 ± 0.1 |
| dinov2_b | select64 | 97.4 ± 0.5 | 98.0 ± 0.1 | 98.1 ± 0.1 | 92.4 ± 0.4 | 92.9 ± 0.3 | 93.8 ± 0.2 |
| dinov2_b | select256 | 98.7 ± 0.4 | 99.0 ± 0.0 | 99.2 ± 0.0 | 93.6 ± 0.1 | 94.2 ± 0.2 | 94.9 ± 0.1 |
| dinov2_b | mlp | 99.2 ± 0.1 | 99.3 ± 0.1 | 99.5 ± 0.0 | 93.3 ± 0.1 | 94.3 ± 0.3 | 95.5 ± 0.2 |

### Depth diagnostic (Imagenette, 10 nets, width 64; mean and max pooled per channel)


| block | mean cosine between images | relative spread (median std / mean) | effective dims of 128, standardised (participation ratio) |
|---|---|---|---|
| 1 | 0.9689 | 0.431 | 6.0 |
| 2 | 0.9704 | 0.334 | 8.0 |
| 3 | 0.9704 | 0.351 | 8.6 |
| 4 | 0.9688 | 0.364 | 9.0 |
| 6 | 0.9690 | 0.342 | 6.6 |
| 10 | 0.9798 | 0.305 | 4.1 |
| 20 | 0.9887 | 0.270 | 2.4 |
| 40 | 0.9938 | 0.250 | 1.7 |

Same, residual wiring (same weights):


| block | mean cosine between images | relative spread (median std / mean) | effective dims of 128, standardised (participation ratio) |
|---|---|---|---|
| 1 | 0.9689 | 0.431 | 6.0 |
| 2 | 0.9704 | 0.334 | 8.0 |
| 3 | 0.9704 | 0.351 | 8.6 |
| 4 | 0.9688 | 0.364 | 9.0 |
| 6 | 0.9675 | 0.315 | 5.6 |
| 10 | 0.9720 | 0.305 | 4.5 |
| 20 | 0.9765 | 0.300 | 3.7 |
| 40 | 0.9821 | 0.287 | 2.8 |
