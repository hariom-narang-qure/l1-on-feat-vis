# Fissure neurons in pretrained vision models

## Update 2026-10-22: re-run on the per-view ground truth

Full log: `fissure-rerun-log.md`. Ground truth is now the agent's raw lines per view
(`fissures_{axial,coronal,sagittal}.nii.gz`, annotated slices with a line, 2274 slices); everything below this
section used the old interpolated `post/sheet.nii.gz` and is a reference only.

- **All layers of the 7 networks** (176,793 units) were scored with 3-fold CV; 3270 generalise, 176 overfit.
- **Nine selected units are fissure detectors in 5 networks** (85–96% of their 48 strongest cells on the line,
  prior 2.6%; random-init twins at chance): TotalSegmentator organs decoder dec1c1 20 / 151 and dec2c1 11,
  ConvNeXt-B s2b0 301 / s2b4 477 / s2b16 272, DINOv2-B mlp7 873, DINOv2-L res10 134, ResNet-50 l3b3 441.
  Pleura detectors (DINOv2-B res4 229, ConvNeXt s1b2 244, DINOv2-S res4 98) are selected as negative evidence.
- **Old DINOv2-B res 449 is still a top unit** (best of res10, AP lift 7.6×), but on per-view lines every single
  unit's line-vs-ring AUROC is lower (best 0.698) than on the old sheet.
- **Best head**: 168 CV-selected units (+16 precision units) + CT skip, test tolerant Dice 0.771–0.774, vs 0.765
  for the old pooled-64 recipe and 0.752 for 168 random generalisers.
- **The labels now limit the head**: the three views agree only 39–63%, and 63% of the head's false-positive
  length is fissure that another view traced.
- Not re-run: erase / insert causal tests for the new units.

## Earlier findings (old ground truth, 2026-10-21)

Code: `src/l1_on_feat_vis/fissure_probe/`. Outputs: `outputs/fissure_probe/`.

**Answer.**
- Models Genesis has no unit that looks at fissures.
  - Its best channels carry the same signal as a randomly initialised Genesis.
  - They do no better than a Hessian sheet filter.
- DINOv2 (`facebook/dinov2-with-registers-base`, run on axial lung-window slices) has four units that do. All four are in layer 8.
  - Residual units 449, 355 and 442.
  - MLP neuron 2235.
- On held-out scans, their strongest responses land on fissures 83–100% of the time.
  - About 6% of lung tokens are fissure.
- That holds against two fissure sources:
  - the agent sheets they were selected on
  - TotalSegmentator lobe boundaries, which they never saw
- For three of them, erasing the fissure from the CT removes their response, and the same erase elsewhere does not.
- The same architecture with random weights scores exactly chance.

## Final set of units

All numbers are on the 5 held-out test scans (3 review v1, 2 severe).
- Ring AUROC: fissure tokens vs lung tokens 3–8 voxels away. 0.5 is chance.
- Top-48 precision: share of a unit's 48 strongest lung tokens that lie on the fissure.
- Fissure AP: chance is the fissure prior, 0.057.

| Unit | What it is | Ring AUROC | Fissure AP | Top-48 on agent sheet | Top-48 on lobe boundary | Erasing the fissure | Synthetic thin line | Top 10 in splits |
|---|---|---|---|---|---|---|---|---|
| `dino_res8` 449 (fires negative) | Fissure region: the band around a fissure, not the line's brightness | **0.728** | 0.260 | **98%** | 98% | Response does not drop | Only at 2× fissure contrast | 4/4 |
| `dino_res8` 355 (fires negative) | Fissure line detector | 0.685 | 0.158 | 83% | 83% | Drops in 94% of slices (p = 2e-12); control unchanged | Dose-dependent; also fires on thicker lines | 4/4 |
| `dino_res8` 442 (fires negative) | Thin-line selective fissure detector | 0.658 | 0.207 | 83% | 90% | Drops in 84% of slices (p = 2e-8) | Prefers a thin line over a thick line and a disc (p < 1e-5) | 3/4 |
| `dino_mlp8` 2235 | Fissure neuron: needs the real fissure, not just any line | 0.674 | **0.306** | **100%** | **100%** | Drops in 86% of slices (p = 2e-9); control unchanged | Barely responds | 2/4 |

- Residual unit = one dimension of the 768-d token after block 8.
- MLP neuron = one of the 3072 GELU units inside block 8's MLP.
- "Fires negative" = the unit marks fissure by going down, so its sign is flipped to read it.
- Unit 357 is borderline and left out.
  - Ring AUROC is 0.666 and 69% of its top 48 are on fissure.
  - But its erase effect is not significant (p = 0.1).
- The other units the sparse probe picked failed the causal tests.
  - Residual units 258, 674, 373 and 275.
  - MLP neurons 1, 69, 332, 443, 1464, 918 and 2517.

Figure: `outputs/fissure_probe/fissure_units_top_activations.png`. It shows each unit's top tokens on held-out slices.
- The red band is the sheet.
- The yellow box is the token.

## Segmentation from the four units

A head trained on only these 4 units, plus the CT through a skip connection, segments fissures on held-out scans.
- Pixel AP is 0.465, about 31× the 1.5% prior.
- Tolerant F1 is 0.589, with precision 0.79.
- The same head fed 4 random layer-8 units reaches 0.275 AP.

Code: `src/l1_on_feat_vis/fissure_probe/seg_head.py`. Results: `outputs/fissure_probe/seg_head/`.

**Head:**
- **Token branch:** the units' 74×74 maps go through a 1×1 conv to 16 channels, then bilinear upsampling to 512 px.
- **CT branch:** the lung-window CT goes through two 3×3 convs, 16 channels each.
- **Merge:** the two branches are concatenated, then three 3×3 convs and a 1×1 give the logit.
- **Why the CT branch is shallow:** with a 5 px receptive field before the merge, it can find a bright line but not learn where fissures are.

**Training:**
- 12 train scans, 30 fissure-rich plus 10 other axial slices each, with 256 px crops.
- Loss: BCE plus soft Dice inside the lung.
- 2500 steps.

**Test:** every sampled slice of the 5 test scans, with the threshold picked on train slices.
- Tolerant F1: a pixel counts if it is within 2 px of the sheet.

| Head input (same network) | Pixel AP | Tolerant F1 | Precision | Recall | F1 on the 5 test scans |
|---|---|---|---|---|---|
| **4 fissure units + CT** | **0.465** | **0.589** | **0.79** | 0.47 | 0.56, 0.62, 0.50, 0.75, 0.38 |
| 4 random layer-8 units + CT | 0.275 | 0.483 | 0.52 | 0.45 | 0.51, 0.43, 0.46, 0.66, 0.36 |
| 4 fissure units only | 0.272 | 0.495 | 0.56 | 0.44 | 0.45, 0.51, 0.51, 0.61, 0.29 |
| CT only | 0.169 | 0.369 | 0.33 | 0.42 | 0.41, 0.17, 0.26, 0.56, 0.31 |
| All 768 layer-8 dims + CT (upper bound) | 0.562 | 0.688 | 0.80 | 0.60 | 0.60, 0.72, 0.72, 0.78, 0.54 |

- **The 4 units beat 4 random units on every test scan.**
  - Random DINOv2 units add some context over CT alone.
  - The fissure units add much more, mostly as precision: they suppress vessels.
- **They recover about two thirds of the full layer's gain over CT alone.**
  - AP rises 0.30 of the full layer's 0.39.
  - That uses 4 of 768 dimensions.
- **The skip connection matters:** units alone reach 0.272 AP, and adding the CT reaches 0.465.
- **Weak spots:**
  - Scan 840568 (F1 0.38), which has the sparsest sheet.
  - Dense or diseased slices, where all variants pick up texture (bottom row of `seg_head/test_slices_labelled.png`).
- **Recall is the limit, at 0.47.** The head misses faint fissures that the units do not flag.

## Evidence, and what each piece rules out

1. **Held-out selection.**
   - Units are picked and signed on 12 train scans and scored on 5 test scans.
   - Choosing among 768 or 3072 units on the test data would inflate the scores.
   - Each unit's best rank is stable across 4 different train/test splits.
2. **Location control (ring AUROC).**
   - A plain detector score also rewards *where* a cell is.
     - Fissures run through the lung interior.
     - On Genesis, a random-init network scored as well as the pretrained one.
   - Comparing fissure tokens with tokens 3–8 voxels away holds the surroundings fixed.
   - The best layer-8 unit reaches 0.728.
   - The layer's 99th percentile is 0.667 and its median is 0.549.
3. **Random-init control.**
   - The same DINOv2 with random weights is at chance in every block. Its best unit scores 0.535.
   - So the signal comes from pretraining, not from the architecture or the input.
4. **Baselines.**
   - Intensity scores 0.500 ring AUROC.
   - A 2D Hessian ridge filter scores 0.528.
5. **Independent label.**
   - The units were selected on the agent's sheets.
   - Their top activations land on TotalSegmentator lobe boundaries just as often (83–100%).
   - So they are not fitting quirks of the agent's tracer.
6. **Erase (input intervention).**
   - On 50 held-out slices, the sheet's pixels were replaced with the 7×7 median around them.
   - Units 355, 442 and 2235 drop in 84–94% of slices.
   - The same erase on a shifted copy of the mask barely moves them (p ≤ 5e-4 for fissure vs control).
   - Unit 449 does not drop. It responds to the fissure's surroundings, likely the vessel-free band along a fissure.
7. **Insert (input intervention).**
   - A 60 px synthetic line at the measured fissure contrast (74 HU) was added to fissure-free lung.
   - It raises 355 and 442, with dose response (0.25× to 2×).
   - 442 prefers it over:
     - a thick line of the same peak
     - a thick line of the same mass
     - a disc of the same peak (a vessel in cross-section)
   - 2235 and 449 barely respond. They need the real fissure context, not any line.
8. **Both cohorts.**
   - The units score above chance on 4 or 5 of the 5 test scans, in both review v1 and severe.
   - On the near-empty severe scan 269479 they sit at chance. That is expected, since its sheet has 1983 voxels.

## Models Genesis: why it was dropped

- **Setup:** native spacing, 128×128×64 patches, all 7 U-Net blocks, probes as above.
- **Loading checked:**
  - The checkpoint loads with strict keys.
  - In-lung reconstruction MSE is 0.0004–0.007, against a lung variance of 0.005–0.010.
- **Best single channel:** ring AUROC 0.623 (`down_tr128` 105).
  - Random-init Genesis reaches 0.53–0.60.
  - The Hessian sheet filter reaches 0.61–0.67 at full and half resolution.
- **Probes:** 8-channel probes reach 0.60–0.69, against 0.52–0.63 at random init.
- **Plain detector scores look better** (AP 0.57 against a 0.165 prior in `down_tr512`). The random-init network gets close (0.44). That signal is location, not fissure.
- **Likely reasons:**
  - The [-1000, 1000] HU window leaves a fissure about 0.05 above parenchyma.
  - Pre-training was restoration, not semantics.

## Data and labels

- **Scans:** 19 under `fissure-annotation/work/scans/<uid>/flex_step4/`.
  - 9 review v1 scans (19 Oct).
  - 10 severe scans (16 Oct).
- **Labels:** `post/sheet.nii.gz`, dilated by one voxel.
  - These are machine labels: LLM waypoints plus a Hessian ridge tracer.
  - That is why the lobe-boundary cross-check matters.
- **Agreement between the two sources:** on one scan, 57% of agent fissure voxels lie on a TotalSegmentator lobe boundary, and 38% of the boundary is covered by the agent.
- **Split:** train 12, test 5, near-empty 2 (severe scans with fewer than 3000 sheet voxels).
- **DINOv2 input:**
  - Axial slices, lung window [-1000, -200] HU.
  - Resized 512 → 1036 px, so one token is about 6.9 px (a 74×74 grid).
  - 10 slices per scan, 6 of them fissure-rich.

## Caveats

- Tokens are about 6.9 px, so these units localise fissures to about 5 mm, not to the voxel.
- Unit 449 is a fissure-*region* unit, so on its own it would make a blurry detector.
  - 355 and 442 respond to the line.
  - 442 is the most line-specific.
- Only axial slices were tested. The horizontal fissure lies in plane on axial slices and is likely under-represented.
- Only layer 8 was tested causally. Layer 11 has the same top unit (449) with similar scores.
- Labels come from a tracer, and the test set has 5 scans. Treat the p-values as per-slice, not per-patient.

## Reproduce

```
uv run python -m l1_on_feat_vis.fissure_probe.features                    # Genesis features
uv run python -m l1_on_feat_vis.fissure_probe.dino_features               # DINOv2 features
uv run python -m l1_on_feat_vis.fissure_probe.unit_scan --features outputs/fissure_probe/dino_features \
    --blocks dino_res8 dino_mlp8 --out outputs/fissure_probe/dino_units  # rank and score every unit
uv run python -m l1_on_feat_vis.fissure_probe.dino_evidence --block dino_res8 --units 449 355 357 258 674 442 373 275
uv run python -m l1_on_feat_vis.fissure_probe.dino_top_activations --block dino_res8 --units 449 355 442 357
```

- Genesis weights go in `weights/Genesis_Chest_CT.pt`, from [Hugging Face](https://huggingface.co/MrGiovanni/ModelsGenesis).
