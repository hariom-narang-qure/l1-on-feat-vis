# Fissure neurons on per-view ground truth: experiment log

Started 2026-10-21. Plan and context: `fissure-handoff.md`. Earlier findings (old ground truth): `fissure-neurons.md`.
Code: `src/l1_on_feat_vis/fissure_probe/` (new modules listed per phase). Data store: `/realdata/hariom.narang/fissure_acts/`.

**Goal.** Raise the segmentation accuracy of the CT-skip head by selecting units ("neurons") from many pretrained
networks that detect fissures reliably, without overfitting.

## Decisions (agreed with the user, 2026-10-21)

- Ground truth: the agent's raw lines `fissures_{axial,coronal,sagittal}.nii.gz`, on annotated slices only (a log exists).
  `fissures.nii.gz` (OR of the views) and `post/sheet.nii.gz` (interpolated) are not used.
- Every annotated slice that has a line is used. Line-free slices are not used at all for now.
- The agent is trusted: every lung pixel that is not line is a negative, including vessels and pleura.
- Metrics (per view): tolerant Dice on skeletons and clDice (headline); completeness and missed segments (recall
  side); correctness and false segments by type vessel / pleura / other (precision side); exact pixel Dice; pixel AP
  for unit selection.
- New models come last, once the full pipeline (all layers, CV scoring, grow / prune, head, recall / precision
  units) works on the 7 current models.
- Slice rules (below) were confirmed before Phase 0 started.

## Phase 0: slices and labels (`views.py`, `view_figures.py`)

How the agent saw each slice (from the annotation code, `fissure_annotation/imaging/slices.py`):
`volume[plane].T[::-1, ::-1]` on the grid of `flex_step4/scan.nii.gz`; axial anterior-up, coronal / sagittal
superior-up; no in-plane resampling, so coronal / sagittal of the 11 scans with native z are stretched
(z / x from 0.64 to 1.35; the 8 z-resampled scans are exactly isotropic).

Canvas rules: rows resampled to the column spacing (CT linear, lobes nearest, lines redrawn segment by segment);
a fixed 512 × 512 window centred on the lung's extent along the rows (padded with air), so one token is 6.9 px
in every view. Labels per canvas: line, positive (line + 1 px), lung (lobes grown 3 px), ring (3–8 px from the
line), pleura (±2 px of the lobes' outer edge), vessel (inside lobes, HU > −500, > 2 px from the line, not pleura).

| Check | Result |
|---|---|
| Our cut of `fissures_<view>` vs the agent's own mask PNGs | 0 mismatches on all 2394 line slices that have a PNG |
| Line pixels on slices without a log | 0 in every view |
| Annotation cadence | every 4th slice of each axis plus the last; coronal / sagittal 129 logs per scan |
| Line connectivity after row resampling (3 scans, f = 0.64, 1.35, 1.36) | no slice gains components |
| Lung clipped by the 512 window | never; 126945 (875 slices) and 269479 lose only lines far outside the lung |

**Finding: 5.6% of line pixels lie outside the lung.** 17k of 303k pixels are outside the lobes grown by 3 px,
and 120 slices have no line inside the lung. They concentrate in severe scans (840568 30%, 418292 22%, 208477 20%).
On inspection (`outputs/fissure_rerun/outside_lung_lines.png`) they are agent errors on edge slices: lines along
the chest wall, fat planes and the diaphragm dome. **Decision:** keep line pixels inside the grown lung only; drop
slices with no line left.

**Result: 2274 line slices** (axial 710, coronal 758, sagittal 806), cached as canvases + label bitmaps in
`fissure_acts/slices/<scan>.npz` (1.8 GB). Sanity figure: `outputs/fissure_rerun/sanity_views.png`.

## Phase 1: row table and activation store (`rows.py`, `store.py`)

**Rows** (train scans only; test is never used for selection). On the 74 × 74 token grid of each canvas, per slice:
all line tokens, ring tokens (≤ 3 × line), 40 pleura, 40 vessel, 80 uniform lung tokens (with a natural-frequency
weight). 302,696 rows: 44k line, 36k ring, 63k pleura, 46k vessel, 127k uniform. Folds: 3 × 4 scans, each 2
review_v1 + 2 severe (line rows per fold 20k / 15k / 9k).

**Columns**: every layer of the 7 networks, 171 layers, 176,793 units:
DINOv2-S / B / RAD-DINO residual + MLP of every layer, DINOv2-L residual of every layer, ConvNeXt-B every block
(36), ResNet-50 every bottleneck (16), TotalSegmentator organs nnU-Net every conv block (22) + organ logits.
int8 per (layer, scan) with per-unit min / max. One scan takes 3–5 min on a shared GPU.

Checks: the nnU-Net's own lobe logits sampled at the rows' voxels agree with `lobes.nii.gz` at 92.5% of in-lobe
rows (rows cluster at lobe boundaries; a misaligned grid would be far lower); canvas → voxel mapping agrees with
the canvas lobes at 99.7%.

## Phases 2–5: method (code written before the store finished)

- `cv_scores.py`: per unit × fold × view: line-vs-ring / vessel / pleura AUROC and natural-frequency AP.
  Generaliser = min-fold ring AUROC ≥ 0.70; overfitter = max ≥ 0.75 but min < 0.60.
- `forward_select.py`: token-level weighted logistic probe (Newton on GPU). Chunks of 16 units chosen by the
  held-out score statistic, minimum over the 3 held-out folds (a unit that helps one fold only is never picked),
  decorrelated (|r| ≤ 0.9 within a chunk); drop-one CV pruning every 4 chunks. Curve = CV AP vs pool size.
- `seg_metrics.py`: skeleton completeness / correctness / tolerant Dice (2 px), clDice, pixel Dice, pixel AP,
  missed / false segments, false segments typed vessel / pleura / other. Checks: a 2 px shift keeps tolerant
  Dice 0.98 (clDice and pixel Dice ≈ 0.5); adding vessels drops correctness to 0.22, all typed vessel.
- `head.py`: the CT-skip head on canvases of all three views, 4000 steps, threshold by train tolerant Dice.
- `finesse.py`: OOF head errors mapped to store rows; recall units (missed vs ring) and precision units
  (false positive vs found line), each ≥ 0.70 AUROC in every fold, then a stacked held-out score test
  against the head's own cell score.

## Phase 2: every unit scored with 3-fold CV (`cv_scores.py`)

Store done: 12 train scans × 171 layers × 176,793 units, 50 GB int8. Scoring all units: ~8 s per layer on GPU.
Natural line-token prior per fold: 2.3–3.1% (axial 1.8–2.6%, coronal 2.1–2.6%, sagittal 3.2–4.0%).

| Finding | Numbers |
|---|---|
| Single units separate line tokens from ring tokens only modestly | best min-fold ring AUROC 0.698 (DINOv2-B mlp6); 91 units ≥ 0.65, 2804 ≥ 0.60 of 176,793 |
| Best layers | DINOv2-B / L middle layers (res6–15, mlp6–9), DINOv2-S res7–8, ConvNeXt s1–s2 |
| Best natural-frequency AP lift (min over folds) | 9.5× (DINOv2-B mlp6), 8.1× (TotalSegmentator), 7.5× (DINOv2-S mlp8) |
| The old fissure unit DINOv2-B res 449 | again the best unit of res10 (min-fold ring 0.650, AP lift 7.6×) |
| Folds agree | almost no unit is strong in one fold only |
| RAD-DINO | weakest: 10 generalisers |

Why ring AUROCs are lower than on the old ground truth (0.83–0.90 then): a line token is any token holding a
line pixel, and ring tokens (3–8 px) are its direct neighbours on a 6.9 px grid, whose receptive fields
overlap the line. The old 3D sheet was thicker and smoother. The contrast is still the right one for ranking.

**Thresholds set from the distribution** (0.70 would leave none): generaliser = min-fold ring AUROC ≥ 0.60 or
min-fold AP lift ≥ 3×; overfitter = strong in one fold, weak in another (ring max ≥ 0.65 & min < 0.55, or lift
max ≥ 3 & min < 1.5). **3270 generalisers** (DINOv2-L 1202, DINOv2-B 888, ConvNeXt 531, DINOv2-S 449,
ResNet 108, TotalSegmentator 82, RAD-DINO 10); **176 overfitters excluded**.

## Phase 3: forward selection over all layers (`forward_select.py`) — running

Token-level CV AP of the probe (prior 0.026) as the pool grows (chunks of 16; prune every 4 chunks):

| step | pool | CV AP | folds | axial / coronal / sagittal | note |
|---|---|---|---|---|---|
| 2 | 32 | 0.606 | 0.689 / 0.607 / 0.521 | 0.461 / 0.657 / 0.652 | |
| 4 | 48 | 0.643 | | 0.493 / 0.693 / 0.691 | pruned 16 |
| 8 | 84 | 0.669 | | 0.520 / 0.721 / 0.712 | pruned 28 |
| 12 | 111 | 0.676 | | 0.522 / 0.729 / 0.724 | pruned 37 |
| 16 | 132 | 0.681 | | 0.530 / 0.734 / 0.727 | pruned 43 |
| 20 | 147 | 0.686 | | 0.538 / 0.741 / 0.728 | pruned 49 |
| 24 | 159 | 0.691 | | 0.547 / 0.741 / 0.731 | pruned 52 |
| 28 | 168 | 0.692 | 0.757 / 0.699 / 0.621 | 0.550 / 0.746 / 0.730 | pruned 55 |

Additions between prunes lower the held-out AP slightly (first-order score tests over-predict), and every
prune removes a quarter of the pool without loss and raises it: the add / prune cycle is doing the work. The
curve is flattening (~+0.002 per cycle). Axial is the hardest view for the token probe.

## Phase 4: heads

Test scans are scored once per milestone; pool sizes are chosen by out-of-fold CV on the train scans.

| Run | Units | Test tolerant Dice (C / K) | clDice | pixel Dice | pixel AP | axial / coronal / sagittal | near-empty |
|---|---|---|---|---|---|---|---|
| CT only | 0 | 0.187 (0.124 / 0.385) | 0.118 | 0.106 | 0.134 | 0.231 / 0.231 / 0.115 | 0.044 |
| **E0** old recipe (30 blocks, top-8, L1 path) | 64 | **0.760** (0.717 / 0.809) | 0.727 | 0.640 | 0.680 | 0.795 / 0.727 / 0.755 | 0.273 |

E0 per test scan: 208477 0.69, 563738 0.82, 659399 0.67, 773438 0.80, 840568 0.41. Errors: 26% of GT segments
missed (19% of length); false segments are 48% of predicted segments but 15% of predicted length, so most are
short fragments, typed mostly "other" (1382) rather than vessel (174) or pleura (14).

**Selection stopped at step 32** (pool 174, CV AP 0.694; step 28: 168 units, 0.692): the curve had flattened to
~+0.002 per add / prune cycle. Checkpoints for heads: step 8 (84 units), step 16 (132), step 28 (168).

Token-level CV AP of fixed pools (same probe, same folds):

| Pool | Units | Token CV AP | axial / coronal / sagittal | ring AUROC |
|---|---|---|---|---|
| E0 old recipe (selected on all train, so slightly optimistic) | 64 | 0.589 | 0.475 / 0.607 / 0.650 | 0.767 |
| Random generalisers (control) | 168 | 0.594 | 0.449 / 0.646 / 0.649 | 0.749 |
| Forward selection, step 8 | 84 | 0.669 | 0.520 / 0.721 / 0.712 | 0.788 |
| Forward selection, step 28 | 168 | 0.692 | 0.550 / 0.746 / 0.730 | 0.804 |

The old recipe is no better than 168 random generalisers at the token level; CV forward selection over all layers
adds +0.08–0.10 AP. Composition of the 168: DINOv2-L 55 (mostly res11–17), DINOv2-B 41 (mlp6–10), ConvNeXt 32,
DINOv2-S 19, TotalSegmentator 14, ResNet 7, RAD-DINO 0. The first picks are TotalSegmentator decoder units
(dec1c1, dec2c1): a lobe-boundary location prior that the 2D units then sharpen.

Out-of-fold CV on the train scans (train on 8, predict 4, rotate; each fold tunes its own threshold and
minimum component size on its training slices). This is what pool sizes are chosen by:

| Run | Units | OOF tolerant Dice (C / K) | per fold | axial / coronal / sagittal |
|---|---|---|---|---|
| E0 old recipe | 64 | 0.728 (0.685 / 0.776) | 0.773 / 0.720 / 0.633 | 0.646 / 0.763 / 0.747 |
| Random generalisers (control) | 168 | 0.723 (0.657 / 0.804) | 0.772 / 0.717 / 0.608 | 0.642 / 0.762 / 0.737 |
| E1 all layers, step 8 | 84 | 0.733 (0.705 / 0.763) | 0.777 / 0.723 / 0.646 | 0.656 / 0.762 / 0.754 |
| E1 all layers, step 16 | 132 | 0.734 (0.694 / 0.779) | 0.777 / 0.729 / 0.647 | 0.656 / 0.768 / 0.753 |
| E1 all layers, step 28 | 168 | 0.736 (0.706 / 0.769) | 0.781 / 0.735 / 0.642 | 0.658 / 0.769 / 0.755 |

**Finding: at the head level, which units go in matters much less than at the token level.** Token CV AP:
E1 0.692 vs random 0.594 (+0.10); head OOF tolerant Dice: E1 0.736 vs random 0.723 (+0.013) vs E0 0.728,
with E1 ahead in every fold. The CT branch recovers most of what weaker units miss. Look at OOF errors
(`outputs/fissure_rerun/oof_errors_step28.png`): probability maps are clean; many "false" segments look like
real fissures the agent did not trace on that slice (e.g. the other lung), and several "misses" are jagged agent
lines along vessels or in diseased lung, so label noise caps these numbers.

The head OOF curve is flat from 84 to 168 units (0.733 → 0.736).
Seed noise: step 28 with seed 1 gives 0.742 (0.784 / 0.741 / 0.650), so ±0.005 between seeds; a 2-seed
average of the OOF maps gives 0.743, no better than the best seed. The per-fold tuned thresholds range from
0.3 to 0.9 with min component sizes 10–80 px: the threshold landscape is flat.

## Phase 5: recall and precision units (`finesse.py`)

From the step-28 head's OOF predictions mapped onto store rows: 11,725 of 44,309 line rows missed (per fold
5066 / 3517 / 3142), 2,458 false-positive rows (714 / 1111 / 633). A fixed 0.70 AUROC bar passed no unit
(best min-fold 0.646 recall, 0.629 precision), so the bar is 0.60 in every fold, as in Phase 2.

- **Recall** (missed line vs ring): 317 units pass. Top by stacked held-out score: DINOv2-L res13 unit 374
  (fold AUROC 0.648 / 0.655 / 0.646; the same unit also in res12 and res14), ResNet l2b2 370, and several
  low-level units with negative sign (DINOv2-L res0 582, DINOv2-B mlp0 1493, DINOv2-S mlp0 647): early
  layers that drop on the faint line, i.e. intensity / contrast detectors for faint fissures.
- **Precision** (false positive vs found line): 66 units pass. Mostly ConvNeXt stage 2 (unit 466 in s2b3, b7,
  b12; unit 129 / 48 / 4 in other blocks) and ResNet l3b0 881. They fire on what the head wrongly calls fissure.
- 16 of each are added (decorrelated, |r| ≤ 0.9), flagged `found_on_misses` / `found_on_false_positives`.

## Finding: the agent's views agree with each other only about half the time (`view_agreement.py`)

A fissure is a surface, so where a coronal line crosses an annotated axial slice, the axial line should pass there
too. Fraction of view-B line voxels (inside lung) on view-A annotated slices that lie near A's line, all 19 scans:

| B lines on A slices | voxels | within 2 px | within 4 px |
|---|---|---|---|
| coronal on axial | 22,800 | 0.393 | 0.441 |
| sagittal on axial | 28,028 | 0.423 | 0.467 |
| axial on coronal | 19,399 | 0.476 | 0.527 |
| sagittal on coronal | 27,902 | 0.565 | 0.611 |
| axial on sagittal | 19,472 | 0.556 | 0.595 |
| coronal on sagittal | 22,211 | 0.628 | 0.663 |

Reading: each view's lines cover roughly half of the fissure that the other views trace through the same slice;
axial is the least complete (≈ 40%), which matches axial being the head's weakest view (OOF 0.66 vs 0.76–0.78).
So per-view "false positives" are often real fissure, and correctness / tolerant Dice against one view are capped
well below 1. As agreed, the per-view ground truth stays the metric; the final test adds a diagnostic of how many
false segments another view confirms.

Phase 5 head OOF CV (seed 0; reference step 28 = 0.736 seed 0, 0.742 seed 1):

| Pool | Units | OOF tolerant Dice | completeness | correctness | per fold | axial / coronal / sagittal |
|---|---|---|---|---|---|---|
| step 28 (seed 0 / 1) | 168 | 0.736 / 0.742 | 0.706 / 0.703 | 0.769 / 0.787 | 0.781 0.735 0.642 / 0.784 0.741 0.650 | 0.658 / 0.769 / 0.755 |
| + 16 recall units | 184 | 0.739 | **0.711** | 0.770 | 0.786 / 0.738 / 0.642 | 0.656 / 0.777 / 0.757 |
| + 16 precision units | 184 | 0.739 | 0.692 | **0.792** | 0.781 / 0.736 / 0.643 | 0.663 / 0.774 / 0.754 |
| + both | 200 | **0.742** | 0.701 | 0.787 | 0.787 / 0.736 / 0.644 | 0.665 / 0.779 / 0.755 |
| random generalisers (seed 0 / 1) | 168 | 0.723 / 0.721 | 0.657 / 0.656 | 0.804 / 0.800 | | |

Each kind moves the metric it targets (recall units: completeness up; precision units: correctness up), but the
total gains (+0.003 to +0.006) are inside the ±0.005 seed noise. The selection gain over random generalisers is
consistent: +0.017 averaged over two seeds each (0.739 vs 0.722); random pools are more conservative
(completeness 0.66 vs 0.70).

**Cross-view check of "false" segments (`cross_view.py`, E0 test head, seed 0).** Other views' line voxels in each
test slice's plane are independent fissure evidence. A predicted segment is "confirmed" if such a point lies
within 2 px:

| View | true segments confirmed | false segments | false confirmed (count) | false confirmed (length) |
|---|---|---|---|---|
| axial | 0.85 of 510 | 459 | 0.38 | 0.65 |
| coronal | 0.85 of 472 | 532 | 0.41 | 0.65 |
| sagittal | 0.85 of 732 | 579 | 0.32 | 0.60 |
| all | 0.85 of 1714 | 1570 | 0.37 | **0.63** |

63% of the "false-positive" length is confirmed by another view's tracing (true segments: 85%, the ceiling of
this check given sparse cross points). Most of what the per-view ground truth counts as false segmentation is
fissure that this view's agent did not trace; the unconfirmed remainder is mostly short fragments.

## Phase 6: strategy table on the test scans (`strategy_table.py`)

5 test scans, every row trained on all 12 train scans, two seeds each, scored once; rows were fixed by train-only CV before any test run. Tolerant Dice at 2 px on 1 px skeletons; near-empty = the 2 scans with almost no fissure.

| Row | units | seeds | tolerant Dice (± half range) | completeness | correctness | clDice | pixel Dice | pixel AP | axial / coronal / sagittal | near-empty |
|---|---|---|---|---|---|---|---|---|---|---|
| CT only | 0 | 1 | 0.187 (±0.000) | 0.124 | 0.385 | 0.118 | 0.106 | 0.134 | 0.231 / 0.231 / 0.115 | 0.044 |
| E0 old recipe (30 blocks, top-8, L1 path) | 64 | 2 | 0.765 (±0.004) | 0.726 | 0.810 | 0.750 | 0.675 | 0.685 | 0.810 / 0.728 / 0.754 | 0.224 |
| random generalisers, 84 (control) | 84 | 2 | 0.728 (±0.002) | 0.655 | 0.818 | 0.709 | 0.633 | 0.633 | 0.773 / 0.702 / 0.708 | 0.315 |
| E1 all layers, CV forward selection, 84 | 84 | 2 | 0.766 (±0.001) | 0.714 | 0.827 | 0.749 | 0.673 | 0.684 | 0.798 / 0.737 / 0.760 | 0.263 |
| random generalisers, 168 (control) | 168 | 2 | 0.752 (±0.001) | 0.702 | 0.810 | 0.732 | 0.656 | 0.653 | 0.800 / 0.723 / 0.733 | 0.366 |
| E1 all layers, 168 | 168 | 2 | 0.771 (±0.001) | 0.733 | 0.813 | 0.753 | 0.678 | 0.692 | 0.807 / 0.739 / 0.762 | 0.316 |
| E1 168 + 16 recall units | 184 | 2 | 0.772 (±0.001) | 0.710 | 0.846 | 0.753 | 0.670 | 0.689 | 0.812 / 0.739 / 0.760 | 0.301 |
| E1 168 + 16 precision units | 184 | 2 | 0.774 (±0.003) | 0.737 | 0.816 | 0.755 | 0.683 | 0.694 | 0.814 / 0.739 / 0.766 | 0.320 |
| random generalisers, 200 (control) | 200 | 2 | 0.756 (±0.000) | 0.707 | 0.813 | 0.737 | 0.663 | 0.659 | 0.805 / 0.724 / 0.738 | 0.352 |
| E1 168 + recall + precision units, 200 | 200 | 2 | 0.766 (±0.002) | 0.709 | 0.835 | 0.752 | 0.676 | 0.687 | 0.804 / 0.737 / 0.755 | 0.271 |

Reading the table:
- **Selection beats random units of the same size every time**: +0.038 at 84 units (0.766 vs 0.728),
  +0.019 at 168 (0.771 vs 0.752), +0.010 at 200 (0.766 vs 0.756). Seed spread is ±0.001–0.004.
- **E1 (all layers, CV forward selection) vs E0 (old recipe): +0.001 at 84, +0.006 at 168.** E0 is itself a
  selection (best ring-AUROC units of good blocks, then the L1 path), and on the head it is nearly as good. The
  large token-level gap (CV AP 0.692 vs 0.589) mostly does not survive the CT-skip head.
- **Recall / precision units: +0.001 / +0.003, both: −0.005, all inside seed noise.** On test they even move
  the opposite metric to the one they target (recall units: completeness 0.710 vs 0.733, correctness 0.846;
  precision units: completeness 0.737, correctness 0.816), whereas in train OOF they moved the targeted one.
  So neither the total gain nor its direction is robust: the head's misses and false positives are not
  explained by any unit beyond what the pool already carries (best min-fold AUROC for misses 0.65). Best row: E1 168 + 16 precision units, test tolerant Dice 0.774 (±0.003),
  pixel AP 0.694, axial 0.814 / coronal 0.739 / sagittal 0.766.
- Axial is the best view on test (0.80–0.81) but the worst in train OOF (0.66): the test scans happen to have
  cleaner axial labels; coronal is the hardest test view.
- Near-empty scans stay poor for every head (0.22–0.37): little fissure to find and it is mostly diseased lung.

## Phase 7: evidence for the selected units (`unit_evidence.py`)

The first 16 units of the step-28 pool (entry order), on the store rows. Top-48: share of a unit's 48 strongest
natural-frequency rows (line + uniform lung) on the line (prior 0.026) or on the pleura band. Pretraining control:
the same unit read from a random-init twin at the same rows of 6 train scans (one review_v1 + one severe per
fold), min over folds. Sheet: `outputs/fissure_rerun/top_activations.png`.

| Unit | sign | top-48 on line | top-48 on pleura | ring AUROC / AP lift, pretrained | random-init twin | role |
|---|---|---|---|---|---|---|
| ts_organs/dec1c1 20 | + | 0.85 | 0.00 | 0.522 / 4.0× | 0.495 / 0.9× | fissure (lobe boundary) |
| convnext_b/s2b0 301 | + | 0.92 | 0.00 | 0.600 / 3.5× | 0.502 / 1.1× | fissure |
| ts_organs/dec2c1 11 | + | 0.92 | 0.00 | 0.578 / 4.3× | 0.506 / 0.9× | fissure (lobe boundary) |
| ts_organs/dec1c1 151 | + | 0.88 | 0.02 | 0.516 / 2.5× | 0.496 / 0.9× | fissure (lobe boundary) |
| dinov2_s/res7 6 | − | 0.60 | 0.02 | 0.603 / 1.4× | 0.516 / 0.7× | mixed |
| dinov2_b/res4 229 | − | 0.08 | 0.79 | 0.637 / 1.5× | 0.519 / 0.7× | pleura (negative evidence) |
| dinov2_b/mlp7 873 | + | 0.90 | 0.00 | 0.626 / 3.5× | 0.507 / 0.6× | fissure |
| convnext_b/s1b2 244 | + | 0.06 | 0.92 | 0.605 / 1.2× | 0.500 / 0.6× | pleura (negative evidence) |
| convnext_b/s2b4 477 | − | 0.88 | 0.02 | 0.618 / 1.9× | 0.497 / 0.8× | fissure |
| dinov2_l/res10 134 | + | **0.96** | 0.00 | 0.596 / 2.5× | 0.539 / 0.7× | fissure |
| resnet50/l3b3 441 | + | 0.92 | 0.02 | 0.598 / 2.2× | 0.484 / 0.9× | fissure |
| convnext_b/s2b16 272 | − | 0.94 | 0.04 | 0.573 / 3.3× | 0.519 / 1.2× | fissure |
| dinov2_b/mlp9 1051 | − | 0.33 | 0.29 | 0.601 / 0.9× | 0.531 / 1.1× | mixed / context |
| dinov2_s/res4 98 | + | 0.08 | 0.90 | 0.602 / 1.0× | 0.529 / 0.7× | pleura (negative evidence) |
| dinov2_b/res8 643 | − | 0.60 | 0.29 | 0.599 / 2.5× | 0.512 / 0.9× | mixed |
| dinov2_l/res17 683 | + | 0.73 | 0.00 | 0.616 / 1.7× | 0.494 / 0.7× | fissure-leaning |

- **Nine of the 16 are fissure detectors in 5 networks** (top-48 on the line 85–96%, ~35× the prior):
  TotalSegmentator decoder units (lobe-boundary detectors), ConvNeXt stage 2, DINOv2-B MLP 7, DINOv2-L
  res10, ResNet layer 3.
- **Four are pleura detectors** (79–92% of their top cells on the pleura band) that the probe uses as negative
  evidence; selection by held-out gain keeps them because they suppress the pleura, the line-like structure
  fissure units confuse.
- **Every random-init twin scores chance** (ring AUROC 0.48–0.54, AP lift 0.6–1.2×): the signals are learned
  in pretraining, not architecture plus location.
- Causal erase / insert tests (old `dino_evidence.py`) were not re-run for these units; they remain
  "units with fissure signal that depends on pretraining", not causally proven fissure neurons.

## Conclusions

1. **The re-run on per-view ground truth is complete** for the 7 current networks: 2274 line slices in 3
   views, 176,793 units of 171 layers scored with 3-fold CV, chunked forward selection with pruning, CT-skip
   heads, recall / precision search, a test table with random controls of equal size, and pretraining controls.
2. **Best head: test tolerant Dice 0.771–0.774** (E1 168 units, ± 16 precision units; pixel AP 0.69,
   clDice 0.75), vs 0.765 for the old recipe and 0.187 for CT alone. All numbers are against single-view
   ground truth that covers only about half the fissure (cross-view agreement 39–63%); 63% of the head's
   "false-positive" length is confirmed by another view's tracing.
3. **Selecting units reliably beats random units without overfitting**: at every pool size selected units beat
   random generalisers (+0.010 to +0.038 test, +0.017 OOF), overfitters are excluded by construction (fold-min
   statistics), and the token-level CV AP rises from 0.59 to 0.69. But the CT-skip head closes most of the gap,
   so E1 beats the old recipe by only +0.006, and recall / precision units add nothing beyond seed noise.
4. **What limits the head now is the labels, not the units**: the OOF curve is flat from 84 to 200 units,
   seed ensembles do not help, and the errors that remain are dominated by fissure the per-view labels omit.

## Next steps (not done)

- New models (TotalSegmentator lung vessels, VoxTell, SAM, DINOv3, STU-Net) through the same pipeline; the
  pipeline is ready (`store.py` takes new model specs; only new layers need extracting). Expect small gains:
  false segments are mostly "other", not vessel (≈ 40 vessel vs ≈ 250 other per test run).
- A multi-view ground truth (union of the three views' lines per slice, or ignoring pixels near other views'
  lines) for training and evaluation; the cross-view check says this is where most of the remaining
  "error" is.
- Erase / insert tests for the nine fissure units above.
- Delete the old-ground-truth features (`outputs/fissure_probe/`, 25 GB) once no longer needed.

## Baseline: frozen DINOv2 final-layer embeddings (added 2026-10-22)

The simplest alternative to selecting units: every dimension of the last DINOv2 block's tokens (residual
stream after block 11 of DINOv2-B / block 23 of DINOv2-L, before the final LayerNorm), frozen, into the same
CT-skip head, same training, threshold / post-processing tuning, test scans and two seeds as the table above.

| Head input | units | test tolerant Dice | completeness | correctness | pixel AP | axial / coronal / sagittal |
|---|---|---|---|---|---|---|
| DINOv2-B final layer | 768 | 0.643 (±0.002) | 0.525 | 0.828 | 0.547 | 0.721 / 0.598 / 0.605 |
| DINOv2-L final layer | 1024 | 0.693 (±0.008) | 0.605 | 0.811 | 0.576 | 0.748 / 0.661 / 0.669 |
| random generalisers (control) | 168 | 0.752 | 0.702 | 0.810 | 0.653 | 0.800 / 0.723 / 0.733 |
| E1 selected units | 168 | 0.771 | 0.733 | 0.813 | 0.692 | 0.807 / 0.739 / 0.762 |

- **Selected units beat the frozen final layer by +0.08 (DINOv2-L) and +0.13 (DINOv2-B)** with 6× fewer inputs;
  even random generalisers from many layers beat it (+0.06). The final layer is too abstract: the fissure
  signal sits in middle layers (DINOv2-B res6–10, DINOv2-L res10–17) and in other networks.
- The loss is almost all recall (completeness 0.53–0.61 vs 0.73); correctness is the same. The final-layer
  heads simply find fewer fissures, worst on coronal / sagittal.
- Not yet compared: a fine-tuned DINOv2 (LoRA or last blocks unfrozen), and the best whole middle layer.

## Baselines: one whole middle layer, and randomly initialised networks (added 2026-10-22)

**One whole layer.** DINOv2-L res17, the layer that contributes most units to the random-generaliser pool
(103 generalisers), all 1024 dims frozen into the same head: test tolerant Dice 0.707 (0.725 / 0.689), vs 0.693
for DINOv2-L's final layer and 0.771 for selected units. Token level, same probe: whole res17 0.526 (best
ridge 0.541), DINOv2-L final 0.468, DINOv2-B final 0.456; all 3270 pretrained generalisers at once 0.677
(best ridge 0.1); selected 168 units 0.692. Gradient descent on the screened pool nearly matches the search;
a single layer, however regularised, does not: the signal is spread over many layers and networks.

**Random networks** (`models/random_convs.py`): 10 nets, 6 blocks of conv 3x3 + ReLU, 64 channels, strides
1, 2, 2, 2, 2, 1, He-normal weights, never trained; 3840 units, scored and selected like the pretrained ones.

| Measure | Random nets | Pretrained |
|---|---|---|
| Units passing the per-fold screen | 0 of 3840 (best min-fold ring AUROC 0.589, AP lift 2.0×) | 3270 of 176,793 |
| Token CV AP, all units in one probe | 1 net 0.128, 2 nets 0.144, 5 nets 0.172, 10 nets 0.187 | 3270 generalisers 0.677 |
| Token CV AP, 180 forward-selected units | 0.131 | 168 selected units 0.692 |
| Head, test tolerant Dice (2 seeds) | **0.385** (0.364 / 0.406); completeness 0.26, correctness 0.75 | 0.771 (selected), 0.752 (random generalisers) |

Random features double what CT alone gives (0.187 → 0.385) but find only a quarter of the fissure length.

**20 random networks** (seeds 0–19, 7680 units; same selection to 180 units, same head): token CV AP of the
selected units 0.154 (10 nets: 0.131); **head test tolerant Dice 0.441** (0.457 / 0.425; completeness 0.32,
correctness 0.73), vs 0.385 with 10. Doubling the networks gained +0.056 at the head, more than the token probe
suggested (~+0.02 per doubling). If the head keeps gaining ~0.05 per doubling, matching selected pretrained units
(0.771) would take roughly 6 more doublings (~1000+ networks); a 40-net run would test whether the gain holds.

**40 random networks** (seeds 0–39, 15,360 units): selected units' token CV AP 0.173, but **head test
tolerant Dice 0.346** (0.359 / 0.333), below both 20 nets (0.441) and 10 nets (0.385). The gain did not hold:
the probe keeps improving with more networks, the head does not. Seed averaging removes the head's training
noise but not the variance of *which* 180 units get selected and of the 5-scan test set (±~0.05 between runs),
so the 20-net result was most likely a favourable draw. With this setup more random networks give no reliable
gain at the head; they stay at 0.35–0.44 vs 0.77 for selected pretrained units.
The fissure signal is mostly from pretraining (every selected unit's random-init twin scores chance).

## Reproduce

All under `uv run python -m l1_on_feat_vis.fissure_probe.<subpackage>.<module>`; set `CUDA_VISIBLE_DEVICES`
(GPUs are shared).

| Step | Module | Output under `/realdata/hariom.narang/fissure_acts/` |
|---|---|---|
| canvases + labels | `data.views` (`data.view_figures` for sanity sheets) | `slices/<scan>.npz`, `slices/report.json` |
| row table | `data.rows` | `rows.npz` |
| activation store | `store.store` (two processes: add `--reverse` to the second) | `acts/<model>/<layer>/<scan>.npy` (50 GB) |
| unit scores | `selection.cv_scores` | `scores/*.npz` |
| forward selection | `selection.forward_select --name all_layers`; `selection.old_recipe` for E0 | `selection/*.json` (`pool_at` rebuilds a step's pool) |
| unit maps | `store.unit_maps --units <selection json>` | `maps/<layer>/<unit>.npy` |
| heads | `segmentation.head --units <json> --k 0 --seeds 0 1 --name ...` (`--oof` for train CV) | `heads/*.json`, `*.pt` |
| recall / precision | `selection.finesse --oof <tag> --pool <json> --name ...` | `selection/finesse_*.json` |
| whole-layer baseline | `selection.whole_layers` | `selection/whole_layers.json` |
| test table | `segmentation.strategy_table` | printed |
| diagnostics | `data.view_agreement`, `segmentation.cross_view --head <pt>`, `evidence.unit_evidence --pool <json>` | `heads/*_cross_view.json`, `evidence/` |

Code layout (`src/l1_on_feat_vis/`): `fissure_probe/` holds `data`, `models`, `store`, `selection`,
`segmentation`, `evidence` and `sheet_probing` (the first pass on the interpolated sheet; a few of its helpers
are still imported by the stages above); `feature_vis/` holds the InceptionV1 feature-visualisation work.
