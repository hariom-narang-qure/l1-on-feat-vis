# Fissure neurons: handoff

As of 2026-10-21. Repo: `/home/hariom.narang/l1-on-feat-vis` (nothing is committed yet; `git status` shows every file untracked).
Read with: `fissure-neurons.md` (findings so far), `experiment-log.md` (the cat feature-vis work that came first).

## Goal

- Find units ("neurons") in pretrained networks that respond to lung fissures on CT.
- Use them as inputs to a small segmentation head.
- Raise the head's accuracy four ways:
  1. **More models.** Add more pretrained networks.
  2. **All layers.** Probe every layer, not a sample of layers.
  3. **Recall finesse.** Add units that fire where the head misses fissures (false negatives).
  4. **Precision finesse.** Add units that fire where the head predicts fissure wrongly (false positives).
- Grow the unit pool in controlled chunks, keep the units that generalise, and prune the ones that overfit.

## Where things stand

- **Best result so far, on the old ground truth:** a head on 64 units pooled from 7 networks, plus a CT skip connection.
  - Held-out test: pixel AP 0.729, tolerant F1 (2 px) 0.822, precision 0.86, recall 0.78.
  - Review page: <https://claude.ai/artifact/KgsuK94MVZp2e2uqRPxS1Z>. It is private; the owner shares it from its Share menu.
- **The ground truth has changed.** Every number above used `post/sheet.nii.gz`. That sheet is interpolated across views. Where it runs almost within an axial slice, it shows as a filled patch instead of a line.
- **The new ground truth** is the agent's raw lines in each view:
  - `fissures_axial.nii.gz`, `fissures_coronal.nii.gz`, `fissures_sagittal.nii.gz`
  - on the slices that view annotated (every 4th slice)
- **What to keep and what to redo:**
  - Everything must be re-run on the new ground truth.
  - The code and the method carry over.
  - The old results are a reference only.

### Findings that carry over

- **Models Genesis has no fissure units.** It matches its random-init twin and loses to a Hessian sheet filter.
- **DINOv2 (with registers) has fissure units.**
  - Base layer 8: residual units 449, 355, 442 and MLP neuron 2235.
  - Evidence: top-48 precision 83–100%, erase/insert tests, and the same model with random init scores chance.
  - Details are in `fissure-neurons.md`.
- **Pooling across models beats any single model.**
  - The 7 models: DINOv2-S/B/L, RAD-DINO, ConvNeXt-B, ResNet-50, TotalSegmentator nnU-Net.
  - Head AP: 4 units 0.465, 32 pooled 0.649, 64 pooled 0.725.
  - All 768 dimensions of one DINOv2 layer reach 0.562.
- **Location control is essential.**
  - Without it, random-init networks look like fissure detectors: fissures run through the lung interior and the pleura lines the edge.
  - Every ranking uses fissure vs "ring" (lung 3–8 voxels from the fissure).
- **The pooled units have no causal evidence yet.** Only the 4 DINOv2 units passed erase/insert. The pooled 64 are "units with fissure signal", not proven fissure neurons.
- **Sparse activation matching does not isolate the fissure line** (`fissure_vis.py`, 2 slices tested).
  - The unit maps can be matched from smooth tissue plus small seed-specific edits.
  - Parked.

## Data

- **Scans:** 19 under `/home/hariom.narang/fissure-annotation/work/scans/<uid>/flex_step4/`.
  - 9 "review v1" scans (run 19 Oct).
  - 10 "severe" scans (run 16 Oct).
  - Ignore `work/scans/_runs` and any `step3` / `step4` folders.
- **Per scan:**
  - `scan.nii.gz`: CT in HU. Thick-slice scans were already resampled along z.
  - `lobes.nii.gz`: TotalSegmentator lobes, labels 10–14. Used as the lung mask.
  - `fissures_<view>.nii.gz`: the agent's lines in that view (**new ground truth**).
  - `fissures_work/<view>/run/logs/slice_NNNN.json`: one log per annotated slice. The logs list which slices are annotated.
  - `post/sheet.nii.gz`: the old ground truth. Do not use it.
- **Axial annotation was checked on all 19 scans:**
  - every 4th slice is annotated (67–220 per scan)
  - lines appear only on annotated slices
  - 1–87 annotated slices per scan have a line
- **Coronal has logs every 4th slice too** (checked on one scan). Sagittal is not checked yet.
- **Split** (`probes.split_scans`, fixed seed):
  - train: `.28402 .76856 125572 126945 271645 418292 427544 609892 658984 756888 809039 932180`
  - test: `208477 563738 659399 773438 840568`
  - near-empty (scored separately): `269479 818602`
  - Two scan ids start with a dot. `ls *.npz` hides them; use `ls -a`.

## Code map (`src/l1_on_feat_vis/fissure_probe/`)

| Module | What it does | Reuse? |
|---|---|---|
| `data.py` | Scan list, cohorts, label maps (fissure, ring, pleura, lobe boundary), patch sampler | Yes. Switch labels to the per-view line files. |
| `genesis.py` | Models Genesis 3D U-Net, strict-key load | Only for completeness |
| `features.py`, `probes.py` | First Genesis probing pass (slow sklearn probes) | `probes.py` helpers are reused (split, BlockRows, l1_entry_order) |
| `dino_features.py` | DINOv2-B on axial slices, shared 74×74 token grid, labels per token | Its grid and labelling helpers are reused |
| `slice_features.py` | Any 2D network on axial slices (ViT via HF, conv via timm), random-init twins | Yes. Extend to all layers and all views. |
| `nnunet_features.py` | Any nnU-Net v2 model folder. Crops the lungs, preprocesses from the model's plans, runs one forward, samples each stage at the same tokens. Replays the 2D sampler so rows line up | Yes |
| `unit_scan.py` | Scores every unit of a block on the GPU (AUROC and AP from ranks), top-64 sparse probe | Yes. This is the fast scorer. |
| `pooled_selection.py` | Cross-model candidates (top 8 per block), then the L1 path over the pool | Replace with chunked selection and pruning |
| `multi_cache.py` | Full 74×74 maps of chosen units on the head's slices (2D and 3D models) | Yes |
| `seg_head.py` | Slice cache, the CT-skip head, variants, metrics (pixel AP, tolerant F1) | Yes. Switch labels and views. |
| `seg_review.py` | Trains or loads the best head, renders the static HTML review page | Yes |
| `dino_evidence.py`, `dino_top_activations.py` | Causal tests and top activations for 2D units | Yes, for the final units |
| `evidence.py` | Genesis 3D version of the evidence tests | No |
| `fissure_vis.py` | Sparse activation matching on CT slices | Parked |
| `miss_analysis.py` | **Draft, not run.** Out-of-fold misses and unit search. Its `pool_scores` function is broken. | Rewrite in Phase 5 |

Other assets:
- `weights/Genesis_Chest_CT.pt`
- TotalSegmentator weights in `~/.totalsegmentator/` (task 291, lobes)
- Hugging Face caches for the 2D models
- `outputs/fissure_probe/seg_review/head_pooled64_ct.pt`: best head on the old ground truth
- About 25 GB of old-ground-truth features in `outputs/fissure_probe/`. Delete them once the new store exists.

## Plan

### Phase 0: labels and slices

1. **Check every view.**
   - Coronal and sagittal on all 19 scans, the same way axial was checked.
   - Logs every 4th slice.
   - Lines only on annotated slices.
2. **Build one table of annotated slices**: (scan, view, slice index, has_line).
3. **Per-view labels on each annotated slice**, all in 2D within the slice:
   - positive = line pixels, dilated 1 px
   - ring = lung 3–8 px from a line (2D distance)
   - lung = lobes > 0
   - Line-free annotated slices are all negatives.
   - Unannotated slices are never used.
4. **Resample coronal and sagittal slices to square pixels** before 2D models see them.
   - Their z spacing can differ from in-plane spacing.
   - Map labels the same way.
5. **Sanity figure**: a few slices per view with their lines overlaid.

### Phase 1: activation store (`/realdata/hariom.narang/fissure_acts/`)

- **Rows:** sampled cells on annotated slices of all 3 views.
  - About 8 slices per view per scan: line slices plus line-free ones.
  - Per slice: all line cells, their ring cells, and a uniform lung sample.
  - One shared row table: scan, view, slice, token, cohort, fold, labels.
- **Columns:** every unit of every layer of every model, pretrained only.
  - 2D ViTs: every residual-stream layer, plus MLP neurons for the small and base ViTs.
  - Conv nets: every block output.
  - nnU-Nets: every conv output, from all three views' slices of the same 3D forward.
  - One memory-mapped matrix per (model, layer), aligned to the row table.
  - Random-init controls are computed on demand for chosen units only.
- **Budget:**
  - `/realdata` has about 240 GB free (98% full).
  - All layers of the current 7 models are about 85k channels.
  - That is about 90 GB in fp16 or about 45 GB in int8 with a per-channel scale. Prefer int8.
  - Keep total use under 100 GB.
- **Check a new model before trusting it.** TotalSegmentator, for example, reproduced `lobes.nii.gz` at Dice 0.96–0.98 from our preprocessing.

### Phase 2: score every unit with cross-validation

- 3 folds of 4 train scans.
- Per unit, per fold, per view: fissure-vs-ring AUROC and AP.
- Generaliser = strong in every fold. Overfitter = strong in one fold only. Flag overfitters and exclude them.

### Phase 3: grow the pool (strategies 1 and 2)

- **Chunked forward selection.**
  - Add 8–16 units at a time by cross-validated gain.
  - Use a token-level logistic probe; it is cheap.
- **Backward pruning**: drop units whose removal does not hurt held-out folds.
- **Order:**
  1. the 7 current models at all layers
  2. then new models one at a time, each kept only if it raises the cross-validated curve
- **Candidate new models, in order:**
  1. TotalSegmentator lung vessels (likely false-positive suppression)
  2. VoxTell's image encoder
  3. SAM's ViT image encoder
  4. DINOv3, if downloadable
  5. STU-Net or other pretrained nnU-Nets
- **Output:** a curve of pool size against cross-validated score.

### Phase 4: segmentation head

- The same CT-skip head (`seg_head.py`).
- Trained on all three views together, scored per view.
- Pool checkpoints, e.g. 64, 128 and 256 units.
- Full maps are recomputed only for chosen units (`multi_cache.py`).

### Phase 5: finesse (strategies 3 and 4)

- **Errors come from out-of-fold head predictions on the train scans** (train on 8 scans, predict the other 4, rotate).
- **Recall:** units that fire on missed line cells vs their ring cells.
- **Precision:** units that fire on false-positive cells vs true line cells. They are used as negative evidence.
- **Each candidate must:**
  - pass in every fold
  - add cross-validated gain over the current pool
  - be flagged `found_on_misses` or `found_on_false_positives`
- **Check the top recall hits visually.** The agent probably misses faint fissures, so some "misses" may be real fissures that were labelled negative.

### Phase 6: final test (once)

- Strategy table: baseline, then + all layers, then + new models, then + recall units, then + precision units.
- Each row has a random-unit control of the same size.
- The test scans are scored once per milestone and never used for selection.

### Phase 7: evidence and reporting

- Top-activation sheets and erase/insert tests for the final top units.
- Update `fissure-neurons.md`.
- Republish the review page (`seg_review.py`, static HTML).

## Pitfalls already hit

- **The GPUs are shared** with other users' training jobs, often both at 100%.
  - Expect 3–10× slowdowns.
  - Check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` before blaming code.
- **CPU torch defaults to 56 threads**, which was about 50× slower than 8 on these models. Always set `torch.set_num_threads(8)`.
- **Lucent hard-codes `cuda:0`** in its transforms and colour decorrelation. Use `CUDA_VISIBLE_DEVICES` to pick a GPU, or `""` for CPU.
- **Piping into `tail` holds back all output until exit**, so progress looks frozen. Log to a file instead.
- **`pkill -f <pattern>` can kill your own shell** when the pattern appears in the command line. Use PIDs.
- **Dot-prefixed scan ids** (`.28402`, `.76856`) are hidden by `ls *`.
- **Row alignment across feature folders relies on replaying the same random draws in the same order** (`SelectionReplay`). Changing the scan order or the sampling code breaks it. `pooled_selection.Pool.check_alignment` asserts it.
- **Slow paths:**
  - sklearn L1 probes over 3000+ units took hours; `unit_scan.py` does it in seconds
  - 3D nnU-Net forwards plus label zooms take 30–100 s per scan
- **Cohort is inferred from the date of `status.json`** (`data.cohort_of`). It breaks if the files are touched.

## Open decisions (proposed defaults)

1. Storage: int8, under 100 GB on `/realdata`. (Default: yes.)
2. Head: one head on all three views. (Default: yes, scored per view.)
3. Model order for additions as listed above. (Default: yes.)
4. Commit the code to git before Phase 0. The repo has no commits yet; it is the user's call.
5. Uploading docs to qSafe needs the qSafe tools connected (`/mcp`, server `default`).
