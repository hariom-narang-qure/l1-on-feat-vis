"""Fissure units in pretrained networks, and a segmentation head built on them.

Subpackages, in pipeline order (see fissure-rerun-log.md):
  data          scans, cohorts, per-view canvases and labels (views), the store's row table (rows)
  models        network wrappers shared across stages (Models Genesis, nnU-Net)
  store         the int8 activation store over every layer (store) and full unit maps for heads (unit_maps)
  selection     CV unit scores, chunked forward selection, the old recipe, recall / precision search
  segmentation  the CT-skip head, its training, line metrics, the test strategy table
  evidence      pretraining controls, top activations, causal (erase / insert) tests
  sheet_probing the first probing pass on the interpolated sheet (old ground truth); a few helpers
                (split_scans, auroc_columns, input constants) are still used by the stages above
"""
