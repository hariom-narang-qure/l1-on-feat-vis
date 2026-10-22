"""A self-contained HTML review of the best head on held-out test slices.

Trains the pooled64+ct head exactly as seg_head.py does (same seed, data and
threshold rule), saves its weights, and renders per test scan its best, median
and worst slice by tolerant F1 (slices with >= 300 sheet pixels):
  CT | ground-truth sheet | predicted probability | errors
Errors use the 2 px tolerance of the metrics:
  - green: predicted pixels within 2 px of the sheet (true positives)
  - orange: predicted pixels with no sheet within 2 px (false positives)
  - magenta: sheet pixels with no prediction within 2 px (false negatives)
Images are cropped to the lungs and embedded as base64 PNGs.
"""

import argparse
import base64
import html
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from l1_on_feat_vis.fissure_probe.data.data import list_scans
from l1_on_feat_vis.fissure_probe.sheet_probing.probes import split_scans
from l1_on_feat_vis.fissure_probe.segmentation.seg_head import (TOLERANCE_PX, HeadTrainer, SliceSet, best_threshold, ct_input,
                                                   f1_from_counts, tolerant_counts)

VARIANT = "pooled64+ct"
GREEN, ORANGE, MAGENTA, RED = (40, 200, 80), (255, 150, 0), (230, 40, 200), (240, 50, 50)


def lung_box(lung, margin=12):
    xs, ys = np.nonzero(lung)
    size = lung.shape[0]
    return max(xs.min() - margin, 0), min(xs.max() + margin, size), max(ys.min() - margin, 0), min(ys.max() + margin, size)


def to_png(rgb, box, width=360):
    x0, x1, y0, y1 = box
    crop = rgb[x0:x1, y0:y1].transpose(1, 0, 2)  # rows = y, the usual axial display
    image = Image.fromarray(crop.astype(np.uint8))
    image = image.resize((width, int(width * image.height / image.width)), Image.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode()


def grey_rgb(hu):
    grey = (ct_input(hu) * 255).astype(np.float32)
    return np.stack([grey] * 3, -1)


def paint(base, mask, colour, alpha=1.0):
    out = base.copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(colour)
    return out


def panels(hu, sheet, lung, prob, threshold):
    ct = grey_rgb(hu)
    prediction = (prob >= threshold) & ndimage.binary_dilation(lung, iterations=3)
    near_sheet = ndimage.binary_dilation(sheet, iterations=TOLERANCE_PX)
    near_prediction = ndimage.binary_dilation(prediction, iterations=TOLERANCE_PX)
    heat = ct * 0.45
    heat[..., 1] += 255 * 0.55 * np.clip(prob, 0, 1) * lung
    errors = paint(ct * 0.6, prediction & near_sheet, GREEN)
    errors = paint(errors, prediction & ~near_sheet, ORANGE)
    errors = paint(errors, sheet & ~near_prediction, MAGENTA)
    return {"ct": ct, "truth": paint(ct, sheet, RED), "prob": np.clip(heat, 0, 255), "errors": errors}


def slice_scores(trainer, test, threshold):
    rows = []
    for i in range(len(test)):
        if test.sheet[i].sum() < 300:
            continue
        prob = trainer.predict(test, i)
        f1, precision, recall = f1_from_counts(tolerant_counts(prob > threshold, test.sheet[i], test.lung[i]))
        rows.append({"index": i, "scan": str(test.scan_of[i]), "f1": f1, "precision": precision, "recall": recall, "prob": prob})
    return rows


def pick_examples(rows):
    picked = []
    for scan in sorted({r["scan"] for r in rows}):
        ranked = sorted([r for r in rows if r["scan"] == scan], key=lambda r: -r["f1"])
        picks = {"best": ranked[0], "median": ranked[len(ranked) // 2], "worst": ranked[-1]}
        picked += [dict(r, label=label) for label, r in picks.items()]
    return picked


PAGE = """<title>Fissure Segmentation Review</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a reading column of slice cards, four panels per card (two at phone width). */
:root {{
  --bg: #f6f7f6; --surface: #ffffff; --ink: #1b1f1d; --muted: #59615d; --rule: #dde2df;
  --truth: rgb{red}; --tp: rgb{green}; --fp: rgb{orange}; --fn: rgb{magenta};
  --sans: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #121513; --surface: #1a1e1c; --ink: #e7ebe8; --muted: #a3aca7; --rule: #2c322f; color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --bg: #121513; --surface: #1a1e1c; --ink: #e7ebe8; --muted: #a3aca7; --rule: #2c322f; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--ink); font: 15px/1.55 var(--sans); }}
main {{ max-width: 1440px; margin: 0 auto; padding-block: 28px 56px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
h1 {{ font-size: 24px; font-weight: 600; margin: 0 0 6px; text-wrap: balance; }}
.lead {{ color: var(--muted); max-width: 72ch; margin: 0 0 18px; }}
.summary {{ display: flex; flex-wrap: wrap; gap: 10px 28px; margin: 0 0 14px; font-family: var(--mono); font-variant-numeric: tabular-nums; }}
.summary div {{ display: flex; flex-direction: column; }} .summary span {{ font: 12px var(--sans); color: var(--muted); letter-spacing: .02em; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px 20px; margin: 0 0 22px; font-size: 13px; color: var(--muted); }}
.sw {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }}
.cards {{ display: grid; gap: 14px; }}
.card {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 12px; min-width: 0; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 4px 14px; align-items: baseline; margin: 0 0 10px; font-size: 14px; }}
.meta b {{ font-weight: 600; font-family: var(--mono); }} .meta .tag {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }}
.meta .nums {{ font-family: var(--mono); font-variant-numeric: tabular-nums; color: var(--muted); }}
.grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }}
figure {{ margin: 0; min-width: 0; }} figcaption {{ font-size: 12px; color: var(--muted); margin-top: 3px; }}
img {{ width: 100%; max-width: 100%; height: auto; border-radius: 6px; display: block; background: #000; }}
@media (max-width: 760px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
</style>
<main>
<h1>Fissure segmentation on held-out scans</h1>
<p class="lead">The best head so far: 64 units pooled from 7 pretrained networks, plus a CT skip connection, trained on 12 scans.
Every slice here comes from the 5 test scans the head never saw: for each scan, its best, median and worst slice by F1.
Ground truth is the waypoint agent's fissure sheet. A pixel counts as correct within {tol} px of the sheet.</p>
<div class="summary">
<div><span>Test slices</span>{slices}</div><div><span>Pixel AP</span>{ap:.3f}</div><div><span>F1 ({tol} px)</span>{f1:.3f}</div>
<div><span>Precision</span>{precision:.3f}</div><div><span>Recall</span>{recall:.3f}</div><div><span>Threshold</span>{threshold:.2f}</div></div>
<div class="legend"><span><i class="sw" style="background:var(--truth)"></i>Ground-truth sheet</span>
<span><i class="sw" style="background:var(--tp)"></i>Correct prediction</span>
<span><i class="sw" style="background:var(--fp)"></i>False positive: no sheet within {tol} px</span>
<span><i class="sw" style="background:var(--fn)"></i>False negative: sheet missed</span>
<span>Probability panel: brighter green is higher</span></div>
<div class="cards">
{rows}
</div>
</main>"""

ROW = """<section class="card"><div class="meta"><b>Scan {scan}</b><span class="tag">{cohort}</span><span class="tag">{label} slice</span>
<span class="nums">z {z} · F1 {f1:.2f} · P {precision:.2f} · R {recall:.2f}</span></div><div class="grid">
<figure><img alt="CT of scan {scan}, slice {z}" src="data:image/png;base64,{ct}"><figcaption>CT, lung window</figcaption></figure>
<figure><img alt="Ground-truth fissure sheet" src="data:image/png;base64,{truth}"><figcaption>Ground truth</figcaption></figure>
<figure><img alt="Predicted fissure probability" src="data:image/png;base64,{prob}"><figcaption>Predicted probability</figcaption></figure>
<figure><img alt="Errors: true positives, false positives, false negatives" src="data:image/png;base64,{errors}"><figcaption>Errors</figcaption></figure>
</div></section>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("outputs/fissure_probe/seg_cache"))
    parser.add_argument("--multi-cache", type=Path, default=Path("outputs/fissure_probe/multi_cache"))
    parser.add_argument("--out", type=Path, default=Path("outputs/fissure_probe/seg_review"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weights", type=Path, help="a saved head (head_pooled64_ct.pt) to render without retraining")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    train_scans, test_scans, _ = split_scans()
    train = SliceSet(args.cache, train_scans, VARIANT, None, args.multi_cache)
    test = SliceSet(args.cache, test_scans, VARIANT, None, args.multi_cache)
    trainer = HeadTrainer(train, torch.device(args.device))
    if args.weights:
        saved = torch.load(args.weights, map_location=args.device, weights_only=False)
        trainer.model.load_state_dict(saved["state_dict"])
        trainer.token_mean, trainer.token_std, threshold = saved["token_mean"], saved["token_std"], saved["threshold"]
    else:
        trainer.fit()
        threshold = best_threshold(trainer, train)
    torch.save({"state_dict": trainer.model.state_dict(), "token_mean": trainer.token_mean, "token_std": trainer.token_std,
                "threshold": threshold, "variant": VARIANT}, args.out / "head_pooled64_ct.pt")

    from l1_on_feat_vis.fissure_probe.segmentation.seg_head import evaluate
    summary = evaluate(trainer, test, threshold)
    cohort = {r.short: r.cohort for r in list_scans()}
    rows = []
    for example in pick_examples(slice_scores(trainer, test, threshold)):
        i = example["index"]
        images = panels(test.hu[i], test.sheet[i], test.lung[i], example["prob"], threshold)
        box = lung_box(test.lung[i])
        rows.append(ROW.format(scan=html.escape(example["scan"]), cohort=cohort.get(example["scan"], "?"),
                               z=int(np.load(args.cache / f"{example['scan']}.npz")["z"][i - int(np.flatnonzero(test.scan_of == example['scan'])[0])]),
                               label=example["label"], f1=example["f1"], precision=example["precision"], recall=example["recall"],
                               **{k: to_png(v, box) for k, v in images.items()}))
    page = PAGE.format(threshold=threshold, tol=TOLERANCE_PX, slices=len(test), ap=summary["pixel_ap"], f1=summary["tolerant_f1"],
                       precision=summary["tolerant_precision"], recall=summary["tolerant_recall"], rows="\n".join(rows),
                       red=RED, green=GREEN, orange=ORANGE, magenta=MAGENTA)
    (args.out / "fissure_segmentation_review.html").write_text(page)
    (args.out / "summary.json").write_text(json.dumps({k: v for k, v in summary.items()}, indent=1))
    print(f"test AP {summary['pixel_ap']:.3f} F1 {summary['tolerant_f1']:.3f} | page {args.out / 'fissure_segmentation_review.html'}")


if __name__ == "__main__":
    main()
