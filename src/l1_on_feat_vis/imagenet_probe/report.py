"""Markdown tables of the grid: val top-1 accuracy, mean +- sd over the 3 replicates.

Also a depth diagnostic, to see whether deep random blocks wash images out: per block, the mean cosine
between raw pooled vectors of different images, the cross-image spread of each feature, and the effective
dimensions the standardised features keep (participation ratio of the correlation-matrix eigenvalues).
Deep plain ReLU nets at init are known to push all inputs towards one direction (Poole et al. 2016,
"Exponential expressivity"; Schoenholz et al. 2017, "Deep information propagation").
Usage: python -m l1_on_feat_vis.imagenet_probe.report > outputs/imagenet_probe/tables.md
"""

import json
from collections import defaultdict

import numpy as np

from l1_on_feat_vis.imagenet_probe.features import DATASETS, random_path
from l1_on_feat_vis.imagenet_probe.run import OUT, SHOTS

READOUTS = ("linear", "select64", "select256", "mlp")
DATASET_NAMES = {"nette": "Imagenette", "woof": "Imagewoof"}


def load_results():
    results = []
    for path in sorted(OUT.glob("results_*.jsonl")):
        results += [json.loads(line) for line in path.read_text().splitlines()]
    return results


def row_id(row):
    if row["source"] != "random":
        return row["source"]
    arch = "" if row.get("arch", "plain") == "plain" else f"{row['arch']} "
    return f"{arch}w{row['width']} d{row['depth']} N{row['nets']} {row['blocks']}"


def accuracies(results):
    """(dataset, row id, shots, readout) -> list of val accuracies over replicates."""
    acc = defaultdict(list)
    for r in results:
        for readout in READOUTS:
            if readout in r:
                acc[r["dataset"], row_id(r["row"]), r["shots"], readout].append(r[readout]["val_acc"])
    return acc


def cell(values):
    if not values:
        return "–"
    return f"{100 * np.mean(values):.1f} ± {100 * np.std(values):.1f}" if len(values) > 1 else f"{100 * values[0]:.1f}"


def table(acc, title, rows, readout, label=lambda r: r):
    head = ["| " + title + " | " + " | ".join(f"{DATASET_NAMES[d]} {s}" for d in DATASETS for s in SHOTS) + " |"]
    head.append("|---" * (1 + len(DATASETS) * len(SHOTS)) + "|")
    lines = [f"| {label(r)} | " + " | ".join(cell(acc[d, r, s, readout]) for d in DATASETS for s in SHOTS) + " |" for r in rows]
    return "\n".join(head + lines)


def readout_table(acc, rows):
    head = ["| row | readout | " + " | ".join(f"{DATASET_NAMES[d]} {s}" for d in DATASETS for s in SHOTS) + " |"]
    head.append("|---" * (2 + len(DATASETS) * len(SHOTS)) + "|")
    lines = [f"| {r} | {k} | " + " | ".join(cell(acc[d, r, s, k]) for d in DATASETS for s in SHOTS) + " |"
             for r in rows for k in READOUTS]
    return "\n".join(head + lines)


def participation_ratio(x):
    """Effective dimensions of the standardised features (what the probes see), from the correlation matrix."""
    x = x[:, x.std(0) > 1e-6]
    eig = np.clip(np.linalg.eigvalsh(np.corrcoef(x, rowvar=False)), 0, None)
    return eig.sum() ** 2 / (eig ** 2).sum()


def depth_diagnostic(dataset="nette", seeds=range(10), blocks=(0, 1, 2, 3, 5, 9, 19, 39), arch="plain"):
    lines = ["| block | mean cosine between images | relative spread (median std / mean) | effective dims of 128, standardised (participation ratio) |",
             "|---|---|---|---|"]
    feats = [np.load(random_path(dataset, 64, s, arch), mmap_mode="r")[:2000].astype(np.float32) for s in seeds]
    for b in blocks:
        cos, spread, dims = [], [], []
        for f in feats:
            x = f[:, b].reshape(len(f), -1)
            n = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9)
            cos.append((n[:500] @ n[:500].T).mean())
            spread.append(np.median(x.std(0) / np.abs(x.mean(0)).clip(1e-9)))
            dims.append(participation_ratio(x))
        lines.append(f"| {b + 1} | {np.mean(cos):.4f} | {np.mean(spread):.3f} | {np.mean(dims):.1f} |")
    return "\n".join(lines)


def main():
    acc = accuracies(load_results())
    nets = (1, 2, 5, 10, 20, 40)
    depths = (1, 2, 3, 4, 6, 10, 20, 40)
    refs = ("pixels", "dinov2_s", "dinov2_b", "dinov2_l", "rad_dino", "resnet50")
    sections = []
    for readout in READOUTS:
        sections.append(f"## Readout: {readout}\n")
        for depth in (6, 20):
            sections.append(table(acc, f"nets (w64, depth {depth}, all blocks)", [f"w64 d{depth} N{n} all" for n in nets], readout,
                                  lambda r: r.split()[2]))
        for blocks in ("all", "last"):
            sections.append(table(acc, f"depth (w64, N10, {blocks} blocks)", [f"w64 d{d} N10 {blocks}" for d in depths], readout,
                                  lambda r: r.split()[1]))
        sections.append(table(acc, "depth, residual (w64, N10)", [f"{a}w64 d{d} N10 {b}" for b in ("all", "last") for d in (6, 10, 20, 40)
                                                                   for a in ("", "res ") if d > 6 or not a], readout, lambda r: r.replace(" N10", "")))
        width_rows = [f"w{w} d{d} N{n} all" for d in (6, 20) for w, n in ((64, 10), (16, 10), (16, 40), (64, 40))]
        sections.append(table(acc, "width", width_rows, readout))
        sections.append(table(acc, "reference", list(refs), readout))
    sections.append("## All readouts side by side (selected rows)\n")
    sections.append(readout_table(acc, ["w64 d6 N10 all", "w64 d20 N10 all", "w64 d20 N40 all", "w64 d6 N40 all", "dinov2_b"]))
    sections.append("## Depth diagnostic (Imagenette, 10 nets, width 64; mean and max pooled per channel)\n")
    sections.append(depth_diagnostic())
    sections.append("Same, residual wiring (same weights):\n")
    sections.append(depth_diagnostic(arch="res"))
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
