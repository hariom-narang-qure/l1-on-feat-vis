"""Plot the flattened, descending-sorted values of a saved activation map.

Two panels share the rank axis but not a y-scale: linear shows the raw shape
(how quickly it falls off the peak), log shows the spread across the tail.
Exact zeros (from the ReLU) cannot appear on a log axis, so the log panel
plots only positive values and the zero count is stated in the title.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES_COLOR = "#2a78d6"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"


def sorted_descending(activation_map):
    return np.sort(activation_map.ravel())[::-1]


def style_axis(ax):
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.set_xlabel("rank (spatial cells, sorted high → low)", color=TEXT_SECONDARY)


def plot(values, title, out_path):
    ranks = np.arange(1, len(values) + 1)
    positive = values > 0
    zero_count = int((~positive).sum())

    fig, (linear_ax, log_ax) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    fig.suptitle(title, fontsize=13)

    linear_ax.plot(ranks, values, color=SERIES_COLOR, linewidth=2)
    linear_ax.set_title("linear scale", fontsize=11)
    linear_ax.set_ylabel("activation", color=TEXT_SECONDARY)
    style_axis(linear_ax)

    log_ax.plot(ranks[positive], values[positive], color=SERIES_COLOR, linewidth=2)
    log_ax.set_yscale("log")
    log_ax.set_title(f"log scale ({zero_count} of {len(values)} cells are exactly 0, not shown)", fontsize=11)
    style_axis(log_ax)

    fig.tight_layout()
    fig.savefig(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("activation_map", type=Path, help=".npy saved by activation_map.py")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    values = sorted_descending(np.load(args.activation_map))
    out_path = args.out or args.activation_map.with_name(f"{args.activation_map.stem}_sorted.png")
    plot(values, f"{args.activation_map.stem}: sorted activations", out_path)

    top_share = values[:10].sum() / values.sum()
    print(
        f"cells={len(values)}  max={values[0]:.2f}  median={np.median(values):.2f}  "
        f"zeros={(values == 0).mean():.1%}  top-10 cells hold {top_share:.1%} of total"
    )
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
