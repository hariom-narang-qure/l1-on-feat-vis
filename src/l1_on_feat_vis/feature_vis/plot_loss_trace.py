"""Plot the trace written by match_spatial_map.py --trace-every.

Three panels sharing the step axis, each with its own y-scale (never a dual
axis): weighted loss values, the L2 norm of each term's gradient on the image,
and the cosine between the two gradients (-1 = pushing in opposite directions).
Losses and gradient norms span orders of magnitude, so those panels are log-y.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MSE_COLOR = "#2a78d6"
L1_COLOR = "#eb6834"
TV_COLOR = "#1baf7a"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"


def style_axis(ax, ylabel):
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.set_xlabel("step", color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)


def plot_terms(ax, steps, mse_values, l1_values, tv_values, title, ylabel):
    ax.plot(steps, mse_values, color=MSE_COLOR, linewidth=2, marker="o", markersize=4, label="MSE (map)")
    ax.plot(steps, l1_values, color=L1_COLOR, linewidth=2, marker="o", markersize=4, label="weighted L1")
    if tv_values and any(tv_values):
        ax.plot(steps, tv_values, color=TV_COLOR, linewidth=2, marker="o", markersize=4, label="weighted TV")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=11)
    style_axis(ax, ylabel)
    ax.legend(frameon=False, labelcolor=TEXT_SECONDARY)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text())
    steps = [row["step"] for row in trace]
    column = lambda key: [row[key] for row in trace]
    optional_column = lambda key: [row.get(key, 0.0) for row in trace]

    fig, (loss_ax, grad_ax, cos_ax) = plt.subplots(1, 3, figsize=(16, 4.5), dpi=150)
    fig.suptitle(args.trace.stem.replace("_trace", ""), fontsize=12)
    plot_terms(loss_ax, steps, column("mse"), column("weighted_l1"), optional_column("weighted_tv"),
               "weighted loss terms", "loss value")
    plot_terms(grad_ax, steps, column("mse_grad_norm"), column("l1_grad_norm"), optional_column("tv_grad_norm"),
               "gradient on the image (L2 norm)", "‖∂term/∂image‖")

    cos_ax.plot(steps, column("grad_cosine"), color=L1_COLOR, linewidth=2, marker="o", markersize=4,
                label="MSE vs L1")
    if any(optional_column("tv_grad_norm")):
        cos_ax.plot(steps, optional_column("grad_cosine_mse_tv"), color=TV_COLOR, linewidth=2, marker="o",
                    markersize=4, label="MSE vs TV")
        cos_ax.legend(frameon=False, labelcolor=TEXT_SECONDARY)
    cos_ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    cos_ax.set_ylim(-1, 1)
    cos_ax.set_title("cosine between gradients", fontsize=11)
    style_axis(cos_ax, "cosine similarity")

    fig.tight_layout()
    out_path = args.trace.with_name(f"{args.trace.stem}.png")
    fig.savefig(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
