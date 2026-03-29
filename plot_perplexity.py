#!/usr/bin/env python3
"""
Generate perplexity comparison charts for ruGPT-3 model family.

Compares perplexity values measured on IlyaGusev/gazeta test set
against the original values reported by ai-forever team.

Usage:
    python plot_perplexity.py
    python plot_perplexity.py --output_dir ./charts
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


MODELS = {
    "ruGPT3-small": {"params_m": 125, "gazeta_ppl": 20.12, "original_ppl": None},
    "ruGPT3-medium": {"params_m": 355, "gazeta_ppl": 15.49, "original_ppl": 17.4},
    "ruGPT3-large": {"params_m": 760, "gazeta_ppl": 14.03, "original_ppl": 13.6},
    "ruGPT3XL\n(dense)": {"params_m": 1300, "gazeta_ppl": 50.11, "original_ppl": None},
    "ruGPT3XL\n(sparse)": {"params_m": 1300, "gazeta_ppl": 11.68, "original_ppl": 12.05},
}

CORRELATION_MODELS = ["ruGPT3-medium", "ruGPT3-large", "ruGPT3XL\n(sparse)"]


def plot_bar_comparison(output_path):
    """Bar chart comparing Gazeta PPL across all models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(MODELS.keys())
    gazeta = [MODELS[n]["gazeta_ppl"] for n in names]
    original = [MODELS[n]["original_ppl"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars_gazeta = ax.bar(
        x - width / 2, gazeta, width,
        label="Gazeta test set (measured)",
        color="#4C72B0", edgecolor="white", linewidth=0.8,
    )
    bars_original = ax.bar(
        x + width / 2,
        [v if v is not None else 0 for v in original],
        width,
        label="Original reported (ai-forever)",
        color="#DD8452", edgecolor="white", linewidth=0.8,
    )

    for i, v in enumerate(original):
        if v is None:
            bars_original[i].set_alpha(0)

    for bar, val in zip(bars_gazeta, gazeta):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    for bar, val in zip(bars_original, original):
        if val is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.set_ylabel("Perplexity (lower is better)", fontsize=12)
    ax.set_title("ruGPT-3 Model Family: Perplexity Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, max(gazeta) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_correlation(output_path):
    """Scatter plot: original PPL vs Gazeta PPL with linear fit."""
    orig_vals = []
    gaz_vals = []
    labels = []
    for name in CORRELATION_MODELS:
        m = MODELS[name]
        orig_vals.append(m["original_ppl"])
        gaz_vals.append(m["gazeta_ppl"])
        labels.append(name.replace("\n", " "))

    orig = np.array(orig_vals)
    gaz = np.array(gaz_vals)

    coeffs = np.polyfit(orig, gaz, 1)
    poly = np.poly1d(coeffs)
    r = np.corrcoef(orig, gaz)[0, 1]

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(orig, gaz, s=120, color="#4C72B0", zorder=5, edgecolors="white", linewidth=1.5)

    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (orig[i], gaz[i]),
            textcoords="offset points",
            xytext=(10, -5),
            fontsize=10,
        )

    fit_x = np.linspace(min(orig) - 1, max(orig) + 1, 100)
    ax.plot(fit_x, poly(fit_x), "--", color="#DD8452", linewidth=1.5, alpha=0.8)

    diag_min = min(min(orig), min(gaz)) - 1
    diag_max = max(max(orig), max(gaz)) + 1
    ax.plot(
        [diag_min, diag_max], [diag_min, diag_max],
        ":", color="gray", alpha=0.5, label="y = x (perfect match)",
    )

    ax.set_xlabel("Original reported PPL (ai-forever)", fontsize=12)
    ax.set_ylabel("Measured PPL (Gazeta test set)", fontsize=12)
    ax.set_title(
        f"Correlation: Original vs Gazeta PPL\n"
        f"R = {r:.4f}, y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_scaling(output_path):
    """PPL vs model size (parameters) on log scale."""
    fig, ax = plt.subplots(figsize=(9, 6))

    names_normal = [n for n in MODELS if "dense" not in n]
    params = [MODELS[n]["params_m"] for n in names_normal]
    gaz = [MODELS[n]["gazeta_ppl"] for n in names_normal]
    orig = [MODELS[n]["original_ppl"] for n in names_normal]

    ax.plot(params, gaz, "o-", color="#4C72B0", markersize=8, linewidth=2,
            label="Gazeta test set (measured)", zorder=5)

    orig_params = [p for p, o in zip(params, orig) if o is not None]
    orig_vals = [o for o in orig if o is not None]
    ax.plot(orig_params, orig_vals, "s--", color="#DD8452", markersize=8, linewidth=2,
            label="Original reported (ai-forever)", zorder=4)

    dense_ppl = MODELS["ruGPT3XL\n(dense)"]["gazeta_ppl"]
    ax.plot(1300, dense_ppl, "x", color="#C44E52", markersize=12, markeredgewidth=3,
            label=f"XL without sparse attention (PPL={dense_ppl:.1f})", zorder=6)

    for n in names_normal:
        m = MODELS[n]
        label = n.replace("\n", " ")
        ax.annotate(
            f"{label}\n({m['gazeta_ppl']:.2f})",
            (m["params_m"], m["gazeta_ppl"]),
            textcoords="offset points",
            xytext=(12, 5),
            fontsize=9,
        )

    ax.annotate(
        f"XL dense\n({dense_ppl:.1f})",
        (1300, dense_ppl),
        textcoords="offset points",
        xytext=(12, -5),
        fontsize=9,
        color="#C44E52",
    )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    ax.set_xlabel("Model Parameters", fontsize=12)
    ax.set_ylabel("Perplexity (lower is better)", fontsize=12)
    ax.set_title(
        "ruGPT-3 Scaling: Perplexity vs Model Size",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate PPL comparison charts")
    parser.add_argument(
        "--output_dir", type=str, default=".",
        help="Directory to save chart images",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plot_bar_comparison(os.path.join(args.output_dir, "ppl_comparison.png"))
    plot_correlation(os.path.join(args.output_dir, "ppl_correlation.png"))
    plot_scaling(os.path.join(args.output_dir, "ppl_scaling.png"))

    print("\nAll charts generated successfully.")


if __name__ == "__main__":
    main()
