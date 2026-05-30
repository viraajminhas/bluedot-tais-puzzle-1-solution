"""Phase 7e: clean summary chart of SAE comparison results.

Shows: standard linear SAE, whitened SAE, Gaussian-RBF SAE, learned quadratic
SAE all fail at top-1, while the hand-built Hotelling T^2 / LLR features hit
~95-97 % accuracy.

The chart drives home the constructive interpretability finding for the
writeup.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).parent / "figs"


def main():
    # Hard-coded results from phases 5b, 7b, 7c, 7d.
    results = {
        "MLP probe\n(h2, full)": 96.6,
        "QDA\n(class covs)": 96.7,
        "T² in cov(F=1)\nwhitened space": 95.0,
        "log-likelihood\nratio LLR": 96.6,
        "T² in cov(F=0)\nwhitened space": 79.4,
        "Linear SAE\ntop-256": 92.1,
        "Linear SAE\ntop-64": 64.1,
        "Quadratic SAE\ntop-256 (learned)": 93.4,
        "Whitened linear\nSAE top-256": 90.0,
        "Linear SAE\ntop-1": 50.3,
        "Whitened linear\nSAE top-1": 51.3,
        "Gaussian RBF\nSAE top-1": 50.8,
        "Quadratic SAE\ntop-1 (learned)": 51.7,
        "Linear probe\non h2": 46.9,
    }

    # Group / colour by family
    family_color = {
        "ceiling":  "#1f77b4",
        "handbuilt_quadratic": "#2ca02c",
        "sae_full_dict": "#ff7f0e",
        "sae_top1": "#d62728",
        "baseline": "#7f7f7f",
    }
    family = {
        "MLP probe\n(h2, full)": "ceiling",
        "QDA\n(class covs)": "ceiling",
        "T² in cov(F=1)\nwhitened space": "handbuilt_quadratic",
        "log-likelihood\nratio LLR": "handbuilt_quadratic",
        "T² in cov(F=0)\nwhitened space": "handbuilt_quadratic",
        "Linear SAE\ntop-256": "sae_full_dict",
        "Linear SAE\ntop-64": "sae_full_dict",
        "Quadratic SAE\ntop-256 (learned)": "sae_full_dict",
        "Whitened linear\nSAE top-256": "sae_full_dict",
        "Linear SAE\ntop-1": "sae_top1",
        "Whitened linear\nSAE top-1": "sae_top1",
        "Gaussian RBF\nSAE top-1": "sae_top1",
        "Quadratic SAE\ntop-1 (learned)": "sae_top1",
        "Linear probe\non h2": "baseline",
    }

    # Sort by accuracy descending
    items = sorted(results.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colors = [family_color[family[k]] for k in labels]

    fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="black", lw=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("test accuracy on country (%)")
    ax.set_title(
        "Country decoding from h2: a single class-aware quadratic feature\n"
        "matches the MLP ceiling; every learned SAE top-1 sits at chance"
    )
    ax.axhline(50, color="grey", lw=0.5, linestyle="--")
    ax.set_ylim(40, 100)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}",
                ha="center", fontsize=8)
    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in family_color.values()]
    labels_legend = ["ceilings (MLP, QDA)",
                     "hand-built quadratic (class-aware)",
                     "SAE with full dictionary",
                     "SAE top-1 most country-selective feature",
                     "linear probe on raw h2"]
    ax.legend(handles, labels_legend, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sae_comparison.png", dpi=140)
    plt.close(fig)
    print(f"saved -> {FIG_DIR / 'sae_comparison.png'}")


if __name__ == "__main__":
    main()
