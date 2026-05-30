"""Phase 5g: per-axis causal ablation.

Hypothesis: the 4 axes where F=0 has 5-11x more spread than F=1 (PCs 4, 6, 7, 9
of F=1's covariance) are the "country circuit." Ablating those specific axes
should crater country accuracy while leaving other features intact. Ablating
other axes should hurt little.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import FEATURE_NAMES, get_activations, load_head

FIG_DIR = Path(__file__).parent / "figs"
COUNTRY_IDX = FEATURE_NAMES.index("country")


def main():
    train = get_activations("train")
    test = get_activations("test")
    X_tr, X_te = train["h2"], test["h2"]
    y_te = test["labels"]

    # Compute F=1 PCA on training
    y_tr = train["labels"][:, COUNTRY_IDX]
    Xp = X_tr[y_tr == 1]
    Xn = X_tr[y_tr == 0]
    mu_p = Xp.mean(0)
    mu_n = Xn.mean(0)
    cov_p = np.cov((Xp - mu_p).T)
    cov_n = np.cov((Xn - mu_n).T)
    eigvals, V = np.linalg.eigh(cov_p)
    eigvals, V = eigvals[::-1], V[:, ::-1]
    eigvals_n = np.linalg.eigvalsh(cov_n)[::-1]

    # For each PC k of F=1, compute the eigenvalue ratio (F=0 spread / F=1 spread)
    # along that direction.
    ratios = []
    for k in range(20):
        v = V[:, k]
        sp = np.var(Xp @ v)
        sn = np.var(Xn @ v)
        ratios.append(sn / (sp + 1e-9))

    # Load puzzle head
    m = load_head("cpu")

    mu = X_tr.mean(0)

    def replay(X):
        with torch.no_grad():
            xt = torch.from_numpy(X.astype(np.float32))
            logits = m.layers[6:](xt).numpy()
        preds = (logits > 0).astype(np.int64)
        accs = (preds == y_te).mean(0)
        return accs

    base_acc = replay(X_te)
    print("Baseline test acc per feature:")
    for i, n in enumerate(FEATURE_NAMES):
        print(f"  {n:<10}: {base_acc[i]:.4f}")

    print(f"\nPer-axis ablation of F=1's PCs (zero out projection on each axis k):")
    print(f"{'PC':<4} {'eigratio':<10} " + " ".join(f"{n[:7]:>8}" for n in FEATURE_NAMES))
    for k in range(15):
        v = V[:, k]
        # Ablate: x' = x - (x · v) v  (project out component along v)
        Xc = X_te - mu
        Xc_abl = Xc - np.outer(Xc @ v, v)
        X_te_abl = mu + Xc_abl
        accs = replay(X_te_abl)
        deltas = accs - base_acc
        # Highlight country delta
        ratio_str = f"{ratios[k]:.2f}x"
        delta_str = " ".join(f"{d:+8.4f}" for d in deltas)
        flag = ""
        if abs(deltas[COUNTRY_IDX]) > 0.05:
            flag = " <-- big country hit"
        print(f"{k:<4} {ratio_str:<10} {delta_str}{flag}")

    # Now ablate the SET of high-ratio axes vs same number of random axes
    high_ratio_axes = sorted(range(15), key=lambda k: -ratios[k])[:4]
    print(f"\nHigh-ratio (F=0/F=1 > 5x) axes among top 15: PCs {high_ratio_axes} "
          f"(ratios {[f'{ratios[k]:.1f}x' for k in high_ratio_axes]})")

    # Ablate all 4 high-ratio axes at once
    V_high = V[:, high_ratio_axes]
    Xc = X_te - mu
    Xc_abl = Xc - (Xc @ V_high) @ V_high.T
    X_te_abl = mu + Xc_abl
    accs = replay(X_te_abl)
    print(f"\nAblating the 4 high-ratio axes simultaneously:")
    for i, n in enumerate(FEATURE_NAMES):
        delta = accs[i] - base_acc[i]
        print(f"  {n:<10}: {accs[i]:.4f}  (delta {delta:+.4f})")

    # Compare with ablating 4 random axes (or low-ratio axes)
    low_ratio_axes = sorted(range(15), key=lambda k: ratios[k])[:4]
    V_low = V[:, low_ratio_axes]
    Xc_abl = Xc - (Xc @ V_low) @ V_low.T
    X_te_abl = mu + Xc_abl
    accs2 = replay(X_te_abl)
    print(f"\nFor contrast, ablating the 4 low-ratio (≈1x) axes: PCs {low_ratio_axes}")
    for i, n in enumerate(FEATURE_NAMES):
        delta = accs2[i] - base_acc[i]
        print(f"  {n:<10}: {accs2[i]:.4f}  (delta {delta:+.4f})")

    # Bar plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    K = 15
    width = 0.42
    feat_to_plot = ["country", "food", "sentiment", "question", "number"]
    feat_ids = [FEATURE_NAMES.index(n) for n in feat_to_plot]
    deltas_per_axis = np.zeros((K, len(feat_to_plot)))
    for k in range(K):
        v = V[:, k]
        Xc = X_te - mu
        Xc_abl = Xc - np.outer(Xc @ v, v)
        accs_k = replay(mu + Xc_abl)
        for j, fi in enumerate(feat_ids):
            deltas_per_axis[k, j] = accs_k[fi] - base_acc[fi]
    colors = ["C3", "#888", "#777", "#666", "#bbb"]
    bottom = np.zeros(K)
    for j, fi in enumerate(feat_ids):
        ax.bar(np.arange(K) + (j - 2) * 0.15, -deltas_per_axis[:, j],
               width=0.15, color=colors[j], label=feat_to_plot[j])
    ax.set_xlabel("PC index of cov(country=1)")
    ax.set_ylabel("Accuracy drop when this PC is ablated")
    ax.set_title("Per-axis ablation: which directions are the country circuit?")
    ax.legend()
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(K))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "axis_ablation.png", dpi=140)
    plt.close(fig)
    print(f"\nsaved -> {FIG_DIR / 'axis_ablation.png'}")


if __name__ == "__main__":
    main()
