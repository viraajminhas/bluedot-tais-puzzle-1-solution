"""Phase 2c: confirm the mechanism the model uses to read country from h2.

Hypothesis: country=1 examples lie on a low-rank (≈1D) manifold passing through
the centre of the F=0 cloud (μ_+ = μ_-). The next layer (h2 → h3) restores
linearity by computing |perpendicular distance to the manifold| via ReLU(+x) +
ReLU(-x) on direction pairs. Predictions to check:

  P1. QDA (class-specific covariance) recovers country at high accuracy.
  P2. Among next-layer weight rows, country-relevant directions come in
      antipodal pairs (rows w and -w both fire for opposite sides of zero).
  P3. The h3 dimensions used by the country logit "light up" precisely
      when h2's projection onto a country manifold direction is far from 0.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from common import FEATURE_NAMES, get_activations, load_head

FIG_DIR = Path(__file__).parent / "figs"
COUNTRY_IDX = FEATURE_NAMES.index("country")


def main():
    train = get_activations("train")
    test = get_activations("test")
    X_tr, X_te = train["h2"], test["h2"]
    y_tr = train["labels"][:, COUNTRY_IDX]
    y_te = test["labels"][:, COUNTRY_IDX]

    # ---- P1: Quadratic Discriminant Analysis ------------------------------ #
    print("[P1] QDA (class-specific covariance)")
    qda = QuadraticDiscriminantAnalysis(reg_param=1e-3).fit(X_tr, y_tr)
    acc = qda.score(X_te, y_te)
    print(f"   QDA test acc: {acc:.4f}")
    # Also: linear-on-quadratic-features (||x - μ||^2 per class)
    mu_p = X_tr[y_tr == 1].mean(0)
    mu_n = X_tr[y_tr == 0].mean(0)
    d1_tr = ((X_tr - mu_p) ** 2).sum(1)
    d0_tr = ((X_tr - mu_n) ** 2).sum(1)
    d1_te = ((X_te - mu_p) ** 2).sum(1)
    d0_te = ((X_te - mu_n) ** 2).sum(1)
    feat_tr = np.stack([d1_tr, d0_tr, d1_tr - d0_tr], axis=1)
    feat_te = np.stack([d1_te, d0_te, d1_te - d0_te], axis=1)
    clf = LogisticRegression(max_iter=2000).fit(feat_tr, y_tr)
    print(f"   linear on (||x-mu+||^2, ||x-mu-||^2, diff) test acc: {clf.score(feat_te, y_te):.4f}")

    # ---- P2: Inspect next layer's weights -------------------------------- #
    print("\n[P2] Next-layer (h2 → pre-h3) weights")
    m = load_head("cpu")
    # layers idx 6 is the 4th Linear: 64 → 64 (h2 post-ReLU → pre-h3)
    W4: torch.Tensor = m.layers[6].weight.detach().numpy()  # (64, 64)
    b4 = m.layers[6].bias.detach().numpy()                  # (64,)
    print(f"   W4 shape={W4.shape}  ||W4||_F = {np.linalg.norm(W4):.3f}")

    # PC1 of F=1 covariance — the manifold direction
    Xp_centered = X_tr[y_tr == 1] - X_tr[y_tr == 1].mean(0)
    cov_p = np.cov(Xp_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov_p)
    v_manifold = eigvecs[:, -1]  # principal direction of F=1 covariance
    print(f"   manifold direction = top-eigvec of cov(F=1)  (eigval={eigvals[-1]:.4f})")

    # For each row of W4, compute its alignment with v_manifold.
    align = W4 @ v_manifold  # (64,)
    print(f"   distribution of W4[i] · v_manifold: "
          f"min={align.min():.3f}  max={align.max():.3f}  mean={align.mean():.4f}  std={align.std():.3f}")
    # Antipodal-pair test: does the model have units i,j such that W4[i] ≈ -W4[j]?
    # Compute pairwise cosine, find pairs with cos < -0.8
    W4n = W4 / (np.linalg.norm(W4, axis=1, keepdims=True) + 1e-9)
    cos = W4n @ W4n.T  # (64, 64)
    # Mask self-pairs and lower triangle
    cos_lower = np.tril(cos, k=-1)
    antipodal = np.argwhere(cos_lower < -0.8)
    print(f"   antipodal pairs (cos < -0.8) in W4: {len(antipodal)}")
    if len(antipodal):
        print(f"     example pairs: {antipodal[:5].tolist()}")
        # For each pair, show alignment with v_manifold
        for (i, j) in antipodal[:5]:
            print(f"     ({i:2d},{j:2d}) cos={cos[i,j]:+.3f}  "
                  f"W4[{i}]·v_mf={align[i]:+.3f}  W4[{j}]·v_mf={align[j]:+.3f}")

    # ---- P3: Visualise the 1D manifold cleanly --------------------------- #
    print("\n[P3] Manifold visualisation")
    # Project all h2 acts onto v_manifold (the country axis) and onto a
    # perpendicular direction (use PC1 of F=0 instead for contrast).
    Xn_centered = X_tr[y_tr == 0] - X_tr[y_tr == 0].mean(0)
    cov_n = np.cov(Xn_centered.T)
    eigvals_n, eigvecs_n = np.linalg.eigh(cov_n)
    v_perp = eigvecs_n[:, -1]
    # Make v_perp orthogonal to v_manifold
    v_perp = v_perp - (v_perp @ v_manifold) * v_manifold
    v_perp /= np.linalg.norm(v_perp) + 1e-9

    X_mean = X_tr.mean(0)
    proj_par = (X_tr - X_mean) @ v_manifold
    proj_per = (X_tr - X_mean) @ v_perp

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].scatter(proj_par[y_tr == 0], proj_per[y_tr == 0],
                   s=3, alpha=0.25, c="lightgrey", label="country=0")
    axs[0].scatter(proj_par[y_tr == 1], proj_per[y_tr == 1],
                   s=4, alpha=0.6, c="C3", label="country=1")
    axs[0].set_xlabel("projection on v_manifold (PC1 of cov(F=1))")
    axs[0].set_ylabel("perp direction (PC1 of cov(F=0))")
    axs[0].set_title("h2 in (manifold, perpendicular) plane")
    axs[0].legend()
    axs[0].axhline(0, color="k", lw=0.4, alpha=0.4)
    axs[0].axvline(0, color="k", lw=0.4, alpha=0.4)

    # Right: histogram of |perp distance| per class
    perp_dist = np.linalg.norm((X_tr - X_mean) - np.outer(proj_par, v_manifold), axis=1)
    axs[1].hist(perp_dist[y_tr == 0], bins=40, alpha=0.55, color="lightgrey",
                label=f"country=0  μ={perp_dist[y_tr==0].mean():.2f}")
    axs[1].hist(perp_dist[y_tr == 1], bins=40, alpha=0.75, color="C3",
                label=f"country=1  μ={perp_dist[y_tr==1].mean():.2f}")
    axs[1].set_xlabel("||h2 − projection on v_manifold|| (perpendicular distance)")
    axs[1].set_title("Distance from country manifold per class")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "manifold_mechanism.png", dpi=140)
    plt.close(fig)
    print(f"   saved -> {FIG_DIR / 'manifold_mechanism.png'}")
    print(f"   mean perp-distance from manifold: F=1={perp_dist[y_tr==1].mean():.3f}  F=0={perp_dist[y_tr==0].mean():.3f}")

    # ---- P3b: Test perp-distance as a 1-feature linear probe ------------- #
    proj_par_te = (X_te - X_mean) @ v_manifold
    perp_dist_te = np.linalg.norm((X_te - X_mean) - np.outer(proj_par_te, v_manifold), axis=1)
    clf2 = LogisticRegression(max_iter=2000).fit(perp_dist.reshape(-1, 1), y_tr)
    acc2 = clf2.score(perp_dist_te.reshape(-1, 1), y_te)
    print(f"   1-feature linear probe on perp-distance: test acc {acc2:.4f}")


if __name__ == "__main__":
    main()
