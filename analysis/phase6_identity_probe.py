"""Phase 6: is PC4 x PC6 of cov(F=1) the COUNTRY-IDENTITY plane?

Test: train a country-name multi-class classifier on F=1 examples, restricted
to various h2 subspaces. If accuracy is high on a small subspace, country
identity (which country is mentioned) lives in that subspace.

If the country-circuit axes (PC4, PC6) carry country identity, the geometric
story becomes:
  - Each country has a coordinate in (PC4, PC6) (and a few more).
  - country=1 -> example sits at the country-specific point (low spread).
  - country=0 -> example wanders the broad cloud.
  - Mean over all countries ~ 0, so means coincide -> linear probe fails.
  - "is country" = "is x close to one of the country prototypes" -> nonlinear.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from common import FEATURE_NAMES, get_activations, load_split
from phase2b_country import COUNTRY_LIST, tag_countries

FIG_DIR = Path(__file__).parent / "figs"
COUNTRY_IDX = FEATURE_NAMES.index("country")


def main():
    train = get_activations("train")
    test = get_activations("test")
    texts_tr, _, _ = load_split("train")
    texts_te, _, _ = load_split("test")

    X_tr = train["h2"]
    X_te = test["h2"]
    y_tr = train["labels"][:, COUNTRY_IDX]
    y_te = test["labels"][:, COUNTRY_IDX]

    tags_tr = tag_countries(texts_tr)
    tags_te = tag_countries(texts_te)

    # Restrict to F=1 examples with identified country
    f1_tr = (y_tr == 1) & np.array([t is not None for t in tags_tr])
    f1_te = (y_te == 1) & np.array([t is not None for t in tags_te])
    X_id_tr = X_tr[f1_tr]
    X_id_te = X_te[f1_te]
    countries_tr = [tags_tr[i] for i in np.where(f1_tr)[0]]
    countries_te = [tags_te[i] for i in np.where(f1_te)[0]]

    # Keep only countries with enough train examples
    train_counts = Counter(countries_tr)
    kept = {c for c, n in train_counts.items() if n >= 10}
    mask_tr = np.array([c in kept for c in countries_tr])
    mask_te = np.array([c in kept for c in countries_te])
    X_id_tr = X_id_tr[mask_tr]
    X_id_te = X_id_te[mask_te]
    countries_tr = [c for c in countries_tr if c in kept]
    countries_te = [c for c in countries_te if c in kept]
    countries_sorted = sorted(kept)
    c2i = {c: i for i, c in enumerate(countries_sorted)}
    y_id_tr = np.array([c2i[c] for c in countries_tr])
    y_id_te = np.array([c2i[c] for c in countries_te])
    print(f"Number of distinct countries (>=10 train examples): {len(kept)}")
    print(f"  train identity-probe set: {len(y_id_tr)}")
    print(f"  test identity-probe set:  {len(y_id_te)}")

    # F=1 PCA on h2 (train all F=1)
    Xp = X_tr[y_tr == 1] - X_tr[y_tr == 1].mean(0)
    cov_p = np.cov(Xp.T)
    eigvals, V = np.linalg.eigh(cov_p)
    eigvals, V = eigvals[::-1], V[:, ::-1]

    mu = X_tr.mean(0)

    def project(X, axes):
        return (X - mu) @ V[:, axes]

    print("\nCountry-identity classification accuracy (multi-class LR):")
    print(f"  baseline (most-common class): {max(train_counts.values()) / sum(train_counts.values()):.4f}")
    sets = [
        ("PC0", [0]),
        ("PC1", [1]),
        ("PC0+PC1", [0, 1]),
        ("PC2+PC3", [2, 3]),
        ("PC4 alone", [4]),
        ("PC6 alone", [6]),
        ("PC4+PC6", [4, 6]),
        ("PC0..PC3", [0, 1, 2, 3]),
        ("PC4..PC7", [4, 5, 6, 7]),
        ("PCs {4,6} (country circuit only)", [4, 6]),
        ("PCs {0,1,2,3} (non-country circuit)", [0, 1, 2, 3]),
        ("top 8 PCs", list(range(8))),
        ("top 16 PCs", list(range(16))),
        ("ALL 64 dims (h2)", "all"),
    ]
    for name, axes in sets:
        if axes == "all":
            Xtr, Xte = X_tr[y_tr == 1][mask_tr.sum() and slice(None) or slice(None)], X_te[y_te == 1]
            # Simpler: re-mask
            Xtr = X_tr[np.where(f1_tr)[0]][mask_tr]
            Xte = X_te[np.where(f1_te)[0]][mask_te]
        else:
            Xtr = project(X_tr[np.where(f1_tr)[0]][mask_tr], axes)
            Xte = project(X_te[np.where(f1_te)[0]][mask_te], axes)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000,
                                  solver="lbfgs").fit(sc.transform(Xtr), y_id_tr)
        acc = clf.score(sc.transform(Xte), y_id_te)
        # Also kNN for non-linear baseline
        knn = KNeighborsClassifier(n_neighbors=5).fit(Xtr, y_id_tr)
        kacc = knn.score(Xte, y_id_te)
        print(f"  {name:<40}  LR {acc:.4f}  kNN-5 {kacc:.4f}  "
              f"(dims={Xtr.shape[1]})")

    # Visualise per-country centroids in PC4 x PC6
    print("\nVisualising PC4 x PC6 ...")
    proj_tr = project(X_tr[np.where(f1_tr)[0]][mask_tr], [4, 6])
    # Plot per-country centroids
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    ax = axs[0]
    centroids = []
    cents_dict = {}
    for c in countries_sorted:
        idx = [i for i, cc in enumerate(countries_tr) if cc == c]
        if len(idx) < 5:
            continue
        cx = proj_tr[idx, 0].mean()
        cy = proj_tr[idx, 1].mean()
        centroids.append((cx, cy, c, len(idx)))
        cents_dict[c] = (cx, cy)
    # Background: country=0 examples on the same plane
    Xn = X_tr[y_tr == 0]
    proj_n = (Xn - mu) @ V[:, [4, 6]]
    ax.scatter(proj_n[:, 0], proj_n[:, 1], s=3, alpha=0.10, c="lightgrey",
               label="country=0 (background)")
    ax.scatter(proj_tr[:, 0], proj_tr[:, 1], s=4, alpha=0.35, c="#777",
               label="country=1 individual examples")
    for cx, cy, c, n in centroids:
        ax.scatter([cx], [cy], s=40, c="C3")
    # Label the most extreme ones
    centroids.sort(key=lambda t: -(t[0] ** 2 + t[1] ** 2))
    for cx, cy, c, _ in centroids[:30]:
        ax.annotate(c, (cx, cy), fontsize=7, alpha=0.85)
    ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("PC4 of cov(country=1)")
    ax.set_ylabel("PC6 of cov(country=1)")
    ax.set_title("h2 projected on (PC4, PC6) — the country-circuit plane")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    # Right: per-country centroids only, coloured by continent
    from phase2b_country import CONTINENT
    ax = axs[1]
    cont_colors = {"Africa": "#e15759", "Asia": "#4e79a7", "Europe": "#59a14f",
                   "Americas": "#f28e2b", "Oceania": "#76b7b2", "Unknown": "#bab0ac"}
    for cx, cy, c, n in centroids:
        cont = CONTINENT.get(c, "Unknown")
        ax.scatter([cx], [cy], s=40, c=cont_colors[cont], alpha=0.8)
    for cont, color in cont_colors.items():
        ax.scatter([], [], s=40, c=color, label=cont)
    for cx, cy, c, _ in centroids[:40]:
        ax.annotate(c, (cx, cy), fontsize=7, alpha=0.85)
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("PC4 of cov(country=1)")
    ax.set_ylabel("PC6 of cov(country=1)")
    ax.set_title("Per-country centroids in (PC4, PC6), coloured by continent")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "country_identity_plane.png", dpi=140)
    plt.close(fig)
    print(f"  saved -> {FIG_DIR / 'country_identity_plane.png'}")


if __name__ == "__main__":
    main()
