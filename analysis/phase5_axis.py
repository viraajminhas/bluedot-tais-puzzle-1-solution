"""Phase 5: crack what the country manifold axis encodes.

Leading hypothesis: the model preserves per-country *direction* (one vector per
country), but cancels the class-mean offset. So countries live on a manifold
whose principal axis is the dominant direction in a country-by-direction PCA.

Candidate explanations for the axis tested here:
  H1. Per-country MiniLM embedding of bare country name           (already weak)
  H2. Per-country mean MiniLM embedding of *training texts* containing it
  H3. Per-country frequency in training data (templates pick countries non-unif.)
  H4. Per-country LDA direction at the MiniLM level (country vs not-country)
  H5. Per-country alphabet position / first letter (silly but checkable)
  H6. Linear map from h2-centroids back to MiniLM-text-centroids
       (i.e. h2 preserves the input-level country geometry)
"""
from __future__ import annotations

from pathlib import Path
import string

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from common import FEATURE_NAMES, get_activations, load_split
from phase2b_country import COUNTRY_LIST, tag_countries

FIG_DIR = Path(__file__).parent / "figs"
COUNTRY_IDX = FEATURE_NAMES.index("country")
MIN_COUNT = 10


def per_country_centroids(values: np.ndarray, tags: list[str | None]):
    by_country: dict[str, list[np.ndarray]] = {}
    for v, t in zip(values, tags):
        if t is None:
            continue
        by_country.setdefault(t, []).append(v)
    centroids = {c: np.stack(arr).mean(0) for c, arr in by_country.items()
                 if len(arr) >= MIN_COUNT}
    counts = {c: len(arr) for c, arr in by_country.items() if len(arr) >= MIN_COUNT}
    return centroids, counts


def main():
    # Load activations + tags
    train = get_activations("train")
    test = get_activations("test")
    texts_tr, _, _ = load_split("train")
    tags_tr = tag_countries(texts_tr)
    X_tr = train["h2"]                                # (N, 64)
    emb_tr = train["embeddings"]                      # (N, 384)
    y_tr = train["labels"][:, COUNTRY_IDX]

    h2_cent, counts = per_country_centroids(X_tr, tags_tr)
    countries = sorted(h2_cent.keys())
    print(f"countries with >={MIN_COUNT} examples: {len(countries)}")

    h2_C = np.stack([h2_cent[c] for c in countries])  # (K, 64) per-country h2 centroids
    h2_global_mean = X_tr.mean(0)
    h2_C_centred = h2_C - h2_global_mean

    # ---- Manifold direction (PC1 of cov(F=1)) ------------------------- #
    Xp = X_tr[y_tr == 1]
    cov_p = np.cov((Xp - Xp.mean(0)).T)
    eigvals, eigvecs = np.linalg.eigh(cov_p)
    v_mf = eigvecs[:, -1]
    # PC1 projection of each country's h2 centroid
    h2_pc1 = h2_C_centred @ v_mf                       # (K,)

    print("\n=== Hypothesis tests ===")

    # ---- H2: text-context MiniLM embedding ------------------------------ #
    print("\nH2: per-country mean MiniLM-embedding of training texts that contain it")
    emb_cent, _ = per_country_centroids(emb_tr, tags_tr)
    emb_C = np.stack([emb_cent[c] for c in countries])
    emb_global_mean = emb_tr.mean(0)
    emb_C_centred = emb_C - emb_global_mean
    pca_emb = PCA(n_components=6).fit(emb_C_centred)
    proj_emb = pca_emb.transform(emb_C_centred)
    for k in range(6):
        r = np.corrcoef(h2_pc1, proj_emb[:, k])[0, 1]
        print(f"  Pearson(h2_PC1, text-MiniLM-PC{k+1}) = {r:+.3f}  "
              f"(explained var {pca_emb.explained_variance_ratio_[k]:.3f})")

    # Linear regression: how much of h2_pc1 is explained by linear combo of all 6 text-PCs?
    lr = LinearRegression().fit(proj_emb, h2_pc1)
    r2 = lr.score(proj_emb, h2_pc1)
    print(f"  R^2(h2_PC1 | top-6 text-MiniLM PCs) = {r2:.3f}")

    # ---- H3: frequency ------------------------------------------------- #
    print("\nH3: per-country training-data frequency")
    cnt_vec = np.array([counts[c] for c in countries], dtype=np.float32)
    r = np.corrcoef(h2_pc1, cnt_vec)[0, 1]
    r_log = np.corrcoef(h2_pc1, np.log(cnt_vec))[0, 1]
    print(f"  Pearson(h2_PC1, count) = {r:+.3f}")
    print(f"  Pearson(h2_PC1, log count) = {r_log:+.3f}")

    # ---- H4: per-country LDA direction at MiniLM input level ----------- #
    print("\nH4: similarity between h2 per-country DIRECTIONS and MiniLM per-country DIRECTIONS")
    # Use the same per-country MiniLM centroids; the *direction* of a country is
    # emb_cent[c] - emb_global_mean.
    # Map each h2 country direction onto the MiniLM space via least-squares.
    # I.e., find W: 64->384 such that emb_C_centred ≈ h2_C_centred @ W^T
    # Then check residuals.
    W, residuals, rank, _ = np.linalg.lstsq(h2_C_centred, emb_C_centred, rcond=None)
    pred_emb = h2_C_centred @ W
    ss_res = ((emb_C_centred - pred_emb) ** 2).sum()
    ss_tot = ((emb_C_centred - emb_C_centred.mean(0)) ** 2).sum()
    R2 = 1 - ss_res / ss_tot
    print(f"  R^2(MiniLM per-country direction | linear map from h2 per-country direction) = {R2:.3f}")
    # Also: cosine similarity per country
    cos_per_country = []
    for i in range(len(countries)):
        h2v = h2_C_centred[i]
        emb_v = emb_C_centred[i]
        # Predict emb from h2 using global W
        pred = h2v @ W
        c = pred @ emb_v / (np.linalg.norm(pred) * np.linalg.norm(emb_v) + 1e-9)
        cos_per_country.append(c)
    print(f"  mean per-country cosine(predicted emb, true emb): {np.mean(cos_per_country):+.3f}")

    # ---- H5: alphabet position / first letter -------------------------- #
    print("\nH5: alphabet position / first letter")
    first = np.array([string.ascii_uppercase.index(c[0].upper())
                      if c[0].upper() in string.ascii_uppercase else 0
                      for c in countries])
    r = np.corrcoef(h2_pc1, first)[0, 1]
    print(f"  Pearson(h2_PC1, alphabet-position-of-first-letter) = {r:+.3f}")

    # ---- H6: shape similarity of per-country PCA --------------------- #
    print("\nH6: do the two PCA spaces (h2 vs text-MiniLM, per-country centroids) agree on PC1?")
    pca_h2 = PCA(n_components=6).fit(h2_C_centred)
    print(f"  cum variance explained by top-k PCs of h2 per-country centroids:")
    for k in range(6):
        cum = pca_h2.explained_variance_ratio_[:k+1].sum()
        print(f"    top-{k+1}: {cum:.3f}")
    print(f"  cum variance explained by top-k PCs of text-MiniLM per-country centroids:")
    for k in range(6):
        cum = pca_emb.explained_variance_ratio_[:k+1].sum()
        print(f"    top-{k+1}: {cum:.3f}")

    # Procrustes-style: subspace alignment between h2-PCs and text-MiniLM-PCs
    # via canonical correlation (simple version with PCA reduced to common dim).
    proj_h2 = pca_h2.transform(h2_C_centred)
    proj_em = pca_emb.transform(emb_C_centred)
    # Match each PC of h2 to its closest match in emb space
    for k in range(4):
        best = (0, 0.0)
        for j in range(4):
            r = abs(np.corrcoef(proj_h2[:, k], proj_em[:, j])[0, 1])
            if r > best[1]:
                best = (j, r)
        print(f"  h2-PC{k+1} best matches text-MiniLM-PC{best[0]+1} with |r|={best[1]:.3f}")

    # ---- Visualization: PC1 of h2 per-country vs PC1 of text-MiniLM per-country --- #
    print("\nGenerating axis_v2 figure...")
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    ax = axs[0]
    ax.scatter(proj_em[:, 0], h2_pc1, s=20, alpha=0.8)
    for i, c in enumerate(countries):
        if i % 6 == 0:
            ax.annotate(c, (proj_em[i, 0], h2_pc1[i]), fontsize=7, alpha=0.75)
    ax.set_xlabel("per-country text-MiniLM PC1")
    ax.set_ylabel("per-country h2 PC1 (= projection on v_manifold)")
    ax.set_title(
        f"Best-correlated MiniLM PC vs h2 PC1\n"
        f"(text-context MiniLM, R²={r2:.2f} from top-6 PCs)"
    )

    ax = axs[1]
    # h2 per-country centroids in 2-D (h2-PC1, h2-PC2)
    proj_h2_2 = pca_h2.transform(h2_C_centred)
    ax.scatter(proj_h2_2[:, 0], proj_h2_2[:, 1], s=20, alpha=0.7)
    for i, c in enumerate(countries):
        if i % 6 == 0:
            ax.annotate(c, (proj_h2_2[i, 0], proj_h2_2[i, 1]), fontsize=7, alpha=0.75)
    ax.set_xlabel("h2 per-country PC1")
    ax.set_ylabel("h2 per-country PC2")
    ax.set_title("Country centroids in h2's first 2 PCs")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "axis_v2_centroids.png", dpi=130)
    plt.close(fig)
    print(f"  saved -> {FIG_DIR / 'axis_v2_centroids.png'}")


if __name__ == "__main__":
    main()
