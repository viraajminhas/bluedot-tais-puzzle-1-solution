"""Phase 2d: causal interventions + axis interpretation.

Two questions:
  Q1. Does the country LOGIT causally depend on the perpendicular-to-manifold
      distance? Edit h2 toward/away from the manifold and check downstream.
  Q2. What does the country manifold's principal axis encode? Compare each
      country's centroid position on PC1 against per-country MiniLM
      embeddings.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import FEATURE_NAMES, get_activations, load_head, load_split
from phase2b_country import COUNTRY_LIST, tag_countries

FIG_DIR = Path(__file__).parent / "figs"
COUNTRY_IDX = FEATURE_NAMES.index("country")


def main():
    train = get_activations("train")
    test = get_activations("test")
    X_tr = train["h2"]
    X_te = test["h2"]
    y_tr = train["labels"][:, COUNTRY_IDX]
    y_te = test["labels"][:, COUNTRY_IDX]

    # Manifold direction and global mean
    mu = X_tr.mean(0)
    Xp_centered = X_tr[y_tr == 1] - X_tr[y_tr == 1].mean(0)
    cov_p = np.cov(Xp_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov_p)
    v_mf = eigvecs[:, -1]   # principal direction
    # Top-K manifold subspace
    K = 4
    V_mf = eigvecs[:, -K:]   # (64, K)

    # ----- Q1: Causal interventions ---------------------------------------- #
    print("[Q1] Causal interventions")
    m = load_head("cpu")

    def acts_to_logits(h2_vec):
        # h2 -> Linear -> ReLU -> Linear -> logits
        x = torch.from_numpy(h2_vec.astype(np.float32))
        with torch.no_grad():
            out = m.layers[6:](x)   # Linear, ReLU, Linear
        return out.numpy()

    base_logits = acts_to_logits(X_te)
    base_country_logit = base_logits[:, COUNTRY_IDX]

    # Intervention A: project F=1 examples ONTO the manifold subspace.
    #   x' = mu + V V^T (x - mu)
    print("\n  A. Project F=1 examples ONTO the K=4 manifold subspace")
    Xc = X_te - mu
    onto = mu + (Xc @ V_mf) @ V_mf.T
    logits_onto = acts_to_logits(onto)
    cnt_logit_onto = logits_onto[:, COUNTRY_IDX]
    print(f"    F=1: mean country logit before {base_country_logit[y_te==1].mean():.3f} "
          f"-> after {cnt_logit_onto[y_te==1].mean():.3f}")
    print(f"    F=0: mean country logit before {base_country_logit[y_te==0].mean():.3f} "
          f"-> after {cnt_logit_onto[y_te==0].mean():.3f}")
    print(f"    F=0 country-acc before {((base_country_logit[y_te==0] > 0).mean()):.4f} "
          f"-> after {((cnt_logit_onto[y_te==0] > 0).mean()):.4f} (proj onto manifold)")

    # Intervention B: REMOVE the manifold component (x' = mu + perp_part).
    print("\n  B. REMOVE the K=4 manifold component (project to its orthogonal complement)")
    perp = mu + (Xc - (Xc @ V_mf) @ V_mf.T)
    logits_perp = acts_to_logits(perp)
    cnt_logit_perp = logits_perp[:, COUNTRY_IDX]
    print(f"    F=1: mean country logit before {base_country_logit[y_te==1].mean():.3f} "
          f"-> after {cnt_logit_perp[y_te==1].mean():.3f}")
    print(f"    F=1 country-acc before {((base_country_logit[y_te==1] > 0).mean()):.4f} "
          f"-> after {((cnt_logit_perp[y_te==1] > 0).mean()):.4f} (manifold removed)")

    # Intervention C: scramble PERPENDICULAR-to-manifold directions for F=0.
    #   Goal: increase off-manifold distance; should make country prediction even
    #   *more* country=0.
    print("\n  C. AMPLIFY perpendicular component (scale by 2x) for F=0")
    Xc_amp = Xc.copy()
    par = (Xc @ V_mf) @ V_mf.T
    perp_comp = Xc - par
    Xc_amp = par + 2.0 * perp_comp
    amp = mu + Xc_amp
    logits_amp = acts_to_logits(amp)
    cnt_logit_amp = logits_amp[:, COUNTRY_IDX]
    print(f"    F=0: mean country logit before {base_country_logit[y_te==0].mean():.3f} "
          f"-> after {cnt_logit_amp[y_te==0].mean():.3f}")
    print(f"    F=0 country-acc before {((base_country_logit[y_te==0] < 0).mean()):.4f} "
          f"-> after {((cnt_logit_amp[y_te==0] < 0).mean()):.4f} (perp amplified)")

    # Cross-feature isolation: did interventions also break OTHER feature predictions?
    print("\n  D. Selectivity: did Intervention A also disturb OTHER feature accuracies?")
    base_preds = (base_logits > 0).astype(np.int64)
    onto_preds = (logits_onto > 0).astype(np.int64)
    y_full = test["labels"]
    for fi in range(8):
        b = (base_preds[:, fi] == y_full[:, fi]).mean()
        a = (onto_preds[:, fi] == y_full[:, fi]).mean()
        print(f"    {FEATURE_NAMES[fi]:<10}: acc before {b:.4f} after {a:.4f}  delta {a - b:+.4f}")

    # ----- Q2: What does the manifold axis encode? ------------------------- #
    print("\n[Q2] What is encoded along the country manifold's principal axis?")
    texts_tr, _, _ = load_split("train")
    tags = tag_countries(texts_tr)
    by_country = {}
    for x, t in zip(X_tr, tags):
        if t is None:
            continue
        by_country.setdefault(t, []).append(x)

    # Per-country centroid → PC1 projection
    centroid = {c: np.stack(arr).mean(0) for c, arr in by_country.items() if len(arr) >= 10}
    counts = {c: len(arr) for c, arr in by_country.items() if len(arr) >= 10}
    print(f"  {len(centroid)} countries with >=10 training examples")

    proj1 = {c: float((centroid[c] - mu) @ v_mf) for c in centroid}
    sorted_by_proj = sorted(proj1.items(), key=lambda kv: kv[1])
    print("\n  Top 10 countries at NEGATIVE end of v_manifold (PC1):")
    for c, p in sorted_by_proj[:10]:
        print(f"    {c:<25} proj={p:+.3f}  n={counts[c]}")
    print("\n  Top 10 countries at POSITIVE end of v_manifold (PC1):")
    for c, p in sorted_by_proj[-10:]:
        print(f"    {c:<25} proj={p:+.3f}  n={counts[c]}")

    # MiniLM embedding of each bare country name → see if its h2 PC1 ~ MiniLM PC1
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    countries = sorted(centroid.keys())
    name_emb = enc.encode(countries, convert_to_numpy=True)  # (n_countries, 384)
    # PCA of bare-name embeddings vs h2-centroid positions on v_mf
    from sklearn.decomposition import PCA
    pca = PCA(n_components=4).fit(name_emb)
    proj_name = pca.transform(name_emb)  # (n_countries, 4)
    proj_h2 = np.array([proj1[c] for c in countries])

    # Correlation between h2_proj and each name_emb PC
    for k in range(4):
        r = np.corrcoef(proj_h2, proj_name[:, k])[0, 1]
        print(f"  Pearson(country-h2-PC1, MiniLM-name-PC{k+1}) = {r:+.3f}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    sc = ax.scatter(proj_name[:, 0], proj_h2, s=20, alpha=0.7)
    for i, c in enumerate(countries):
        if i % 5 == 0:
            ax.annotate(c, (proj_name[i, 0], proj_h2[i]), fontsize=7, alpha=0.7)
    ax.set_xlabel("MiniLM-of-country-name PC1")
    ax.set_ylabel("country-h2-centroid projected on v_manifold")
    ax.set_title("Does h2 country axis reflect MiniLM-encoded semantic structure?")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "axis_interpretation.png", dpi=130)
    plt.close(fig)
    print(f"  saved -> {FIG_DIR / 'axis_interpretation.png'}")


if __name__ == "__main__":
    main()
