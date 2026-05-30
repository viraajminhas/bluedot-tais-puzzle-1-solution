"""Phase 2: characterise the geometric structure of feature F at hidden 2.

Hypothesis families tested:
  A. Radial / magnitude-superposition (Gorton): F = (||proj_S(acts)|| > thresh)
  B. Circular (Engels): F-positive acts form a ring in some 2D subspace
  C. XOR / interaction with another feature G
  D. Multi-cluster: F-positive splits into K sub-clusters with different mean directions
  E. Parity over multiple latent binary dirs
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import FEATURE_NAMES, N_FEATURES, get_activations

FIG_DIR = Path(__file__).parent / "figs"
FIG_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _logreg(X_tr, y_tr, X_te, y_te, C=1.0):
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear").fit(
        scaler.transform(X_tr), y_tr
    )
    return clf.score(scaler.transform(X_te), y_te), clf, scaler


def _lda_direction(X, y):
    mu1 = X[y == 1].mean(0)
    mu0 = X[y == 0].mean(0)
    d = mu1 - mu0
    return d / (np.linalg.norm(d) + 1e-9)


# --------------------------------------------------------------------------- #
# Hypothesis tests
# --------------------------------------------------------------------------- #
def test_radial(X_tr, y_tr, X_te, y_te, *, k=8):
    """If F is encoded radially in a k-D subspace, the *norm* of the projection
    should be a far better predictor than the projection itself.

    Procedure: pick the top-k LDA / class-mean-diff directions, project, then
    classify by ||proj|| via a 1-feature LR.
    """
    # Centre, then take top-k principal directions of (X[y=1] - mean(X)).
    mu = X_tr.mean(0)
    Xc_tr = X_tr - mu
    Xc_te = X_te - mu

    # Build a candidate subspace by PCA on F=1 examples (with mean removed).
    Xp = Xc_tr[y_tr == 1]
    if len(Xp) < k:
        return None
    pca = PCA(n_components=k).fit(Xp)
    P_tr = pca.transform(Xc_tr)
    P_te = pca.transform(Xc_te)

    # Feature = squared L2 norm of projection
    feat_tr = (P_tr ** 2).sum(axis=1, keepdims=True)
    feat_te = (P_te ** 2).sum(axis=1, keepdims=True)

    clf = LogisticRegression(max_iter=2000).fit(feat_tr, y_tr)
    acc = clf.score(feat_te, y_te)
    return {
        "subspace_dim": k,
        "acc_from_norm": acc,
        "mean_norm_pos": float(feat_tr[y_tr == 1].mean() ** 0.5),
        "mean_norm_neg": float(feat_tr[y_tr == 0].mean() ** 0.5),
    }


def test_xor_with_other(X_tr, y_tr_F, all_labels_tr, X_te, y_te_F, all_labels_te, fi):
    """For each G != F, train a linear probe on F XOR G. If the XOR is linearly
    separable from X (and F itself is NOT), then F is encoded via XOR-like
    interaction with G.
    """
    results = []
    for gi in range(N_FEATURES):
        if gi == fi:
            continue
        y_xor_tr = (y_tr_F ^ all_labels_tr[:, gi]).astype(np.int64)
        y_xor_te = (y_te_F ^ all_labels_te[:, gi]).astype(np.int64)
        acc, _, _ = _logreg(X_tr, y_xor_tr, X_te, y_xor_te)
        results.append((acc, FEATURE_NAMES[gi]))
    results.sort(reverse=True)
    return results  # list of (acc, feature_name)


def test_clusters_within_positive(X_tr, y_tr, X_te, y_te, *, max_k=6):
    """Run k-means on F=1 examples for k=2..max_k, then for each cluster train a
    linear probe (cluster vs F=0). If clusters give much higher accuracy than
    raw F probe, F lives as a *union of K linear sub-features* — non-linear
    overall but a clean union of linear pieces.
    """
    Xp_tr = X_tr[y_tr == 1]
    Xn_tr = X_tr[y_tr == 0]
    Xp_te = X_te[y_te == 1]
    Xn_te = X_te[y_te == 0]

    results = {}
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xp_tr)
        sub_accs = []
        for c in range(k):
            mask = km.labels_ == c
            if mask.sum() < 20:
                continue
            Xc = Xp_tr[mask]
            # Train probe: this cluster vs F=0
            yc_tr = np.concatenate([np.ones(mask.sum()), np.zeros(len(Xn_tr))])
            Xc_tr = np.vstack([Xc, Xn_tr])
            # Assign test points to this cluster (within F=1 test)
            test_clusters = km.predict(Xp_te) == c
            yc_te = np.concatenate([np.ones(test_clusters.sum()), np.zeros(len(Xn_te))])
            Xc_te = np.vstack([Xp_te[test_clusters], Xn_te])
            acc, _, _ = _logreg(Xc_tr, yc_tr, Xc_te, yc_te)
            sub_accs.append((mask.sum(), acc))
        results[k] = sub_accs
    return results


def test_circular(X_tr, y_tr, fi_name: str):
    """Look for circular / annular structure: in the top-2 PCA plane of the
    F=1 examples (with global mean removed), do positives lie on a *ring*
    while negatives sit at the centre?
    """
    mu = X_tr.mean(0)
    Xc = X_tr - mu
    pca = PCA(n_components=2).fit(Xc)
    P = pca.transform(Xc)

    Pp = P[y_tr == 1]
    Pn = P[y_tr == 0]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(Pn[:, 0], Pn[:, 1], s=4, alpha=0.4, label=f"{fi_name}=0", c="lightgrey")
    ax[0].scatter(Pp[:, 0], Pp[:, 1], s=6, alpha=0.6, label=f"{fi_name}=1", c="C3")
    ax[0].set_title(f"PCA(top 2 on F=1) — coloured by {fi_name}")
    ax[0].legend()
    ax[0].set_aspect("equal")

    # Radial histogram: ||P|| distribution per class
    norms_p = np.linalg.norm(Pp, axis=1)
    norms_n = np.linalg.norm(Pn, axis=1)
    ax[1].hist(norms_n, bins=40, alpha=0.5, label=f"{fi_name}=0", color="lightgrey")
    ax[1].hist(norms_p, bins=40, alpha=0.7, label=f"{fi_name}=1", color="C3")
    ax[1].set_xlabel("||projection onto top-2 plane||")
    ax[1].set_title("Radial distribution")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"circular_{fi_name}.png", dpi=130)
    plt.close(fig)

    return {
        "fig": str(FIG_DIR / f"circular_{fi_name}.png"),
        "mean_norm_pos": float(norms_p.mean()),
        "mean_norm_neg": float(norms_n.mean()),
        "median_norm_pos": float(np.median(norms_p)),
        "median_norm_neg": float(np.median(norms_n)),
    }


def causal_projection_ablation(X_tr, y_tr, X_te, y_te, direction):
    """Project out a 1D direction from acts, then re-train linear probe.
    Useful to test whether a hypothesised direction *causally* carries F.
    """
    d = direction / (np.linalg.norm(direction) + 1e-9)
    proj_tr = X_tr - np.outer(X_tr @ d, d)
    proj_te = X_te - np.outer(X_te @ d, d)
    acc_before, _, _ = _logreg(X_tr, y_tr, X_te, y_te)
    acc_after, _, _ = _logreg(proj_tr, y_tr, proj_te, y_te)
    return {"before": acc_before, "after": acc_after, "drop": acc_before - acc_after}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=int, required=True,
                        help="Feature index to characterise (0..7)")
    parser.add_argument("--layer", default="h2", choices=["embeddings", "h0", "h1", "h2", "h3"])
    args = parser.parse_args()

    name = FEATURE_NAMES[args.feature]
    print(f"\n=== Characterising feature F = {name} (idx {args.feature}) at {args.layer} ===\n")

    train = get_activations("train")
    test = get_activations("test")
    X_tr, X_te = train[args.layer], test[args.layer]
    y_tr = train["labels"][:, args.feature]
    y_te = test["labels"][:, args.feature]

    # ---- A. Radial ----
    print("[A] Radial / magnitude superposition")
    for k in [2, 4, 8, 16, 32]:
        r = test_radial(X_tr, y_tr, X_te, y_te, k=k)
        print(f"   k={k:2d}: acc_from_||proj||^2 = {r['acc_from_norm']:.4f} "
              f"(mean norm pos={r['mean_norm_pos']:.3f}, neg={r['mean_norm_neg']:.3f})")

    # ---- B. Circular plot ----
    print("\n[B] Circular / annular structure")
    r = test_circular(X_tr, y_tr, name)
    print(f"   mean ||top-2 proj|| pos={r['mean_norm_pos']:.3f}  neg={r['mean_norm_neg']:.3f}")
    print(f"   figure -> {r['fig']}")

    # ---- C. XOR with another feature ----
    print("\n[C] XOR with another feature G (linear probe on F XOR G)")
    xor_results = test_xor_with_other(X_tr, y_tr, train["labels"], X_te, y_te,
                                       test["labels"], args.feature)
    for acc, gname in xor_results[:5]:
        print(f"   {name:>10} XOR {gname:<10}: linear-probe acc {acc:.4f}")

    # ---- D. Multi-cluster within F=1 ----
    print("\n[D] Multi-cluster within F=1 (k-means then per-cluster linear probe)")
    clusters = test_clusters_within_positive(X_tr, y_tr, X_te, y_te)
    for k, accs in clusters.items():
        accs_fmt = ", ".join(f"({sz},{a:.3f})" for sz, a in accs)
        print(f"   k={k}: per-cluster (size,acc) = {accs_fmt}")

    # ---- E. Causal ablation of LDA direction ----
    print("\n[E] Causal ablation: project out LDA direction, retrain linear probe")
    d = _lda_direction(X_tr, y_tr)
    abl = causal_projection_ablation(X_tr, y_tr, X_te, y_te, d)
    print(f"   linear acc before={abl['before']:.4f}  after={abl['after']:.4f}  drop={abl['drop']:+.4f}")


if __name__ == "__main__":
    main()
