"""Phase 5b: (i) output-gradient alignment with the manifold axis, and
(ii) a small sparse autoencoder on h2.

(i) is a *causal* test of the axis: if the gradient of the country logit w.r.t.
    h2 aligns with v_manifold, the axis is the direction the model actually
    READS to predict country.
(ii) SAE on h2: train an overcomplete dictionary with top-k sparsity, look at
    which features fire for country=1 vs country=0. If the SAE discovers K
    sparse "country prototypes," the manifold is a union of K direction-features.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common import FEATURE_NAMES, get_activations, load_head

FIG_DIR = Path(__file__).parent / "figs"
COUNTRY_IDX = FEATURE_NAMES.index("country")


# --------------------------------------------------------------------------- #
# (i) Gradient alignment
# --------------------------------------------------------------------------- #
def gradient_alignment():
    print("=== (i) Country-logit gradient w.r.t. h2 ===")
    train = get_activations("train")
    X_tr = train["h2"]
    y_tr = train["labels"][:, COUNTRY_IDX]

    # Build a tiny torch module: just l4, ReLU, l5 of the puzzle head, taking h2 as input.
    m = load_head("cpu")
    l4, l5 = m.layers[6], m.layers[8]

    # Compute per-example gradient of country logit w.r.t. h2 input
    x = torch.from_numpy(X_tr.astype(np.float32)).requires_grad_(True)
    h3 = torch.relu(l4(x))
    logits = l5(h3)
    country_logit = logits[:, COUNTRY_IDX]
    # Sum gradient: ∂(sum logit_country) / ∂x = sum of per-example gradients
    grad = torch.autograd.grad(country_logit.sum(), x)[0].detach().numpy()
    # Per-class mean gradient direction
    grad_pos_mean = grad[y_tr == 1].mean(0)
    grad_neg_mean = grad[y_tr == 0].mean(0)
    grad_overall = grad.mean(0)

    # Manifold direction from cov(F=1)
    Xp_c = X_tr[y_tr == 1] - X_tr[y_tr == 1].mean(0)
    cov_p = np.cov(Xp_c.T)
    eigvals, eigvecs = np.linalg.eigh(cov_p)
    v_mf = eigvecs[:, -1]

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    print(f"  |grad_country|_F=1 = {np.linalg.norm(grad_pos_mean):.3f}")
    print(f"  |grad_country|_F=0 = {np.linalg.norm(grad_neg_mean):.3f}")
    print(f"  cos(grad_country, v_manifold)            = {cos(grad_overall, v_mf):+.3f}")
    print(f"  cos(grad_country_F=1_mean, v_manifold)   = {cos(grad_pos_mean, v_mf):+.3f}")
    print(f"  cos(grad_country_F=0_mean, v_manifold)   = {cos(grad_neg_mean, v_mf):+.3f}")

    # Project each example's grad onto v_manifold
    grad_proj = grad @ v_mf
    print(f"  mean(grad_country · v_manifold) on F=1 examples: {grad_proj[y_tr==1].mean():+.4f}")
    print(f"  mean(grad_country · v_manifold) on F=0 examples: {grad_proj[y_tr==0].mean():+.4f}")
    print(f"  sign(grad·v_mf) flips between F=1 and F=0?  "
          f"{'YES' if grad_proj[y_tr==1].mean() * grad_proj[y_tr==0].mean() < 0 else 'no'}")

    # Stronger test: rank correlation between |grad·v_mf| and perpendicular distance.
    # If country logit is computed from perp-distance, |grad| should be LARGER for points off-manifold.
    perp = X_tr - X_tr.mean(0) - np.outer((X_tr - X_tr.mean(0)) @ v_mf, v_mf)
    perp_dist = np.linalg.norm(perp, axis=1)
    grad_norm = np.linalg.norm(grad, axis=1)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(grad_norm, perp_dist)
    print(f"  Spearman(|grad country logit|, perp-distance-from-manifold) = {rho:+.3f}")


# --------------------------------------------------------------------------- #
# (ii) Sparse autoencoder on h2
# --------------------------------------------------------------------------- #
class TopKSAE(nn.Module):
    def __init__(self, d=64, m=256, k=8):
        super().__init__()
        self.enc = nn.Linear(d, m, bias=True)
        self.dec = nn.Linear(m, d, bias=True)
        self.k = k

    def forward(self, x):
        pre = self.enc(x)
        # Top-K sparsity
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        mask = torch.zeros_like(pre)
        mask.scatter_(-1, topk_idx, topk_vals.clamp(min=0.0))
        recon = self.dec(mask)
        return recon, mask


def train_sae():
    print("\n=== (ii) Sparse autoencoder on h2 ===")
    train = get_activations("train")
    test = get_activations("test")
    X_tr = train["h2"].astype(np.float32)
    X_te = test["h2"].astype(np.float32)
    y_tr = train["labels"][:, COUNTRY_IDX]
    y_te = test["labels"][:, COUNTRY_IDX]

    mu = X_tr.mean(0)
    sd = X_tr.std(0) + 1e-6
    Xtr = torch.from_numpy((X_tr - mu) / sd)
    Xte = torch.from_numpy((X_te - mu) / sd)

    torch.manual_seed(0)
    sae = TopKSAE(d=64, m=256, k=8)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3, weight_decay=0.0)
    bs = 256
    n = len(Xtr)

    for ep in range(80):
        perm = torch.randperm(n)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            recon, _ = sae(Xtr[idx])
            loss = ((recon - Xtr[idx]) ** 2).mean()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        if ep % 10 == 0 or ep == 79:
            with torch.no_grad():
                recon_te, _ = sae(Xte)
                mse_te = ((recon_te - Xte) ** 2).mean().item()
            print(f"  ep {ep:3d}  train MSE {ep_loss/nb:.4f}  test MSE {mse_te:.4f}")

    # Analyse: for each SAE feature, how often it fires on F=1 vs F=0
    with torch.no_grad():
        _, codes_tr = sae(Xtr)
    codes_np = codes_tr.numpy()    # (N, 256)
    fires = codes_np > 1e-4

    # Per-feature: P(fire | F=1), P(fire | F=0), and their selectivity ratio
    p_fire_pos = fires[y_tr == 1].mean(0)
    p_fire_neg = fires[y_tr == 0].mean(0)
    log_odds = np.log(p_fire_pos + 1e-6) - np.log(p_fire_neg + 1e-6)
    # Sort features by selectivity for F=1
    order = np.argsort(-log_odds)
    print("\n  Top 10 SAE features selective for country=1 (log-odds ratio):")
    for f in order[:10]:
        print(f"    feat {f:3d}  P(fire|F=1)={p_fire_pos[f]:.3f}  "
              f"P(fire|F=0)={p_fire_neg[f]:.3f}  log-odds={log_odds[f]:+.2f}")
    print("\n  Top 10 SAE features selective for country=0:")
    for f in order[-10:][::-1]:
        print(f"    feat {f:3d}  P(fire|F=1)={p_fire_pos[f]:.3f}  "
              f"P(fire|F=0)={p_fire_neg[f]:.3f}  log-odds={log_odds[f]:+.2f}")

    # How many SAE features needed to discriminate F=1 from F=0?
    # Linear probe on SAE codes
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    codes_tr_np = codes_tr.numpy()
    with torch.no_grad():
        _, codes_te = sae(Xte)
    codes_te_np = codes_te.numpy()

    sc = StandardScaler().fit(codes_tr_np)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(codes_tr_np), y_tr)
    acc = clf.score(sc.transform(codes_te_np), y_te)
    print(f"\n  Linear probe on SAE codes (256-D): test acc {acc:.4f}")

    # Try restricting to top-K most country-selective SAE features
    for K in [1, 2, 4, 8, 16, 32, 64]:
        sel = order[:K]
        sc2 = StandardScaler().fit(codes_tr_np[:, sel])
        clf2 = LogisticRegression(max_iter=2000).fit(sc2.transform(codes_tr_np[:, sel]), y_tr)
        acc2 = clf2.score(sc2.transform(codes_te_np[:, sel]), y_te)
        print(f"    using top-{K:2d} country=1-selective SAE features:   {acc2:.4f}")

    # Cumulative variance of country=1 in PCA of h2
    Xp = X_tr[y_tr == 1] - X_tr[y_tr == 1].mean(0)
    cov_p = np.cov(Xp.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov_p))[::-1]
    cum = np.cumsum(eigvals) / eigvals.sum()
    print(f"\n  Cumulative variance of cov(F=1) in h2 (effective manifold dim):")
    for k in [1, 2, 3, 5, 10, 20]:
        print(f"    top-{k:2d}: {cum[k-1]:.3f}")

    # Save eigenvalue plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(np.arange(1, 21), cum[:20], "o-", label="country=1 cumulative variance")
    Xn = X_tr[y_tr == 0] - X_tr[y_tr == 0].mean(0)
    cov_n = np.cov(Xn.T)
    eigvals_n = np.sort(np.linalg.eigvalsh(cov_n))[::-1]
    cum_n = np.cumsum(eigvals_n) / eigvals_n.sum()
    ax.plot(np.arange(1, 21), cum_n[:20], "s-", label="country=0 cumulative variance")
    ax.set_xlabel("# top eigenvalues")
    ax.set_ylabel("cumulative explained variance fraction")
    ax.set_title("Effective dimensionality of country=1 manifold vs country=0 cloud")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "manifold_dimensionality.png", dpi=140)
    plt.close(fig)
    print(f"  saved -> {FIG_DIR / 'manifold_dimensionality.png'}")


if __name__ == "__main__":
    gradient_alignment()
    train_sae()
