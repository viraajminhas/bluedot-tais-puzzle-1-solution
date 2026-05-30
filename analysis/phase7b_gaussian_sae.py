"""Phase 7b: covariance-aware Gaussian-RBF SAE on h2.

Standard SAEs use linear features ReLU(W·x). They cannot capture features that
are SHAPED LIKE A GAUSSIAN BLOB in activation space (e.g. country=1, which
inhabits a thin region whose mean coincides with country=0's mean).

A Gaussian-RBF SAE has features f_k(x) = exp(-||x - mu_k||^2 / sigma_k^2),
which fire when x is close to a LEARNED PROTOTYPE. Each prototype + bandwidth
can capture a covariance-shape feature: a single prototype near the country
manifold should fire often for country=1 and rarely for country=0.

We train and check whether any single Gaussian feature is country-discriminative
(top-1 LR -> high acc), unlike the standard top-k SAE where top-1 ≈ chance.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common import FEATURE_NAMES, get_activations

FIG_DIR = Path(__file__).parent / "figs"
COUNTRY_IDX = FEATURE_NAMES.index("country")


class GaussianSAE(nn.Module):
    """K Gaussian-RBF features with learned prototypes mu_k and bandwidths
    sigma_k (per-dimension diagonal). Decoder is linear from feature
    activations back to input. Top-k sparsity for selecting which features
    fire."""

    def __init__(self, d=64, K=256, top_k=8, init_std=0.5):
        super().__init__()
        # Prototypes: shape (K, d)
        self.mu = nn.Parameter(torch.randn(K, d) * init_std)
        # Log of per-(feature, dim) bandwidth: shape (K, d)
        self.log_sigma = nn.Parameter(torch.zeros(K, d))
        # Decoder: linear map from K features back to d
        self.dec = nn.Linear(K, d, bias=True)
        self.K = K
        self.top_k = top_k

    def features(self, x):
        # x: (B, d)  mu: (K, d)  sigma: (K, d)
        sigma = torch.exp(self.log_sigma).clamp(min=1e-3, max=10.0)
        # Squared Mahalanobis-ish distance with diagonal sigma:
        # ||(x - mu) / sigma||^2 = sum_d ((x_d - mu_kd) / sigma_kd)^2
        # Shape: (B, K)
        diff = x.unsqueeze(1) - self.mu.unsqueeze(0)         # (B, K, d)
        dist_sq = ((diff / sigma.unsqueeze(0)) ** 2).sum(-1)  # (B, K)
        # Gaussian RBF feature activation
        f = torch.exp(-0.5 * dist_sq)
        return f

    def forward(self, x):
        f = self.features(x)
        # Top-k sparsity
        topk_vals, topk_idx = f.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(f)
        mask.scatter_(-1, topk_idx, topk_vals)
        recon = self.dec(mask)
        return recon, mask, f


def main():
    print("Loading activations...")
    train = get_activations("train")
    test = get_activations("test")
    X_tr = train["h2"].astype(np.float32)
    X_te = test["h2"].astype(np.float32)
    y_tr = train["labels"][:, COUNTRY_IDX]
    y_te = test["labels"][:, COUNTRY_IDX]

    # Pre-standardise
    mu = X_tr.mean(0)
    sd = X_tr.std(0) + 1e-6
    Xtr = torch.from_numpy((X_tr - mu) / sd)
    Xte = torch.from_numpy((X_te - mu) / sd)

    # Initialise some prototypes near class means, the rest random.
    # In particular, seed one prototype at mean(F=1) and one at mean(F=0).
    K = 256
    sae = GaussianSAE(d=64, K=K, top_k=8, init_std=0.7)
    with torch.no_grad():
        mu_pos = ((X_tr[y_tr == 1].mean(0) - mu) / sd)
        mu_neg = ((X_tr[y_tr == 0].mean(0) - mu) / sd)
        sae.mu[0] = torch.from_numpy(mu_pos.astype(np.float32))
        sae.mu[1] = torch.from_numpy(mu_neg.astype(np.float32))

    opt = torch.optim.Adam(sae.parameters(), lr=2e-3, weight_decay=0.0)
    bs = 256
    n = len(Xtr)
    epochs = 80

    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            recon, _, _ = sae(Xtr[idx])
            loss = ((recon - Xtr[idx]) ** 2).mean()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        if ep % 10 == 0 or ep == epochs - 1:
            with torch.no_grad():
                recon_te, _, _ = sae(Xte)
                mse_te = ((recon_te - Xte) ** 2).mean().item()
            print(f"  ep {ep:3d}  train MSE {ep_loss/nb:.4f}  test MSE {mse_te:.4f}")

    # Analysis: per-feature firing on country=1 vs country=0
    print("\nAnalysing per-feature selectivity...")
    with torch.no_grad():
        _, codes_tr, _ = sae(Xtr)
        _, codes_te, _ = sae(Xte)
    codes_tr_np = codes_tr.numpy()
    codes_te_np = codes_te.numpy()

    # Selectivity: mean firing on F=1 vs F=0
    mean_fire_pos = codes_tr_np[y_tr == 1].mean(0)
    mean_fire_neg = codes_tr_np[y_tr == 0].mean(0)
    # Each feature's bandwidth (geometric mean of per-dim sigmas)
    sigma = torch.exp(sae.log_sigma).detach().numpy()
    log_odds = np.log(mean_fire_pos + 1e-6) - np.log(mean_fire_neg + 1e-6)
    order = np.argsort(-log_odds)

    print("\nTop 10 Gaussian SAE features by country=1 selectivity (log fire ratio):")
    for f in order[:10]:
        print(f"  feat {f:3d}  mean_fire(F=1)={mean_fire_pos[f]:.4f}  "
              f"mean_fire(F=0)={mean_fire_neg[f]:.4f}  "
              f"log-odds={log_odds[f]:+.3f}  "
              f"bandwidth_mean={float(sigma[f].mean()):.3f}")

    print("\nTop 10 by country=0 selectivity:")
    for f in order[-10:][::-1]:
        print(f"  feat {f:3d}  mean_fire(F=1)={mean_fire_pos[f]:.4f}  "
              f"mean_fire(F=0)={mean_fire_neg[f]:.4f}  "
              f"log-odds={log_odds[f]:+.3f}  "
              f"bandwidth_mean={float(sigma[f].mean()):.3f}")

    # Linear probe scaling test
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    print("\nLinear probe on top-K most country=1-selective Gaussian-SAE features:")
    for K_ in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        sel = order[:K_]
        Xtr_sel = codes_tr_np[:, sel]
        Xte_sel = codes_te_np[:, sel]
        sc = StandardScaler().fit(Xtr_sel)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr_sel), y_tr)
        acc = clf.score(sc.transform(Xte_sel), y_te)
        print(f"  top {K_:3d} features  -> {acc:.4f}")

    # 1-feature scan: for EACH SAE feature, train a 1-feature LR and report the
    # best country accuracy. This is the cleanest test: does ANY single feature
    # carry significant country signal?
    print("\nSingle-feature scan: best country-acc over all 256 features:")
    sc = StandardScaler().fit(codes_tr_np)
    Xtr_s = sc.transform(codes_tr_np)
    Xte_s = sc.transform(codes_te_np)
    best_accs = []
    for f in range(K):
        clf = LogisticRegression(max_iter=2000).fit(Xtr_s[:, f:f+1], y_tr)
        acc = clf.score(Xte_s[:, f:f+1], y_te)
        best_accs.append((acc, f))
    best_accs.sort(reverse=True)
    for acc, f in best_accs[:10]:
        print(f"  feat {f:3d}  1-feature LR acc = {acc:.4f}  "
              f"log-odds={log_odds[f]:+.3f}")

    # Now compare to the SAME experiment on the LINEAR (puzzle) SAE.
    # (Same single-feature scan, but with codes from the linear top-k SAE from
    # phase 5b. Reload them here.)
    print("\nFor reference (recap from phase 5b):")
    print("  Linear-SAE top-1 country-selective feature: 50.3% (chance)")
    print("  Linear-SAE top-64 selective features:        64.1%")
    print("  Linear-SAE full 256-D probe:                  92.1%")

    # Visualise the country=1-selective feature in 2D PCA
    if best_accs[0][0] > 0.75:
        best_feat = best_accs[0][1]
        with torch.no_grad():
            f_all_tr = sae.features(Xtr).numpy()
        from sklearn.decomposition import PCA
        Xp = X_tr[y_tr == 1] - X_tr[y_tr == 1].mean(0)
        pca = PCA(n_components=2).fit(Xp)
        proj = pca.transform(X_tr - mu)
        fire = f_all_tr[:, best_feat]
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=fire, cmap="viridis", s=4, alpha=0.6)
        plt.colorbar(sc, ax=ax, label=f"Gaussian SAE feat {best_feat} activation")
        ax.set_title(f"Gaussian SAE feature {best_feat} firing (covariance-aware)\n"
                     f"single-feature LR acc = {best_accs[0][0]:.3f}")
        ax.set_xlabel("PC1 of cov(F=1)")
        ax.set_ylabel("PC2 of cov(F=1)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "gaussian_sae_top_feature.png", dpi=140)
        plt.close(fig)
        print(f"  saved -> {FIG_DIR / 'gaussian_sae_top_feature.png'}")


if __name__ == "__main__":
    main()
