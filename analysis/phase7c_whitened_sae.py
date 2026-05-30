"""Phase 7c: pre-whiten h2 with a class-aware basis, then train a standard SAE.

The country feature lives in the COVARIANCE ratio cov(F=0) / cov(F=1). Whitening
h2 by cov(F=0) makes F=0 isotropic; F=1 will then be a shrunk Gaussian in the
direction where its eigenvalues are smaller than F=0's. After whitening,
"is country" might be captureable as a sparse LINEAR feature.

This is the constructive complement to the SAE-fails finding: if whitening
unlocks country detection by a standard SAE, that documents what *kind* of
pre-processing safety practitioners need.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import FEATURE_NAMES, get_activations

FIG_DIR = Path(__file__).parent / "figs"
COUNTRY_IDX = FEATURE_NAMES.index("country")


class TopKSAE(nn.Module):
    def __init__(self, d=64, m=256, k=8):
        super().__init__()
        self.enc = nn.Linear(d, m, bias=True)
        self.dec = nn.Linear(m, d, bias=True)
        self.k = k

    def forward(self, x):
        pre = self.enc(x)
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        mask = torch.zeros_like(pre)
        mask.scatter_(-1, topk_idx, topk_vals.clamp(min=0.0))
        recon = self.dec(mask)
        return recon, mask


def main():
    train = get_activations("train")
    test = get_activations("test")
    X_tr = train["h2"].astype(np.float32)
    X_te = test["h2"].astype(np.float32)
    y_tr = train["labels"][:, COUNTRY_IDX]
    y_te = test["labels"][:, COUNTRY_IDX]

    # ---- Whitening basis: ZCA-whiten by cov(F=0) ---- #
    mu = X_tr.mean(0)
    Xn = X_tr[y_tr == 0] - mu
    cov_n = np.cov(Xn.T)
    # cov_n^(-1/2) for ZCA whitening of F=0
    eigvals, V = np.linalg.eigh(cov_n)
    eps = 1e-3
    cov_n_inv_half = V @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, eps))) @ V.T
    # Pre-process: x_w = cov_n^{-1/2} (x - mu)
    Xtr_w = (X_tr - mu) @ cov_n_inv_half
    Xte_w = (X_te - mu) @ cov_n_inv_half

    # Check geometry post-whitening
    Xp_w = Xtr_w[y_tr == 1]
    Xn_w = Xtr_w[y_tr == 0]
    mu_p_w = Xp_w.mean(0)
    mu_n_w = Xn_w.mean(0)
    cov_p_w = np.cov((Xp_w - mu_p_w).T)
    print(f"After whitening by cov(F=0):")
    print(f"  ||mean(F=1) - mean(F=0)|| = {np.linalg.norm(mu_p_w - mu_n_w):.4f}")
    print(f"  trace(cov F=1) = {cov_p_w.trace():.4f}")
    print(f"  trace(cov F=0) = (should be ~64) {np.cov((Xn_w - mu_n_w).T).trace():.4f}")
    print(f"  top 5 eigvals cov(F=1): {np.sort(np.linalg.eigvalsh(cov_p_w))[::-1][:5]}")

    # Now: linear probe on whitened h2 — does country become linearly readable?
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr_w)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr_w), y_tr)
    acc_lin_w = clf.score(sc.transform(Xte_w), y_te)
    print(f"\nLinear probe on whitened h2: {acc_lin_w:.4f}  "
          f"(was {0.469:.4f} on raw h2)")

    # Standard linear SAE on the whitened activations
    Xtr_t = torch.from_numpy(Xtr_w.astype(np.float32))
    Xte_t = torch.from_numpy(Xte_w.astype(np.float32))

    torch.manual_seed(0)
    sae = TopKSAE(d=64, m=256, k=8)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    bs = 256
    n = len(Xtr_t)
    for ep in range(80):
        perm = torch.randperm(n)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            recon, _ = sae(Xtr_t[idx])
            loss = ((recon - Xtr_t[idx]) ** 2).mean()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        if ep % 10 == 0 or ep == 79:
            with torch.no_grad():
                recon_te, _ = sae(Xte_t)
                mse_te = ((recon_te - Xte_t) ** 2).mean().item()
            print(f"  ep {ep:3d}  train MSE {ep_loss/nb:.4f}  test MSE {mse_te:.4f}")

    # Selectivity analysis
    with torch.no_grad():
        _, codes_tr = sae(Xtr_t)
        _, codes_te = sae(Xte_t)
    codes_tr_np = codes_tr.numpy()
    codes_te_np = codes_te.numpy()
    fires = codes_tr_np > 1e-4
    p_pos = fires[y_tr == 1].mean(0)
    p_neg = fires[y_tr == 0].mean(0)
    log_odds = np.log(p_pos + 1e-6) - np.log(p_neg + 1e-6)
    order = np.argsort(-log_odds)
    print("\nTop 10 country=1-selective SAE features (WHITENED-input SAE):")
    for f in order[:10]:
        print(f"  feat {f:3d}  P(fire|F=1)={p_pos[f]:.3f}  "
              f"P(fire|F=0)={p_neg[f]:.3f}  log-odds={log_odds[f]:+.2f}")

    print("\nLinear probe on top-K country-selective features (WHITENED-SAE):")
    for K in [1, 2, 4, 8, 16, 32, 64, 256]:
        sel = order[:K]
        sc2 = StandardScaler().fit(codes_tr_np[:, sel])
        clf2 = LogisticRegression(max_iter=2000).fit(
            sc2.transform(codes_tr_np[:, sel]), y_tr)
        acc = clf2.score(sc2.transform(codes_te_np[:, sel]), y_te)
        print(f"  top {K:3d}: {acc:.4f}")

    # Best single feature
    print("\nSingle-feature LR scan over all 256 SAE features:")
    best = []
    for f in range(256):
        sc2 = StandardScaler().fit(codes_tr_np[:, f:f+1])
        clf2 = LogisticRegression(max_iter=2000).fit(
            sc2.transform(codes_tr_np[:, f:f+1]), y_tr)
        acc = clf2.score(sc2.transform(codes_te_np[:, f:f+1]), y_te)
        best.append((acc, f))
    best.sort(reverse=True)
    for acc, f in best[:10]:
        print(f"  feat {f:3d}  1-feature LR = {acc:.4f}  log-odds={log_odds[f]:+.2f}")


if __name__ == "__main__":
    main()
