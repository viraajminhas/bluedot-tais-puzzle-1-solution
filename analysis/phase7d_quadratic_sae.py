"""Phase 7d: quadratic-feature SAE — detect covariance-shape features.

The country signal at h2 is purely SECOND-ORDER: same mean, different
covariance. Any first-order SAE (linear features) cannot capture it. We try
two cleanly-constructive moves:

  (1) Single hand-built quadratic feature: x' = cov_F=0^{-1/2}(x - mu);
      score = ||x'||^2 (the Hotelling T^2 statistic).
  (2) A learned 'quadratic SAE' with features f_k(x) = (w_k . x)^2 + b_k.
      Each feature detects magnitude along a learned direction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import FEATURE_NAMES, get_activations

COUNTRY_IDX = FEATURE_NAMES.index("country")


def main():
    train = get_activations("train")
    test = get_activations("test")
    X_tr = train["h2"].astype(np.float32)
    X_te = test["h2"].astype(np.float32)
    y_tr = train["labels"][:, COUNTRY_IDX]
    y_te = test["labels"][:, COUNTRY_IDX]

    # ---- (1) Hand-built Hotelling T² feature ----
    print("[1] Hand-built quadratic feature: Hotelling T^2 in cov(F=0)-whitened space")
    mu = X_tr.mean(0)
    Xn_c = X_tr[y_tr == 0] - mu
    cov_n = np.cov(Xn_c.T)
    eigvals, V = np.linalg.eigh(cov_n)
    eps = 1e-3
    cov_n_inv_half = V @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, eps))) @ V.T
    Xtr_w = (X_tr - mu) @ cov_n_inv_half
    Xte_w = (X_te - mu) @ cov_n_inv_half
    t2_tr = (Xtr_w ** 2).sum(axis=1)   # Hotelling T^2 per example
    t2_te = (Xte_w ** 2).sum(axis=1)
    print(f"  mean T^2 (F=1): {t2_tr[y_tr==1].mean():.3f}")
    print(f"  mean T^2 (F=0): {t2_tr[y_tr==0].mean():.3f}")
    # 1-feature linear probe on T^2
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000).fit(t2_tr.reshape(-1, 1), y_tr)
    acc = clf.score(t2_te.reshape(-1, 1), y_te)
    print(f"  1-feature LR on Hotelling T^2: test acc {acc:.4f}")

    # Also try the *symmetrised* version: combine T² with both class-conditional whitenings.
    Xp_c = X_tr[y_tr == 1] - mu
    cov_p = np.cov(Xp_c.T)
    eigvals_p, Vp = np.linalg.eigh(cov_p)
    cov_p_inv_half = Vp @ np.diag(1.0 / np.sqrt(np.maximum(eigvals_p, eps))) @ Vp.T
    Xtr_wp = (X_tr - mu) @ cov_p_inv_half
    Xte_wp = (X_te - mu) @ cov_p_inv_half
    t2p_tr = (Xtr_wp ** 2).sum(axis=1)
    t2p_te = (Xte_wp ** 2).sum(axis=1)
    print(f"\n  Hotelling T^2 in cov(F=1)-whitened space:")
    print(f"  mean T^2_p (F=1): {t2p_tr[y_tr==1].mean():.3f}")
    print(f"  mean T^2_p (F=0): {t2p_tr[y_tr==0].mean():.3f}")
    clf2 = LogisticRegression(max_iter=2000).fit(t2p_tr.reshape(-1, 1), y_tr)
    acc2 = clf2.score(t2p_te.reshape(-1, 1), y_te)
    print(f"  1-feature LR on T^2_p: test acc {acc2:.4f}")

    # Difference of T^2s = log likelihood ratio under same-mean QDA
    diff_tr = t2_tr - t2p_tr
    diff_te = t2_te - t2p_te
    clf3 = LogisticRegression(max_iter=2000).fit(diff_tr.reshape(-1, 1), y_tr)
    acc3 = clf3.score(diff_te.reshape(-1, 1), y_te)
    print(f"\n  1-feature LR on (T^2 − T^2_p) (= log-likelihood-ratio for QDA): {acc3:.4f}")

    # Also: 2-feature LR using BOTH
    Xtr_pair = np.stack([t2_tr, t2p_tr], axis=-1)
    Xte_pair = np.stack([t2_te, t2p_te], axis=-1)
    clf4 = LogisticRegression(max_iter=2000).fit(Xtr_pair, y_tr)
    acc4 = clf4.score(Xte_pair, y_te)
    print(f"  2-feature LR on (T^2, T^2_p): {acc4:.4f}")

    # ---- (2) Learned quadratic SAE ----
    print("\n[2] Learned quadratic SAE: features f_k(x) = (w_k · x)^2 + b_k.")
    class QuadSAE(nn.Module):
        def __init__(self, d=64, m=256, k=8):
            super().__init__()
            self.W = nn.Parameter(torch.randn(m, d) * 0.5)
            self.b = nn.Parameter(torch.zeros(m))
            self.dec = nn.Linear(m, d, bias=True)
            self.k = k

        def features(self, x):
            return (x @ self.W.T) ** 2 + self.b   # (B, m)

        def forward(self, x):
            f = self.features(x)
            topk_vals, topk_idx = f.topk(self.k, dim=-1)
            mask = torch.zeros_like(f)
            mask.scatter_(-1, topk_idx, topk_vals)
            recon = self.dec(mask)
            return recon, mask

    mu_t = X_tr.mean(0)
    sd_t = X_tr.std(0) + 1e-6
    Xtr_n = torch.from_numpy(((X_tr - mu_t) / sd_t).astype(np.float32))
    Xte_n = torch.from_numpy(((X_te - mu_t) / sd_t).astype(np.float32))
    torch.manual_seed(0)
    sae = QuadSAE(d=64, m=256, k=8)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    bs = 256
    n = len(Xtr_n)
    for ep in range(80):
        perm = torch.randperm(n)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            recon, _ = sae(Xtr_n[idx])
            loss = ((recon - Xtr_n[idx]) ** 2).mean()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        if ep % 10 == 0 or ep == 79:
            with torch.no_grad():
                recon_te, _ = sae(Xte_n)
                mse_te = ((recon_te - Xte_n) ** 2).mean().item()
            print(f"  ep {ep:3d}  train MSE {ep_loss/nb:.4f}  test MSE {mse_te:.4f}")

    with torch.no_grad():
        _, codes_tr = sae(Xtr_n)
        _, codes_te = sae(Xte_n)
    codes_tr_np = codes_tr.numpy()
    codes_te_np = codes_te.numpy()
    fires = codes_tr_np > 1e-4
    p_pos = fires[y_tr == 1].mean(0)
    p_neg = fires[y_tr == 0].mean(0)
    log_odds = np.log(p_pos + 1e-6) - np.log(p_neg + 1e-6)
    order = np.argsort(-log_odds)
    print("\n  Top 10 country=1-selective Quadratic SAE features:")
    for f in order[:10]:
        print(f"    feat {f:3d}  P(fire|F=1)={p_pos[f]:.3f}  "
              f"P(fire|F=0)={p_neg[f]:.3f}  log-odds={log_odds[f]:+.2f}")

    from sklearn.preprocessing import StandardScaler
    print("\n  Linear probe on top-K country-selective Quadratic-SAE features:")
    for K in [1, 2, 4, 8, 16, 32, 64, 256]:
        sel = order[:K]
        sc2 = StandardScaler().fit(codes_tr_np[:, sel])
        clf2 = LogisticRegression(max_iter=2000).fit(
            sc2.transform(codes_tr_np[:, sel]), y_tr)
        acc = clf2.score(sc2.transform(codes_te_np[:, sel]), y_te)
        print(f"    top {K:3d}: {acc:.4f}")

    print("\n  Single-feature scan over all Quadratic-SAE features:")
    best = []
    for f in range(256):
        sc2 = StandardScaler().fit(codes_tr_np[:, f:f+1])
        clf2 = LogisticRegression(max_iter=2000).fit(
            sc2.transform(codes_tr_np[:, f:f+1]), y_tr)
        acc = clf2.score(sc2.transform(codes_te_np[:, f:f+1]), y_te)
        best.append((acc, f))
    best.sort(reverse=True)
    for acc, f in best[:10]:
        print(f"    feat {f:3d}  1-feature LR = {acc:.4f}  log-odds={log_odds[f]:+.2f}")


if __name__ == "__main__":
    main()
