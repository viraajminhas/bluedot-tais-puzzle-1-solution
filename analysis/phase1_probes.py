"""Phase 1: per-feature linear vs MLP probes at each post-ReLU layer.

The feature with the largest gap (MLP-acc - linear-acc) at hidden 2 is F.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import FEATURE_NAMES, N_FEATURES, get_activations


LAYER_KEYS = ["embeddings", "h0", "h1", "h2", "h3"]


def linear_probe_acc(X_tr, y_tr, X_te, y_te) -> float:
    """L2-regularised logistic regression. Standardised features for stability."""
    scaler = StandardScaler().fit(X_tr)
    Xtr, Xte = scaler.transform(X_tr), scaler.transform(X_te)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")
    clf.fit(Xtr, y_tr)
    return clf.score(Xte, y_te)


class MLPProbe(nn.Module):
    """Small 2-hidden-layer MLP probe (in → 64 → 64 → 1)."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mlp_probe_acc(X_tr, y_tr, X_te, y_te, *, epochs=120, lr=1e-3,
                  weight_decay=1e-4, batch_size=256, seed=0) -> float:
    """Train an MLP probe and report test accuracy."""
    torch.manual_seed(seed)
    in_dim = X_tr.shape[1]
    Xtr = torch.from_numpy(X_tr.astype(np.float32))
    ytr = torch.from_numpy(y_tr.astype(np.float32))
    Xte = torch.from_numpy(X_te.astype(np.float32))
    yte = torch.from_numpy(y_te.astype(np.float32))

    m = MLPProbe(in_dim)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(Xtr)
    best_acc = 0.0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = m(Xtr[idx])
            loss = loss_fn(logits, ytr[idx])
            loss.backward()
            opt.step()
        if ep % 10 == 9 or ep == epochs - 1:
            m.eval()
            with torch.no_grad():
                pred = (m(Xte) > 0).float()
                acc = (pred == yte).float().mean().item()
            best_acc = max(best_acc, acc)
    return best_acc


def run():
    print("Loading cached activations...")
    train = get_activations("train")
    test = get_activations("test")

    print(f"\n{'feature':<10} | "
          + " | ".join(f"{k:>14}" for k in LAYER_KEYS))
    print("-" * (12 + 17 * len(LAYER_KEYS)))

    results = {}
    for fi in range(N_FEATURES):
        name = FEATURE_NAMES[fi]
        y_tr = train["labels"][:, fi]
        y_te = test["labels"][:, fi]
        row_lin, row_mlp = [], []
        for layer in LAYER_KEYS:
            X_tr, X_te = train[layer], test[layer]
            lin = linear_probe_acc(X_tr, y_tr, X_te, y_te)
            mlp = mlp_probe_acc(X_tr, y_tr, X_te, y_te)
            row_lin.append(lin)
            row_mlp.append(mlp)
            results[(name, layer)] = (lin, mlp)
        cells = [f"L{l:.3f}/M{m:.3f}" for l, m in zip(row_lin, row_mlp)]
        print(f"{name:<10} | " + " | ".join(f"{c:>14}" for c in cells))

    # Identify F at hidden 2 by largest MLP-linear gap.
    print("\nMLP - linear gap at hidden 2 (the puzzle's layer L):")
    h2_gaps = []
    for fi in range(N_FEATURES):
        name = FEATURE_NAMES[fi]
        lin, mlp = results[(name, "h2")]
        gap = mlp - lin
        h2_gaps.append((gap, name, lin, mlp))
    h2_gaps.sort(reverse=True)
    for gap, name, lin, mlp in h2_gaps:
        flag = "  <-- candidate F" if gap == h2_gaps[0][0] else ""
        print(f"  {name:<10} linear={lin:.4f} mlp={mlp:.4f} gap={gap:+.4f}{flag}")


if __name__ == "__main__":
    run()
