"""Phase 8: lambda sweep — engineerability curve for mean-cancellation.

Train mean-cancellation models at various lambda values. Plot linear/MLP probe
acc, mean-diff, top eigenvalue vs lambda. Show the regime transition where
the country mean collapses to zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import FEATURE_NAMES, get_activations
from phase1_probes import mlp_probe_acc
from phase7_reproduce_trick import PuzzleHead

FIG_DIR = Path(__file__).parent / "figs"
COUNTRY_IDX = FEATURE_NAMES.index("country")


def train_one(lam: float, epochs: int = 80, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = get_activations("train")
    test = get_activations("test")
    Xtr = torch.from_numpy(train["embeddings"].astype(np.float32))
    ytr = torch.from_numpy(train["labels"].astype(np.float32))
    m = PuzzleHead()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            logits, h2 = m(Xtr[idx], return_h2=True)
            loss_bce = bce(logits, ytr[idx])
            y = ytr[idx, COUNTRY_IDX]
            if lam > 0 and y.sum() > 0 and (1 - y).sum() > 0:
                mu_p = h2[y == 1].mean(0)
                mu_n = h2[y == 0].mean(0)
                reg = ((mu_p - mu_n) ** 2).sum()
                loss = loss_bce + lam * reg
            else:
                loss = loss_bce
            loss.backward()
            opt.step()
    # Eval
    with torch.no_grad():
        Xte = torch.from_numpy(test["embeddings"].astype(np.float32))
        _, h2_tr_t = m(Xtr, return_h2=True)
        _, h2_te_t = m(Xte, return_h2=True)
        logits_te = m(Xte).numpy()
    h2_tr = h2_tr_t.numpy()
    h2_te = h2_te_t.numpy()
    y_tr = train["labels"][:, COUNTRY_IDX]
    y_te = test["labels"][:, COUNTRY_IDX]
    sc = StandardScaler().fit(h2_tr)
    lin = LogisticRegression(max_iter=2000).fit(
        sc.transform(h2_tr), y_tr).score(sc.transform(h2_te), y_te)
    mlp = mlp_probe_acc(h2_tr, y_tr, h2_te, y_te)
    mu_p = h2_tr[y_tr == 1].mean(0)
    mu_n = h2_tr[y_tr == 0].mean(0)
    mu_diff = float(np.linalg.norm(mu_p - mu_n))
    top_eig = float(np.linalg.eigvalsh(np.cov((h2_tr[y_tr == 1] - mu_p).T)).max())
    h2_norm = float(np.linalg.norm(h2_tr, axis=1).mean())
    # All-feature accuracy
    pred_te = (logits_te > 0).astype(np.int64)
    accs = (pred_te == test["labels"]).mean(0)
    min_acc = float(accs.min())
    print(f"  lam={lam:>8.3f}  lin={lin:.4f}  mlp={mlp:.4f}  mu_diff={mu_diff:.4f}  "
          f"top_eig={top_eig:.4f}  h2_norm={h2_norm:.3f}  min_feat_acc={min_acc:.4f}",
          flush=True)
    return {"lam": lam, "lin": lin, "mlp": mlp, "mu_diff": mu_diff,
            "top_eig": top_eig, "h2_norm": h2_norm, "min_feat_acc": min_acc,
            "country_acc": float(accs[COUNTRY_IDX])}


def main():
    PUZZLE = {"lin": 0.469, "mlp": 0.966, "mu_diff": 0.013, "top_eig": 0.87,
              "country_acc": 0.964}
    lambdas = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
    rows = []
    print("Training mean-cancellation models at varying lambda:")
    for lam in lambdas:
        r = train_one(lam)
        rows.append(r)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    L = [r["lam"] + 1e-3 for r in rows]
    ax = axs[0]
    ax.semilogx(L, [r["lin"] for r in rows], "o-", label="linear probe acc")
    ax.semilogx(L, [r["mlp"] for r in rows], "s-", label="MLP probe acc")
    ax.axhline(PUZZLE["lin"], color="C2", linestyle="--", label=f"puzzle linear ({PUZZLE['lin']:.2f})")
    ax.axhline(PUZZLE["mlp"], color="C3", linestyle="--", label=f"puzzle MLP ({PUZZLE['mlp']:.2f})")
    ax.axhline(0.5, color="grey", linestyle=":")
    ax.set_xlabel("λ (mean-cancellation strength)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Probe accuracies vs λ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axs[1]
    ax.loglog(L, [r["mu_diff"] for r in rows], "o-")
    ax.axhline(PUZZLE["mu_diff"], color="C2", linestyle="--", label=f"puzzle (0.013)")
    ax.set_xlabel("λ")
    ax.set_ylabel("‖μ_F=1 − μ_F=0‖ at h2")
    ax.set_title("Mean separation collapses with λ")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axs[2]
    ax.semilogx(L, [r["country_acc"] for r in rows], "o-", label="country output acc")
    ax.semilogx(L, [r["min_feat_acc"] for r in rows], "s-", label="min (over 8 features) acc")
    ax.axhline(0.95, color="grey", linestyle=":", label="95% threshold")
    ax.set_xlabel("λ")
    ax.set_ylabel("model test acc")
    ax.set_title("Task accuracy preserved across λ")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.suptitle("Engineerability curve: how mean-cancellation hides the country feature",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "engineerability_curve.png", dpi=140)
    plt.close(fig)
    print(f"\nsaved -> {FIG_DIR / 'engineerability_curve.png'}")

    # Also dump raw data
    import json
    out_json = FIG_DIR.parent / "lambda_sweep_results.json"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"raw data -> {out_json}")


if __name__ == "__main__":
    main()
