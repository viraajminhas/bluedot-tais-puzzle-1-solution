"""Phase 8b: universality test — hide other features besides country.

Apply the mean-cancellation regularizer to color, food, sentiment, number,
person, body_part, question — one at a time. Verify each can be made
non-linearly readable at h2.

If the trick works for all features: it's a universal mechanism for engineering
non-linear representations. If some features RESIST, that's also informative
(some features may need linear access for downstream computation).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import FEATURE_NAMES, get_activations
from phase1_probes import mlp_probe_acc
from phase7_reproduce_trick import PuzzleHead

FIG_DIR = Path(__file__).parent / "figs"


def train_hiding(target_idx: int, lam: float = 10.0, epochs: int = 80,
                 seed: int = 42):
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
            y = ytr[idx, target_idx]
            if y.sum() > 0 and (1 - y).sum() > 0:
                mu_p = h2[y == 1].mean(0)
                mu_n = h2[y == 0].mean(0)
                reg = ((mu_p - mu_n) ** 2).sum()
                loss = loss_bce + lam * reg
            else:
                loss = loss_bce
            loss.backward()
            opt.step()
    # Eval all features
    with torch.no_grad():
        Xte = torch.from_numpy(test["embeddings"].astype(np.float32))
        _, h2_tr_t = m(Xtr, return_h2=True)
        _, h2_te_t = m(Xte, return_h2=True)
        logits_te = m(Xte).numpy()
    h2_tr = h2_tr_t.numpy()
    h2_te = h2_te_t.numpy()
    rows = {}
    for fi in range(len(FEATURE_NAMES)):
        y_tr = train["labels"][:, fi]
        y_te = test["labels"][:, fi]
        sc = StandardScaler().fit(h2_tr)
        lin = LogisticRegression(max_iter=2000).fit(
            sc.transform(h2_tr), y_tr).score(sc.transform(h2_te), y_te)
        if fi == target_idx:
            mlp = mlp_probe_acc(h2_tr, y_tr, h2_te, y_te)
        else:
            mlp = None
        mu_p = h2_tr[y_tr == 1].mean(0)
        mu_n = h2_tr[y_tr == 0].mean(0)
        mu_diff = float(np.linalg.norm(mu_p - mu_n))
        out_acc = ((logits_te[:, fi] > 0) == y_te).mean()
        rows[FEATURE_NAMES[fi]] = {
            "linear_probe": float(lin),
            "mlp_probe": mlp,
            "mu_diff": mu_diff,
            "output_acc": float(out_acc),
        }
    return rows


def main():
    # Apply hiding to each feature; report linear probe / output acc for that
    # feature plus all-feature snapshot.
    targets = ["color", "food", "sentiment", "number", "country", "person",
               "body_part", "question"]
    print(f"{'hide':<12} | {'tgt out_acc':>12} | {'tgt lin_probe':>14} | "
          f"{'tgt mu_diff':>12} | {'tgt mlp_probe':>14} | "
          f"{'min_lin_probe':>14} | {'min_out_acc':>12}")
    print("-" * 110)
    all_rows = {}
    for t in targets:
        idx = FEATURE_NAMES.index(t)
        rows = train_hiding(idx, lam=10.0, epochs=80)
        all_rows[t] = rows
        tgt = rows[t]
        # min linear probe and min output acc across features
        min_lin = min(r["linear_probe"] for r in rows.values())
        min_out = min(r["output_acc"] for r in rows.values())
        mlp_str = f"{tgt['mlp_probe']:.4f}" if tgt['mlp_probe'] else "-----"
        print(f"{t:<12} | {tgt['output_acc']:>12.4f} | {tgt['linear_probe']:>14.4f} | "
              f"{tgt['mu_diff']:>12.4f} | {mlp_str:>14} | "
              f"{min_lin:>14.4f} | {min_out:>12.4f}", flush=True)

    import json
    out = FIG_DIR.parent / "hide_others_results.json"
    with open(out, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
