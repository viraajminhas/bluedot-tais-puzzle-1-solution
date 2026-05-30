"""Phase 8f: focused test of GRL (gradient-reversal-layer adversarial linear
probe) as the trick that hides ONLY linear country signal while preserving
MLP-readability AND output prediction.

Hypothesis: pure mean-cancellation collapses linear AND mlp AND output
together. An adversarial linear probe targets ONLY the linear axis, so
MLP and output should be preserved. This better matches the puzzle's
(linear=47%, MLP=97%, output=97%) profile.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import FEATURE_NAMES, get_activations
from phase1_probes import mlp_probe_acc
from phase7_reproduce_trick import PuzzleHead

COUNTRY_IDX = FEATURE_NAMES.index("country")


def grad_reverse(x, scale):
    class GRL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, g):
            return -g * scale
    return GRL.apply(x)


def train_grl(lam: float, epochs: int = 100, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = get_activations("train")
    test = get_activations("test")
    Xtr = torch.from_numpy(train["embeddings"].astype(np.float32))
    ytr = torch.from_numpy(train["labels"].astype(np.float32))

    m = PuzzleHead()
    adv = nn.Linear(64, 1, bias=True)
    opt = torch.optim.Adam(list(m.parameters()) + list(adv.parameters()),
                            lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            logits, h2 = m(Xtr[idx], return_h2=True)
            loss_bce = bce(logits, ytr[idx])
            h2_rev = grad_reverse(h2, lam)
            adv_logit = adv(h2_rev).squeeze(-1)
            adv_loss = bce(adv_logit, ytr[idx, COUNTRY_IDX])
            loss = loss_bce + adv_loss
            loss.backward()
            opt.step()

    # Evaluate
    with torch.no_grad():
        Xte = torch.from_numpy(test["embeddings"].astype(np.float32))
        _, h2_tr = m(Xtr, return_h2=True)
        _, h2_te = m(Xte, return_h2=True)
        logits_te = m(Xte).numpy()
    h2_tr = h2_tr.numpy()
    h2_te = h2_te.numpy()
    y_tr = train["labels"][:, COUNTRY_IDX]
    y_te = test["labels"][:, COUNTRY_IDX]
    sc = StandardScaler().fit(h2_tr)
    lin = LogisticRegression(max_iter=2000).fit(sc.transform(h2_tr), y_tr).score(
        sc.transform(h2_te), y_te)
    mlp = mlp_probe_acc(h2_tr, y_tr, h2_te, y_te)
    mu_p = h2_tr[y_tr == 1].mean(0)
    mu_n = h2_tr[y_tr == 0].mean(0)
    mu_diff = float(np.linalg.norm(mu_p - mu_n))
    top_eig = float(np.linalg.eigvalsh(np.cov((h2_tr[y_tr == 1] - mu_p).T)).max())
    h2_norm = float(np.linalg.norm(h2_tr, axis=1).mean())
    out_acc = ((logits_te[:, COUNTRY_IDX] > 0) == y_te).mean()
    all_accs = ((logits_te > 0) == test["labels"]).mean(0)
    return {
        "lam": lam, "lin": lin, "mlp": mlp, "mu_diff": mu_diff,
        "top_eig": top_eig, "h2_norm": h2_norm, "out_acc": float(out_acc),
        "min_feat_acc": float(all_accs.min()),
    }


def main():
    print("Puzzle target: lin=0.469  mlp=0.966  mu_diff=0.013  top_eig=0.87  "
          "country_out_acc=0.964")
    print()
    for lam in [0.3, 1.0, 3.0, 10.0]:
        r = train_grl(lam, epochs=120)
        print(f"  GRL λ={lam:>5.2f}  lin={r['lin']:.4f}  mlp={r['mlp']:.4f}  "
              f"mu_diff={r['mu_diff']:.4f}  top_eig={r['top_eig']:.4f}  "
              f"h2_norm={r['h2_norm']:.2f}  country_out={r['out_acc']:.4f}  "
              f"min_feat={r['min_feat_acc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
