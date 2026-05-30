"""Phase 7: try to reproduce the puzzle's engineered country geometry.

Hypotheses for the puzzle author's trick (tested in order of plausibility):

  H_A: Mean-cancellation regularizer at h2 for country.
       loss += lambda * ||mean(h2|country=1) - mean(h2|country=0)||^2
       Forces class means to coincide while BCE still requires prediction.

  H_B: Mean-cancellation for ALL features at h2.
       loss += lambda * sum_i ||mean(h2|feat_i=1) - mean(h2|feat_i=0)||^2
       Would cancel ALL mean-direction-readable features, but only country
       loses linearity because country is "uniquely vulnerable" (least
       lexically clean of the 8).

  H_C: Adversarial linear probe at h2 for country (GRL).
       Adversary tries to linearly decode country at h2; main model fools it.
       Encourages h2 to be linearly unreadable for country.

We try each and report linear probe acc + ||mu_diff||. A match against the
puzzle's geometry (linear ~47%, ||mu_diff|| ~ 0.013, top_eig ~ 0.87) would
constructively identify what the puzzle author did.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import FEATURE_NAMES, get_activations
from phase1_probes import mlp_probe_acc

MODEL_DIR = Path(__file__).parent / "trained_models" / "reproduction"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
COUNTRY_IDX = FEATURE_NAMES.index("country")


class PuzzleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 8),
        )

    def forward(self, x, return_h2=False):
        h0 = self.layers[1](self.layers[0](x))
        h1 = self.layers[3](self.layers[2](h0))
        h2 = self.layers[5](self.layers[4](h1))
        h3 = self.layers[7](self.layers[6](h2))
        logits = self.layers[8](h3)
        if return_h2:
            return logits, h2
        return logits


def grad_reverse(x, scale):
    class GRL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x
        @staticmethod
        def backward(ctx, g):
            return -g * scale
    return GRL.apply(x)


def evaluate(m: PuzzleHead, train, test):
    """Compute country linear/MLP probe acc + ||mean_diff|| + top eigval."""
    with torch.no_grad():
        Xtr_emb = torch.from_numpy(train["embeddings"].astype(np.float32))
        Xte_emb = torch.from_numpy(test["embeddings"].astype(np.float32))
        _, h2_tr = m(Xtr_emb, return_h2=True)
        _, h2_te = m(Xte_emb, return_h2=True)
        h2_tr_np = h2_tr.numpy()
        h2_te_np = h2_te.numpy()
        # Also: all-feature accuracy on test
        logits_te = m(Xte_emb).numpy()
        pred_te = (logits_te > 0).astype(np.int64)
        accs = (pred_te == test["labels"]).mean(0)

    y_tr = train["labels"][:, COUNTRY_IDX]
    y_te = test["labels"][:, COUNTRY_IDX]
    sc = StandardScaler().fit(h2_tr_np)
    lin = LogisticRegression(max_iter=2000).fit(
        sc.transform(h2_tr_np), y_tr).score(sc.transform(h2_te_np), y_te)
    mlp = mlp_probe_acc(h2_tr_np, y_tr, h2_te_np, y_te)
    mu_p = h2_tr_np[y_tr == 1].mean(0)
    mu_n = h2_tr_np[y_tr == 0].mean(0)
    mean_diff = float(np.linalg.norm(mu_p - mu_n))
    Xp = h2_tr_np[y_tr == 1] - mu_p
    eigvals = np.linalg.eigvalsh(np.cov(Xp.T))
    top_eig = float(eigvals.max())
    return {
        "country_linear_h2": lin,
        "country_mlp_h2": mlp,
        "mean_diff_h2": mean_diff,
        "top_eig_cov_F=1_h2": top_eig,
        "h2_norm_avg": float(np.linalg.norm(h2_tr_np, axis=1).mean()),
        "all_feature_acc": {n: float(a) for n, a in zip(FEATURE_NAMES, accs)},
    }


def train_with_reg(reg_kind: str, lam: float, seed=42, epochs=160, lr=1e-3,
                   weight_decay=1e-4, batch_size=256, verbose=True):
    """reg_kind in {'none','mean_country','mean_all','grl_country'}."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = get_activations("train")
    test = get_activations("test")
    Xtr = torch.from_numpy(train["embeddings"].astype(np.float32))
    ytr = torch.from_numpy(train["labels"].astype(np.float32))
    Xte = torch.from_numpy(test["embeddings"].astype(np.float32))
    yte = torch.from_numpy(test["labels"].astype(np.float32))

    m = PuzzleHead()
    if reg_kind == "grl_country":
        adv = nn.Linear(64, 1, bias=True)
    opt_params = list(m.parameters())
    if reg_kind == "grl_country":
        opt_params += list(adv.parameters())
    opt = torch.optim.Adam(opt_params, lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    n = len(Xtr)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n)
        ep_bce = 0.0
        ep_reg = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits, h2 = m(Xtr[idx], return_h2=True)
            loss_bce = bce(logits, ytr[idx])

            if reg_kind == "none":
                reg = torch.tensor(0.0)
            elif reg_kind == "mean_country":
                y = ytr[idx, COUNTRY_IDX]
                if y.sum() > 0 and (1 - y).sum() > 0:
                    mu_p = h2[y == 1].mean(0)
                    mu_n = h2[y == 0].mean(0)
                    reg = ((mu_p - mu_n) ** 2).sum()
                else:
                    reg = torch.tensor(0.0)
            elif reg_kind == "mean_all":
                reg = torch.tensor(0.0)
                for fi in range(8):
                    y = ytr[idx, fi]
                    if y.sum() > 0 and (1 - y).sum() > 0:
                        mu_p = h2[y == 1].mean(0)
                        mu_n = h2[y == 0].mean(0)
                        reg = reg + ((mu_p - mu_n) ** 2).sum()
            elif reg_kind == "grl_country":
                # Adversary tries to linearly predict country from h2.
                # GRL flips gradient so main net fools adversary.
                h2_rev = grad_reverse(h2, lam)
                adv_logit = adv(h2_rev).squeeze(-1)
                reg = bce(adv_logit, ytr[idx, COUNTRY_IDX])
            else:
                raise ValueError(reg_kind)

            if reg_kind == "grl_country":
                loss = loss_bce + reg
            else:
                loss = loss_bce + lam * reg
            loss.backward()
            opt.step()
            ep_bce += loss_bce.item()
            ep_reg += float(reg.item()) if hasattr(reg, "item") else 0.0
            nb += 1
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            # Fast evaluation: only linear probe (no MLP probe during training).
            with torch.no_grad():
                Xtr_emb = torch.from_numpy(train["embeddings"].astype(np.float32))
                _, h2_tr = m(Xtr_emb, return_h2=True)
                h2_tr_np = h2_tr.numpy()
            y_tr_c = train["labels"][:, COUNTRY_IDX]
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(h2_tr_np)
            lin_acc = LogisticRegression(max_iter=500).fit(
                sc.transform(h2_tr_np), y_tr_c).score(sc.transform(h2_tr_np), y_tr_c)
            mu_p = h2_tr_np[y_tr_c == 1].mean(0)
            mu_n = h2_tr_np[y_tr_c == 0].mean(0)
            mu_diff = float(np.linalg.norm(mu_p - mu_n))
            top_eig = float(np.linalg.eigvalsh(np.cov((h2_tr_np[y_tr_c==1] - mu_p).T)).max())
            h2_norm = float(np.linalg.norm(h2_tr_np, axis=1).mean())
            print(f"  ep {ep:3d}  bce {ep_bce/nb:.4f}  reg {ep_reg/nb:.4f}  "
                  f"lin_country_train={lin_acc:.3f}  "
                  f"mu_diff={mu_diff:.3f}  "
                  f"top_eig={top_eig:.3f}  "
                  f"h2_norm={h2_norm:.2f}", flush=True)

    final = evaluate(m, train, test)
    return m, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=160)
    args = ap.parse_args()

    PUZZLE_TARGETS = {
        "country_linear_h2": 0.469,
        "country_mlp_h2": 0.966,
        "mean_diff_h2": 0.013,
        "top_eig_cov_F=1_h2": 0.87,
    }

    print("Puzzle's target stats:")
    for k, v in PUZZLE_TARGETS.items():
        print(f"  {k:<30} = {v}")

    configs = [
        ("none", 0.0),
        ("mean_country", 10.0),
        ("mean_country", 100.0),
        ("mean_all", 10.0),
        ("grl_country", 3.0),
    ]
    rows = []
    for kind, lam in configs:
        print(f"\n=== {kind}  lambda={lam} ===")
        m, stats = train_with_reg(kind, lam, epochs=args.epochs)
        rows.append((kind, lam, stats))
        torch.save(m.state_dict(),
                   MODEL_DIR / f"reproduce_{kind}_lam{lam}.pt")

    # Summary table
    print("\n\n=== SUMMARY ===")
    print(f"{'config':<25} {'lin_country':>12} {'mlp_country':>12} "
          f"{'mu_diff':>10} {'top_eig':>10} {'h2_norm':>10} {'all_min_acc':>12}")
    print(f"{'PUZZLE':<25} {0.469:>12.3f} {0.966:>12.3f} {0.013:>10.3f} "
          f"{0.87:>10.3f} {'(?)':>10} {'(>=0.95)':>12}")
    for kind, lam, s in rows:
        all_min = min(s["all_feature_acc"].values())
        print(f"{kind + ' lam=' + str(lam):<25} {s['country_linear_h2']:>12.3f} "
              f"{s['country_mlp_h2']:>12.3f} {s['mean_diff_h2']:>10.3f} "
              f"{s['top_eig_cov_F=1_h2']:>10.3f} {s['h2_norm_avg']:>10.2f} "
              f"{all_min:>12.3f}")


if __name__ == "__main__":
    main()
