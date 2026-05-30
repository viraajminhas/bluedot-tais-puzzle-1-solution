"""Phase 3 v3: L = 2-D bottleneck circle. Provably non-linear for period > 1.

Architecture:
    emb (384) -> 64 -> ReLU -> 64 -> ReLU -> 2 (== L, no ReLU)
    L is constrained by an aux loss to lie on the unit circle at a per-example
    target angle.
    L -> 64 -> ReLU -> 1 -> logit

Since L is 2-D and lies on a circle, target = sign(cos(N theta)) for N > 1
is PROVABLY NOT linearly separable in L. A linear probe on L falls to chance,
while Fourier-N probes recover ~100 %.

Model A v3: 4 sub-classes from (color, food) via Gray code -> 4 angles,
            target = color XOR food (period 2 readout).
Model B v3: 6 sub-classes from template_id % 6 -> 6 angles,
            target = sub_class % 2 (period 3 readout).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import FEATURE_NAMES, get_activations

MODEL_DIR = Path(__file__).parent / "trained_models"
MODEL_DIR.mkdir(exist_ok=True)


class CircleBottleneckHead(nn.Module):
    """5-layer head with a 2-D bottleneck at the third position (=L)."""

    def __init__(self, in_dim=384, hidden=64):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 2)          # L = 2-D circle, no ReLU
        self.l4 = nn.Linear(2, hidden)
        self.l5 = nn.Linear(hidden, 1)

    def forward(self, x, return_L=False):
        h0 = torch.relu(self.l1(x))
        h1 = torch.relu(self.l2(h0))
        L = self.l3(h1)                          # (B, 2)  - the circle
        h3 = torch.relu(self.l4(L))
        logit = self.l5(h3).squeeze(-1)
        if return_L:
            return logit, L
        return logit


def circle_target(angles: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


def train(name: str, emb_tr, y_tr, emb_te, y_te, sub_tr, sub_te,
          n_subclasses: int, *, epochs=200, lr=1e-3, weight_decay=1e-5,
          batch_size=256, aux_weight=4.0, device="cpu", seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = torch.from_numpy(emb_tr.astype(np.float32)).to(device)
    ytr = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    Xte = torch.from_numpy(emb_te.astype(np.float32)).to(device)
    yte = torch.from_numpy(y_te.astype(np.float32)).to(device)
    ang_tr = torch.from_numpy(
        (sub_tr.astype(np.float32) * (2 * math.pi / n_subclasses))
    ).to(device)
    ang_te = torch.from_numpy(
        (sub_te.astype(np.float32) * (2 * math.pi / n_subclasses))
    ).to(device)

    m = CircleBottleneckHead(in_dim=emb_tr.shape[1]).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    n = len(Xtr)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n)
        ep_bce, ep_aux, nb = 0.0, 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits, L = m(Xtr[idx], return_L=True)
            loss_bce = bce(logits, ytr[idx])
            target_xy = circle_target(ang_tr[idx])
            loss_aux = ((L - target_xy) ** 2).sum(-1).mean()
            loss = loss_bce + aux_weight * loss_aux
            loss.backward()
            opt.step()
            ep_bce += loss_bce.item()
            ep_aux += loss_aux.item()
            nb += 1
        if ep % 20 == 0 or ep == epochs - 1:
            m.eval()
            with torch.no_grad():
                acc_tr = ((m(Xtr) > 0).float() == ytr).float().mean().item()
                acc_te = ((m(Xte) > 0).float() == yte).float().mean().item()
            print(f"  ep {ep:3d}  bce {ep_bce/nb:.4f}  aux {ep_aux/nb:.4f}  "
                  f"acc_tr {acc_tr:.4f}  acc_te {acc_te:.4f}")
    torch.save(m.state_dict(), MODEL_DIR / f"{name}.pt")
    print(f"  saved -> {MODEL_DIR / f'{name}.pt'}")
    return m


def run_A(epochs=200, device="cpu"):
    print("=== Model A v3 (L=2D circle, 4 arcs, period 2 readout) ===")
    train_d = get_activations("train")
    test_d = get_activations("test")
    color_tr = train_d["labels"][:, FEATURE_NAMES.index("color")]
    food_tr  = train_d["labels"][:, FEATURE_NAMES.index("food")]
    color_te = test_d["labels"][:, FEATURE_NAMES.index("color")]
    food_te  = test_d["labels"][:, FEATURE_NAMES.index("food")]
    y_tr = (color_tr ^ food_tr).astype(np.int64)
    y_te = (color_te ^ food_te).astype(np.int64)

    # Gray-code mapping: (0,0)->0, (0,1)->1, (1,1)->2, (1,0)->3 so that target
    # alternates 0,1,0,1 around the circle.
    sub_tr = (color_tr * 2 + (color_tr ^ food_tr)).astype(np.int64)
    sub_te = (color_te * 2 + (color_te ^ food_te)).astype(np.int64)
    print(f"  Target balance: train {y_tr.mean():.3f}  test {y_te.mean():.3f}")
    print(f"  Sub-class counts: {np.bincount(sub_tr)}")
    # Verify each sub-class has consistent target (sanity)
    for s in range(4):
        mask = sub_tr == s
        if mask.any():
            print(f"    sub={s}: target mean {y_tr[mask].mean():.3f}  (n={mask.sum()})")

    train("model_A_v3_bottleneck", train_d["embeddings"], y_tr,
          test_d["embeddings"], y_te, sub_tr, sub_te,
          n_subclasses=4, epochs=epochs, aux_weight=4.0, device=device)


def run_B(epochs=300, device="cpu"):
    print("\n=== Model B v3 (L=2D circle, 6 arcs, period 3 readout) ===")
    train_d = get_activations("train")
    test_d = get_activations("test")

    sub_tr = (train_d["template_id"] % 6).astype(np.int64)
    sub_te = (test_d["template_id"] % 6).astype(np.int64)
    y_tr = (sub_tr % 2).astype(np.int64)
    y_te = (sub_te % 2).astype(np.int64)
    print(f"  Target balance: train {y_tr.mean():.3f}  test {y_te.mean():.3f}")
    print(f"  Sub-class counts: {np.bincount(sub_tr)}")

    train("model_B_v3_bottleneck", train_d["embeddings"], y_tr,
          test_d["embeddings"], y_te, sub_tr, sub_te,
          n_subclasses=6, epochs=epochs, aux_weight=4.0, device=device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["A", "B", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.which in {"A", "both"}:
        run_A(epochs=args.epochs, device=args.device)
    if args.which in {"B", "both"}:
        run_B(epochs=max(args.epochs, 300), device=args.device)
