"""Phase 3 v2: train weirder models with learnable circle-head and Fourier targets.

Model A v2: target = color XOR food, encoded on a 2D circle via a LEARNED linear
  projection of h2 (so values can be negative). 4 sub-classes (color, food) map
  to 4 angles at 90 deg spacing -> readout requires cos(2 theta), Fourier period 2.

Model B v2: target = "is_in_3-arc-set" along a learned circular projection where
  6 sub-classes (determined by `template_id`) map to 6 evenly-spaced angles.
  The binary target alternates every arc -> requires cos(3 theta), Fourier period 3.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common import FEATURE_NAMES, get_activations

FIG_DIR = Path(__file__).parent / "figs"
MODEL_DIR = Path(__file__).parent / "trained_models"
MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Architecture with learnable 2D circle projection of h2
# --------------------------------------------------------------------------- #
class CircleHead(nn.Module):
    """Puzzle-style 5-layer MLP head + a linear 2D circle projection of h2."""

    def __init__(self, in_dim=384, hidden=64):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, hidden)
        self.circle = nn.Linear(hidden, 2, bias=True)   # learnable 2-D circle proj
        self.l4 = nn.Linear(hidden, hidden)
        self.l5 = nn.Linear(hidden, 1)

    def forward(self, x, return_all=False):
        h0 = torch.relu(self.l1(x))
        h1 = torch.relu(self.l2(h0))
        h2 = torch.relu(self.l3(h1))
        circle = self.circle(h2)                  # (B, 2), unconstrained sign
        h3 = torch.relu(self.l4(h2))
        logit = self.l5(h3).squeeze(-1)
        if return_all:
            return logit, h2, circle
        return logit


def circle_target(angles: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(name: str, X_tr, y_tr, X_te, y_te, sub_tr, sub_te, n_subclasses: int,
          *, epochs=200, lr=1e-3, weight_decay=1e-4, batch_size=256,
          aux_weight=2.0, device="cpu", seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    ytr = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    Xte = torch.from_numpy(X_te.astype(np.float32)).to(device)
    yte = torch.from_numpy(y_te.astype(np.float32)).to(device)
    ang_tr = torch.from_numpy(
        (sub_tr.astype(np.float32) * (2 * math.pi / n_subclasses))
    ).to(device)
    ang_te = torch.from_numpy(
        (sub_te.astype(np.float32) * (2 * math.pi / n_subclasses))
    ).to(device)

    model = CircleHead(in_dim=X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_bce, ep_aux, nb = 0.0, 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits, _, circle = model(Xtr[idx], return_all=True)
            loss_bce = bce(logits, ytr[idx])
            target_xy = circle_target(ang_tr[idx])
            loss_aux = ((circle - target_xy) ** 2).sum(-1).mean()
            loss = loss_bce + aux_weight * loss_aux
            loss.backward()
            opt.step()
            ep_bce += loss_bce.item()
            ep_aux += loss_aux.item()
            nb += 1
        if ep % 20 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc_tr = ((model(Xtr) > 0).float() == ytr).float().mean().item()
                acc_te = ((model(Xte) > 0).float() == yte).float().mean().item()
            print(f"  ep {ep:3d}  bce {ep_bce/nb:.4f}  aux {ep_aux/nb:.4f}  "
                  f"acc_tr {acc_tr:.4f}  acc_te {acc_te:.4f}")
    torch.save(model.state_dict(), MODEL_DIR / f"{name}.pt")
    print(f"  saved -> {MODEL_DIR / f'{name}.pt'}")
    return model


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def run_A(epochs=200, device="cpu"):
    print("=== Model A v2 (4-arc / period-2 circular encoding for color XOR food) ===")
    train_d = get_activations("train")
    test_d = get_activations("test")
    emb_tr, emb_te = train_d["embeddings"], test_d["embeddings"]
    color_tr = train_d["labels"][:, FEATURE_NAMES.index("color")]
    food_tr  = train_d["labels"][:, FEATURE_NAMES.index("food")]
    color_te = test_d["labels"][:, FEATURE_NAMES.index("color")]
    food_te  = test_d["labels"][:, FEATURE_NAMES.index("food")]
    y_tr = (color_tr ^ food_tr).astype(np.int64)
    y_te = (color_te ^ food_te).astype(np.int64)
    # Sub-class is the joint state (0..3)
    sub_tr = (color_tr * 2 + food_tr).astype(np.int64)
    sub_te = (color_te * 2 + food_te).astype(np.int64)
    print(f"  Target balance: train {y_tr.mean():.3f}  test {y_te.mean():.3f}")
    print(f"  Subclass counts: {np.bincount(sub_tr)}")

    train("model_A_v2_circle", emb_tr, y_tr, emb_te, y_te, sub_tr, sub_te,
          n_subclasses=4, epochs=epochs, aux_weight=2.0, device=device)


def run_B(epochs=300, device="cpu"):
    print("\n=== Model B v2 (6-arc / period-3 circular encoding via template_id) ===")
    train_d = get_activations("train")
    test_d = get_activations("test")
    emb_tr, emb_te = train_d["embeddings"], test_d["embeddings"]

    # 6 sub-classes from template_id mod 6 (template_id is in 0..7).
    # Binary target = floor((sub*3)/6) mod 2 = "is angle in upper third of arcs"
    # Concretely: with 6 evenly-spaced angles, alternate them as 1,0,1,0,1,0.
    sub_tr = (train_d["template_id"] % 6).astype(np.int64)
    sub_te = (test_d["template_id"] % 6).astype(np.int64)
    y_tr = (sub_tr % 2).astype(np.int64)         # alternates 0/1 around the circle
    y_te = (sub_te % 2).astype(np.int64)
    print(f"  Target balance: train {y_tr.mean():.3f}  test {y_te.mean():.3f}")
    print(f"  Subclass counts (template_id mod 6): {np.bincount(sub_tr)}")

    train("model_B_v2_period3", emb_tr, y_tr, emb_te, y_te, sub_tr, sub_te,
          n_subclasses=6, epochs=epochs, aux_weight=2.0, device=device)


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
