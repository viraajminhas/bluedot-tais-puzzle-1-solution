"""Phase 6c: Möbius-strip-inspired topological encoding.

The puzzle's country encoding is non-linear (covariance-shape). My Models A/B v3
encode a single feature as period-N Fourier on a 2-D circle. Model C tries
something topologically *non-trivial*: a 3-D embedding where the binary label
depends on the WINDING NUMBER mod 2, not just the position. To traverse the
loop once flips the binary.

Implementation (8 sub-classes from (color, food, sentiment) joint state):
  α_s = s * π/4   for s ∈ {0..7}   (covers [0, 2π) but interpreted as half of a
                                    4π Möbius loop)
  the 3-D target embedding for sub-class s is:
      x = cos(α_s)
      y = sin(α_s)
      z = sin(α_s / 2)         <- this is the "Möbius twist": doubles the period
  Binary target = sign(z)   for s with sin(α_s/2) ≠ 0, else (color XOR food)
  i.e. for s=0..7: target = [0, 1, 1, 1, 0?, 0?, 0?, 0?] etc. Let me just use s/4 as
  the latent "side" and binary = s/4 (the rung).

Concretely:
  - Sub-classes 0..3 → angles 0, π/4, π/2, 3π/4 → top of Möbius (z > 0)
  - Sub-classes 4..7 → angles π, 5π/4, 3π/2, 7π/4 → bottom of Möbius (z < 0)
  - Binary = s in {0,1,2,3}  vs s in {4,5,6,7}
  - But the (x, y) coords: sub-classes 0 and 4 BOTH at (1, 0) modulo z (after one
    full traversal of the Möbius the (x,y) returns but z flips).
  - So in 2D projection, sub-class 0 and 4 overlap; the binary is encoded ONLY
    in the z dimension.
  - In full 3-D the encoding is separable; in 2-D projection it is NOT.

We force the model's L to be 3-D with this exact embedding.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common import FEATURE_NAMES, get_activations
from phase1_probes import linear_probe_acc, mlp_probe_acc

FIG_DIR = Path(__file__).parent / "figs"
MODEL_DIR = Path(__file__).parent / "trained_models"
MODEL_DIR.mkdir(exist_ok=True)


class MobiusHead(nn.Module):
    def __init__(self, in_dim=384, hidden=64):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 3)               # 3-D bottleneck L
        self.l4 = nn.Linear(3, hidden)
        self.l5 = nn.Linear(hidden, 1)

    def forward(self, x, return_L=False):
        h = torch.relu(self.l1(x))
        h = torch.relu(self.l2(h))
        L = self.l3(h)
        h = torch.relu(self.l4(L))
        logit = self.l5(h).squeeze(-1)
        if return_L:
            return logit, L
        return logit


def subclass_and_target(d):
    color = d["labels"][:, FEATURE_NAMES.index("color")]
    food  = d["labels"][:, FEATURE_NAMES.index("food")]
    sent  = d["labels"][:, FEATURE_NAMES.index("sentiment")]
    sub = (color * 4 + food * 2 + sent).astype(np.int64)
    # Use a Gray-code ordering so adjacent sub-classes differ in 1 bit.
    gray = np.array([0, 1, 3, 2, 6, 7, 5, 4], dtype=np.int64)
    sub = gray[sub]
    # Binary = sub // 4 (top half vs bottom half of Möbius)
    y = (sub // 4).astype(np.int64)
    return sub, y


def mobius_target_xyz(sub: torch.Tensor) -> torch.Tensor:
    """Map sub-class index 0..7 to a point on the Möbius."""
    alpha = sub.float() * (math.pi / 4)
    x = torch.cos(alpha)
    y = torch.sin(alpha)
    z = torch.sin(alpha / 2) * 0.7
    return torch.stack([x, y, z], dim=-1)


def main():
    train_d = get_activations("train")
    test_d = get_activations("test")
    emb_tr, emb_te = train_d["embeddings"], test_d["embeddings"]
    sub_tr, y_tr = subclass_and_target(train_d)
    sub_te, y_te = subclass_and_target(test_d)
    print(f"Sub-class counts (train): {np.bincount(sub_tr, minlength=8)}")
    print(f"Target balance: train {y_tr.mean():.3f}  test {y_te.mean():.3f}")

    torch.manual_seed(0)
    Xtr = torch.from_numpy(emb_tr.astype(np.float32))
    ytr = torch.from_numpy(y_tr.astype(np.float32))
    Xte = torch.from_numpy(emb_te.astype(np.float32))
    yte = torch.from_numpy(y_te.astype(np.float32))
    sub_tr_t = torch.from_numpy(sub_tr.astype(np.int64))

    m = MobiusHead(in_dim=emb_tr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    bs = 256
    n = len(Xtr)
    epochs = 250
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_bce, ep_aux, nb = 0.0, 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits, L = m(Xtr[idx], return_L=True)
            loss_bce = bce(logits, ytr[idx])
            target_xyz = mobius_target_xyz(sub_tr_t[idx])
            loss_aux = ((L - target_xyz) ** 2).sum(-1).mean()
            loss = loss_bce + 4.0 * loss_aux
            loss.backward()
            opt.step()
            ep_bce += loss_bce.item()
            ep_aux += loss_aux.item()
            nb += 1
        if ep % 25 == 0 or ep == epochs - 1:
            m.eval()
            with torch.no_grad():
                acc = ((m(Xte) > 0).float() == yte).float().mean().item()
            print(f"  ep {ep:3d}  bce {ep_bce/nb:.4f}  aux {ep_aux/nb:.4f}  acc_te {acc:.4f}")
    torch.save(m.state_dict(), MODEL_DIR / "model_C_mobius.pt")
    print(f"  saved -> {MODEL_DIR / 'model_C_mobius.pt'}")

    # Analyse
    m.eval()
    with torch.no_grad():
        logit_tr, L_tr = m(Xtr, return_L=True)
        logit_te, L_te = m(Xte, return_L=True)
    L_tr = L_tr.numpy()
    L_te = L_te.numpy()

    print("\n--- Probes on the 3-D bottleneck L ---")
    lin = linear_probe_acc(L_tr, y_tr, L_te, y_te)
    mlp = mlp_probe_acc(L_tr, y_tr, L_te, y_te)
    print(f"  Linear probe on 3-D L: {lin:.4f}")
    print(f"  MLP probe on 3-D L:    {mlp:.4f}")

    # Probes on JUST the (x,y) 2-D projection — should fail because the Möbius
    # identifies top/bottom on the same (x,y).
    lin2 = linear_probe_acc(L_tr[:, :2], y_tr, L_te[:, :2], y_te)
    mlp2 = mlp_probe_acc(L_tr[:, :2], y_tr, L_te[:, :2], y_te)
    print(f"  Linear probe on 2-D (x,y) only: {lin2:.4f}  (topological collapse)")
    print(f"  MLP probe on 2-D (x,y) only:    {mlp2:.4f}")

    # Probe on just z
    lin_z = linear_probe_acc(L_tr[:, 2:3], y_tr, L_te[:, 2:3], y_te)
    print(f"  Linear probe on z alone:        {lin_z:.4f}")

    print("\n--- Causal: flip z sign per example ---")
    L_flipz = L_te.copy()
    L_flipz[:, 2] *= -1
    with torch.no_grad():
        h3 = torch.relu(m.l4(torch.from_numpy(L_flipz.astype(np.float32))))
        logit_f = m.l5(h3).squeeze(-1).numpy()
    flip_rate = ((logit_f > 0) != (logit_te.numpy() > 0)).mean()
    print(f"  flipping z (the Möbius twist) -> flip rate {flip_rate:.4f}  "
          f"(theory: full ~1.0 because z is THE label dim)")

    print("\n--- Causal: rotate (x,y) by 2π/8 = 45° (one sub-class shift) ---")
    R = np.array([[math.cos(math.pi/4), -math.sin(math.pi/4), 0],
                  [math.sin(math.pi/4),  math.cos(math.pi/4), 0],
                  [0, 0, 1]], dtype=np.float32)
    L_rot = L_te @ R.T
    with torch.no_grad():
        h3 = torch.relu(m.l4(torch.from_numpy(L_rot.astype(np.float32))))
        logit_r = m.l5(h3).squeeze(-1).numpy()
    flip = ((logit_r > 0) != (logit_te.numpy() > 0)).mean()
    print(f"  rotating (x,y) keeping z fixed -> flip rate {flip:.4f}  "
          f"(theory: small because (x,y) rotation doesn't change the side z)")

    # Visualise in 3-D
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    cmap = plt.cm.tab10
    for s in range(8):
        mask = sub_tr == s
        ax.scatter(L_tr[mask, 0], L_tr[mask, 1], L_tr[mask, 2],
                   s=4, alpha=0.5, c=[cmap(s)], label=f"sub={s}")
    ax.set_xlabel("L[0] = x"); ax.set_ylabel("L[1] = y"); ax.set_zlabel("L[2] = z")
    ax.set_title("Model C: L embedded on a Möbius-inspired 3-D loop\n(coloured by sub-class)")
    ax.legend(loc="best", fontsize=7, ncol=2)

    ax = fig.add_subplot(1, 2, 2)
    # 2-D (x, y) projection: top and bottom of Möbius overlap
    for v in [0, 1]:
        mask = y_tr == v
        ax.scatter(L_tr[mask, 0], L_tr[mask, 1], s=5, alpha=0.6,
                   c=("#bab0ac" if v == 0 else "C3"), label=f"target={v}")
    ax.set_aspect("equal")
    ax.set_title("Same data projected to 2-D (x, y):\ntargets collapse onto each other — topological obstruction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_C_mobius.png", dpi=140)
    plt.close(fig)
    print(f"  saved -> {FIG_DIR / 'model_C_mobius.png'}")


if __name__ == "__main__":
    main()
