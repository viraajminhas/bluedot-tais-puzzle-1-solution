"""Probe balanced Model B."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import get_activations
from phase1_probes import linear_probe_acc, mlp_probe_acc
from phase3v3_train import CircleBottleneckHead
from phase3v3_visualize import model_B_subclass_target

FIG_DIR = Path(__file__).parent / "figs"
MODEL_DIR = Path(__file__).parent / "trained_models"


def main():
    m = CircleBottleneckHead()
    m.load_state_dict(torch.load(MODEL_DIR / "model_B_v3_balanced.pt",
                                  map_location="cpu", weights_only=False))
    m.eval()

    train = get_activations("train")
    test = get_activations("test")

    # Use BALANCED train for probe fitting
    sub_tr, y_tr = model_B_subclass_target(train)
    sub_te, y_te = model_B_subclass_target(test)
    rng = np.random.default_rng(0)
    counts = np.bincount(sub_tr)
    min_n = counts.min()
    keep = []
    for s in range(6):
        idx = np.where(sub_tr == s)[0]
        keep.append(rng.choice(idx, size=min_n, replace=False))
    keep = np.sort(np.concatenate(keep))

    with torch.no_grad():
        emb_tr_b = train["embeddings"][keep]
        _, L_tr = m(torch.from_numpy(emb_tr_b.astype(np.float32)), return_L=True)
        _, L_te = m(torch.from_numpy(test["embeddings"].astype(np.float32)), return_L=True)
    L_tr = L_tr.numpy()
    L_te = L_te.numpy()
    y_tr_b = y_tr[keep]
    sub_tr_b = sub_tr[keep]

    print("Balanced sub-class counts (train, used for probes):", np.bincount(sub_tr_b))
    lin = linear_probe_acc(L_tr, y_tr_b, L_te, y_te)
    mlp = mlp_probe_acc(L_tr, y_tr_b, L_te, y_te)
    print(f"\nBalanced Model B v3 (period 3, 6 arcs):")
    print(f"  linear probe on L: {lin:.4f}")
    print(f"  MLP probe on L:    {mlp:.4f}")
    print(f"  gap: {mlp - lin:+.4f}")

    theta_tr = np.arctan2(L_tr[:, 1], L_tr[:, 0])
    theta_te = np.arctan2(L_te[:, 1], L_te[:, 0])
    for k in range(1, 5):
        ftr = np.stack([np.cos(k * theta_tr), np.sin(k * theta_tr)], axis=-1)
        fte = np.stack([np.cos(k * theta_te), np.sin(k * theta_te)], axis=-1)
        acc = linear_probe_acc(ftr, y_tr_b, fte, y_te)
        print(f"  Fourier k={k}: {acc:.4f}")

    # Causal: rotation tests
    print("\n  Causal rotations on test L:")
    with torch.no_grad():
        logit_base = m.l5(torch.relu(m.l4(torch.from_numpy(L_te.astype(np.float32))))).squeeze(-1).numpy()
    base_pred = (logit_base > 0).astype(np.int64)
    base_acc = (base_pred == y_te).mean()
    print(f"    baseline acc: {base_acc:.4f}")
    for deg in [60, 90, 120, 180, 240, 300, 360]:
        r = math.radians(deg)
        R = np.array([[math.cos(r), -math.sin(r)],
                      [math.sin(r),  math.cos(r)]], dtype=np.float32)
        Lr = L_te @ R.T
        with torch.no_grad():
            logit_r = m.l5(torch.relu(m.l4(torch.from_numpy(Lr.astype(np.float32))))).squeeze(-1).numpy()
        flip = ((logit_r > 0) != base_pred).mean()
        # For period 3: flip rate predicted by sign(cos(3θ)) shift.
        # cos(3*(θ+r)) = cos(3θ)cos(3r) - sin(3θ)sin(3r)
        # If 3r = π (mod 2π) -> negation = full flip. 3r = π means r = 60°/180°/300°.
        # If 3r = 0 (mod 2π) -> identity = no flip. r = 120°/240°/360°.
        expected_flip = abs(math.sin(3 * r / 2))    # smooth interp
        marker = ""
        if abs((3 * r) % (2 * math.pi) - math.pi) < 0.01:
            marker = "  <-- expect ~100% flip"
        elif abs((3 * r) % (2 * math.pi)) < 0.01:
            marker = "  <-- expect ~0% flip"
        print(f"    rotate {deg:3d}°: flip rate {flip:.4f}{marker}")

    # Visualize
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    cmap = plt.cm.tab10
    ax = axs[0]
    for s in range(6):
        mask = sub_tr_b == s
        ax.scatter(L_tr[mask, 0], L_tr[mask, 1], s=6, alpha=0.6, c=[cmap(s)],
                   label=f"sub={s}")
    t = np.linspace(0, 2 * math.pi, 200)
    ax.plot(np.cos(t), np.sin(t), "k--", lw=0.7, alpha=0.5)
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.set_title("Balanced Model B v3: L coloured by sub-class")
    ax = axs[1]
    for v in [0, 1]:
        mask = y_tr_b == v
        ax.scatter(L_tr[mask, 0], L_tr[mask, 1], s=6, alpha=0.55,
                   c=("#bab0ac" if v == 0 else "C3"), label=f"target={v}")
    ax.plot(np.cos(t), np.sin(t), "k--", lw=0.7, alpha=0.5)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Balanced Model B v3: L coloured by target")
    ax = axs[2]
    ax.hist(theta_tr[y_tr_b == 0], bins=60, alpha=0.55, color="lightgrey", label="target=0")
    ax.hist(theta_tr[y_tr_b == 1], bins=60, alpha=0.75, color="C3", label="target=1")
    ax.set_xlabel("angle θ (rad)")
    ax.legend()
    ax.set_title("Angular distribution per class (balanced)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_B_v3_balanced_geometry.png", dpi=140)
    plt.close(fig)
    print(f"  saved -> {FIG_DIR / 'model_B_v3_balanced_geometry.png'}")


if __name__ == "__main__":
    main()
