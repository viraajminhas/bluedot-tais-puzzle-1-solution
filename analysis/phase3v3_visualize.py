"""Phase 3 v3 analysis: probe & visualise the 2-D bottleneck models."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import FEATURE_NAMES, get_activations
from phase1_probes import linear_probe_acc, mlp_probe_acc
from phase3v3_train import CircleBottleneckHead

FIG_DIR = Path(__file__).parent / "figs"
MODEL_DIR = Path(__file__).parent / "trained_models"


def load(name: str) -> CircleBottleneckHead:
    m = CircleBottleneckHead()
    m.load_state_dict(torch.load(MODEL_DIR / f"{name}.pt", map_location="cpu",
                                  weights_only=False))
    m.eval()
    return m


def extract(model: CircleBottleneckHead, X: np.ndarray):
    with torch.no_grad():
        x = torch.from_numpy(X.astype(np.float32))
        logits, L = model(x, return_L=True)
    return logits.numpy(), L.numpy()


def analyze(name: str, n_subclasses: int, expected_period: int,
            get_subclass_target):
    print(f"\n=== {name} | n_sub={n_subclasses} | expected period={expected_period} ===")
    train = get_activations("train")
    test = get_activations("test")
    sub_tr, y_tr = get_subclass_target(train)
    sub_te, y_te = get_subclass_target(test)
    y_tr_i, y_te_i = y_tr.astype(np.int64), y_te.astype(np.int64)

    m = load(name)
    logit_tr, L_tr = extract(m, train["embeddings"])
    logit_te, L_te = extract(m, test["embeddings"])
    acc_target = ((logit_te > 0).astype(np.int64) == y_te_i).mean()
    print(f"  Model test acc on target: {acc_target:.4f}")

    # -- 1. Linear vs MLP probe on L (2-D bottleneck) --------------------- #
    lin = linear_probe_acc(L_tr, y_tr_i, L_te, y_te_i)
    mlp = mlp_probe_acc(L_tr, y_tr_i, L_te, y_te_i)
    print(f"\n  Probes on L (2-D bottleneck):")
    print(f"    linear: {lin:.4f}   mlp: {mlp:.4f}   gap: {mlp - lin:+.4f}")
    print(f"    (Theory: linear must fail for period > 1; expect ~50%.)")

    # -- 2. Per-harmonic Fourier probe ------------------------------------ #
    theta_tr = np.arctan2(L_tr[:, 1], L_tr[:, 0])
    theta_te = np.arctan2(L_te[:, 1], L_te[:, 0])
    print("\n  Linear probe on Fourier features (cos kθ, sin kθ) at harmonic k:")
    for k in range(1, 5):
        ftr = np.stack([np.cos(k * theta_tr), np.sin(k * theta_tr)], axis=-1)
        fte = np.stack([np.cos(k * theta_te), np.sin(k * theta_te)], axis=-1)
        acc = linear_probe_acc(ftr, y_tr_i, fte, y_te_i)
        marker = "  <-- the right harmonic" if k == expected_period else ""
        print(f"    k={k}: {acc:.4f}{marker}")

    # -- 3. Visualisation ------------------------------------------------- #
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    cmap = plt.cm.tab10
    ax = axs[0]
    for s in range(n_subclasses):
        mask = sub_tr == s
        ax.scatter(L_tr[mask, 0], L_tr[mask, 1], s=6, alpha=0.6, c=[cmap(s)],
                   label=f"sub={s}")
    t = np.linspace(0, 2 * math.pi, 200)
    ax.plot(np.cos(t), np.sin(t), "k--", lw=0.7, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_title(f"{name}: L coloured by sub-class")
    ax.legend(loc="best", fontsize=7, ncol=2)

    ax = axs[1]
    for v in [0, 1]:
        mask = y_tr_i == v
        ax.scatter(L_tr[mask, 0], L_tr[mask, 1], s=6, alpha=0.55,
                   c=("#bab0ac" if v == 0 else "C3"), label=f"target={v}")
    ax.plot(np.cos(t), np.sin(t), "k--", lw=0.7, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_title(f"{name}: same scatter coloured by binary target")
    ax.legend()

    ax = axs[2]
    ax.hist(theta_tr[y_tr_i == 0], bins=60, alpha=0.55, color="lightgrey",
            label="target=0")
    ax.hist(theta_tr[y_tr_i == 1], bins=60, alpha=0.75, color="C3",
            label="target=1")
    ax.set_xlabel("angle θ (rad)")
    ax.set_title("Angular distribution per class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_geometry.png", dpi=140)
    plt.close(fig)
    print(f"\n  saved -> {FIG_DIR / f'{name}_geometry.png'}")

    # -- 4. Causal: rotate L by one arc, downstream should flip -------------- #
    rot = 2 * math.pi / n_subclasses
    R = np.array([[math.cos(rot), -math.sin(rot)],
                  [math.sin(rot),  math.cos(rot)]], dtype=np.float32)
    L_rot = (L_te @ R.T).astype(np.float32)
    with torch.no_grad():
        Lr = torch.from_numpy(L_rot)
        h3r = torch.relu(m.l4(Lr))
        logit_r = m.l5(h3r).squeeze(-1).numpy()
    flip = (np.sign(logit_r) != np.sign(logit_te)).mean()
    pred_te = (logit_te > 0).astype(np.int64)
    pred_r  = (logit_r > 0).astype(np.int64)
    flip_pred = (pred_te != pred_r).mean()
    print(f"\n  Causal: rotate L by 2π/{n_subclasses} -> downstream prediction flip rate: {flip_pred:.4f}")
    print(f"    (Theory: should be ~1.0 because adjacent sub-classes have opposite targets.)")


def model_A_subclass_target(d):
    color = d["labels"][:, FEATURE_NAMES.index("color")]
    food = d["labels"][:, FEATURE_NAMES.index("food")]
    sub = (color * 2 + (color ^ food)).astype(np.int64)   # Gray code
    y = (color ^ food).astype(np.int64)
    return sub, y


def model_B_subclass_target(d):
    sub = (d["template_id"] % 6).astype(np.int64)
    y = (sub % 2).astype(np.int64)
    return sub, y


if __name__ == "__main__":
    analyze("model_A_v3_bottleneck", 4, expected_period=2,
            get_subclass_target=model_A_subclass_target)
    analyze("model_B_v3_bottleneck", 6, expected_period=3,
            get_subclass_target=model_B_subclass_target)
