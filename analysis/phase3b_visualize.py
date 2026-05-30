"""Phase 3b: visualise and probe Model A / Model B representations.

For Model A (circular): confirm the 4-arc structure emerged at h2 and that
the target = color XOR food is encoded via a Fourier-period-2 readout.
For Model B (parity): characterise the failure / structure if any.
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
from sklearn.linear_model import LogisticRegression

from common import FEATURE_NAMES, get_activations
from phase1_probes import linear_probe_acc, mlp_probe_acc
from phase3_train_weird import WeirdHead, make_target_A, make_target_B

FIG_DIR = Path(__file__).parent / "figs"
MODEL_DIR = Path(__file__).parent / "trained_models"


def load_weird(name: str, in_dim=384) -> WeirdHead:
    m = WeirdHead(in_dim=in_dim)
    m.load_state_dict(torch.load(MODEL_DIR / f"{name}.pt", map_location="cpu",
                                  weights_only=False))
    m.eval()
    return m


def extract_h2(model: WeirdHead, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(X.astype(np.float32))
        _, h2 = model(x, return_h2=True)
    return h2.numpy()


def analyze_A():
    print("=== Analysing Model A (circular) ===")
    train = get_activations("train")
    test = get_activations("test")
    m = load_weird("model_A_circular")
    h2_tr = extract_h2(m, train["embeddings"])
    h2_te = extract_h2(m, test["embeddings"])
    y_tr, color_tr, food_tr = make_target_A(train["labels"])
    y_te, color_te, food_te = make_target_A(test["labels"])
    y_tr_i = y_tr.astype(np.int64)
    y_te_i = y_te.astype(np.int64)
    subclass_tr = (color_tr * 2 + food_tr).astype(np.int64)   # 0..3
    subclass_te = (color_te * 2 + food_te).astype(np.int64)

    # ---- (1) Probe accuracies on h2 -------------------------------------- #
    print("\n  Probe accuracies on h2:")
    lin = linear_probe_acc(h2_tr, y_tr_i, h2_te, y_te_i)
    mlp = mlp_probe_acc(h2_tr, y_tr_i, h2_te, y_te_i)
    print(f"    linear={lin:.4f}  mlp={mlp:.4f}")
    # 1-feature probe on first 2 coords (the forced circle dims)
    lin2 = linear_probe_acc(h2_tr[:, :2], y_tr_i, h2_te[:, :2], y_te_i)
    mlp2 = mlp_probe_acc(h2_tr[:, :2], y_tr_i, h2_te[:, :2], y_te_i)
    print(f"    on (h2[:,0:2]) linear={lin2:.4f}  mlp={mlp2:.4f}")
    # Fourier features: cos(theta), sin(theta), cos(2 theta), sin(2 theta)
    theta_tr = np.arctan2(h2_tr[:, 1], h2_tr[:, 0])
    theta_te = np.arctan2(h2_te[:, 1], h2_te[:, 0])
    f_tr = np.stack([np.cos(theta_tr), np.sin(theta_tr),
                     np.cos(2 * theta_tr), np.sin(2 * theta_tr)], axis=1)
    f_te = np.stack([np.cos(theta_te), np.sin(theta_te),
                     np.cos(2 * theta_te), np.sin(2 * theta_te)], axis=1)
    # Linear probe on EACH subset
    print("    linear on Fourier features:")
    for name_, idx in [("[cos, sin] (period 1)", [0, 1]),
                       ("[cos2, sin2] (period 2)", [2, 3]),
                       ("[all 4]", [0, 1, 2, 3])]:
        acc = linear_probe_acc(f_tr[:, idx], y_tr_i, f_te[:, idx], y_te_i)
        print(f"      {name_:<24}: {acc:.4f}")

    # ---- (2) Visualise circle ------------------------------------------- #
    print("\n  Visualising h2[:, 0:2]:")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    ax = axs[0]
    colors = ["#4e79a7", "#e15759", "#76b7b2", "#f28e2b"]
    sc_labels = ["color=0,food=0", "color=0,food=1", "color=1,food=0", "color=1,food=1"]
    for s in range(4):
        mask = subclass_tr == s
        ax.scatter(h2_tr[mask, 0], h2_tr[mask, 1], s=8, alpha=0.5, c=colors[s],
                   label=sc_labels[s])
    # Draw target unit circle
    t = np.linspace(0, 2 * math.pi, 200)
    ax.plot(np.cos(t), np.sin(t), "k--", lw=0.6, alpha=0.4)
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("h2[:, 0]")
    ax.set_ylabel("h2[:, 1]")
    ax.set_title("Model A: h2 first two coords coloured by (color, food)")

    # Right: same scatter coloured by TARGET = color XOR food
    ax = axs[1]
    for v in [0, 1]:
        mask = y_tr_i == v
        ax.scatter(h2_tr[mask, 0], h2_tr[mask, 1], s=8, alpha=0.5,
                   c=("#bab0ac" if v == 0 else "C3"),
                   label=f"target={v}")
    ax.plot(np.cos(t), np.sin(t), "k--", lw=0.6, alpha=0.4)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_xlabel("h2[:, 0]")
    ax.set_ylabel("h2[:, 1]")
    ax.set_title("Model A: same scatter coloured by target = color XOR food")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_A_circle.png", dpi=140)
    plt.close(fig)
    print(f"    saved -> {FIG_DIR / 'model_A_circle.png'}")

    # ---- (3) Mean angle per subclass ------------------------------------- #
    print("\n  Empirical mean angle (degrees) per subclass:")
    for s in range(4):
        mask = subclass_tr == s
        ang_deg = np.degrees(np.arctan2(h2_tr[mask, 1].mean(), h2_tr[mask, 0].mean()))
        rad = np.linalg.norm([h2_tr[mask, 0].mean(), h2_tr[mask, 1].mean()])
        target_deg = s * 90.0
        print(f"    s={s} ({sc_labels[s]:<18}): mean angle = {ang_deg:+7.1f}°  "
              f"target = {target_deg:+5.1f}°  centroid_radius={rad:.3f}")


def analyze_B():
    print("\n=== Analysing Model B (4-way parity) ===")
    train = get_activations("train")
    test = get_activations("test")
    m = load_weird("model_B_parity")
    h2_tr = extract_h2(m, train["embeddings"])
    h2_te = extract_h2(m, test["embeddings"])
    y_tr, _ = make_target_B(train["labels"])
    y_te, _ = make_target_B(test["labels"])
    y_tr_i = y_tr.astype(np.int64)
    y_te_i = y_te.astype(np.int64)

    lin = linear_probe_acc(h2_tr, y_tr_i, h2_te, y_te_i)
    mlp = mlp_probe_acc(h2_tr, y_tr_i, h2_te, y_te_i)
    print(f"  Probe on h2: linear={lin:.4f}  mlp={mlp:.4f}")

    # Final model accuracy
    with torch.no_grad():
        x = torch.from_numpy(test["embeddings"].astype(np.float32))
        pred = (m(x) > 0).numpy().astype(np.int64)
    acc = (pred == y_te_i).mean()
    print(f"  Final model test accuracy on parity target: {acc:.4f}")


if __name__ == "__main__":
    analyze_A()
    analyze_B()
