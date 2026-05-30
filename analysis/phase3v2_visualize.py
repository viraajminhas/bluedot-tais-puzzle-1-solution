"""Phase 3 v2: visualise and probe the trained circle-head models."""
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
from phase3v2_train import CircleHead

FIG_DIR = Path(__file__).parent / "figs"
MODEL_DIR = Path(__file__).parent / "trained_models"


def load_circle(name: str, in_dim=384) -> CircleHead:
    m = CircleHead(in_dim=in_dim)
    m.load_state_dict(torch.load(MODEL_DIR / f"{name}.pt", map_location="cpu",
                                  weights_only=False))
    m.eval()
    return m


def extract(model: CircleHead, X: np.ndarray):
    with torch.no_grad():
        x = torch.from_numpy(X.astype(np.float32))
        logits, h2, circle = model(x, return_all=True)
    return logits.numpy(), h2.numpy(), circle.numpy()


def fourier_features(theta: np.ndarray, max_k: int) -> np.ndarray:
    """Stack [cos(k theta), sin(k theta)] for k = 1..max_k."""
    cols = []
    for k in range(1, max_k + 1):
        cols.append(np.cos(k * theta))
        cols.append(np.sin(k * theta))
    return np.stack(cols, axis=-1)


def analyze(name: str, n_subclasses: int, period: int, get_subclass_target):
    """Generic analysis of a circle-head model with N subclasses placed evenly
    on the unit circle, and binary target = parity of subclass index (period N/2).
    """
    print(f"\n=== {name}  (n_subclasses={n_subclasses}, period={period}) ===")
    train = get_activations("train")
    test = get_activations("test")
    sub_tr, y_tr = get_subclass_target(train)
    sub_te, y_te = get_subclass_target(test)
    y_tr_i, y_te_i = y_tr.astype(np.int64), y_te.astype(np.int64)

    m = load_circle(name)
    logit_tr, h2_tr, circle_tr = extract(m, train["embeddings"])
    logit_te, h2_te, circle_te = extract(m, test["embeddings"])
    acc_target = ((logit_te > 0).astype(np.int64) == y_te_i).mean()
    print(f"  Model test accuracy on target: {acc_target:.4f}")

    # --- 1) Probes on h2: linear vs MLP -------------------------------- #
    print("\n  Probes on h2:")
    lin_h2 = linear_probe_acc(h2_tr, y_tr_i, h2_te, y_te_i)
    mlp_h2 = mlp_probe_acc(h2_tr, y_tr_i, h2_te, y_te_i)
    print(f"    linear: {lin_h2:.4f}   mlp: {mlp_h2:.4f}   gap: {mlp_h2 - lin_h2:+.4f}")

    # --- 2) Probes on the 2-D circle projection ------------------------- #
    print("\n  Probes on circle projection (2-D):")
    lin_c = linear_probe_acc(circle_tr, y_tr_i, circle_te, y_te_i)
    mlp_c = mlp_probe_acc(circle_tr, y_tr_i, circle_te, y_te_i)
    print(f"    linear: {lin_c:.4f}   mlp: {mlp_c:.4f}   gap: {mlp_c - lin_c:+.4f}")

    # --- 3) Probes on Fourier features per harmonic -------------------- #
    print("\n  Linear probe on Fourier features at harmonic k:")
    theta_tr = np.arctan2(circle_tr[:, 1], circle_tr[:, 0])
    theta_te = np.arctan2(circle_te[:, 1], circle_te[:, 0])
    for k in range(1, 5):
        ftr = np.stack([np.cos(k * theta_tr), np.sin(k * theta_tr)], axis=-1)
        fte = np.stack([np.cos(k * theta_te), np.sin(k * theta_te)], axis=-1)
        acc = linear_probe_acc(ftr, y_tr_i, fte, y_te_i)
        print(f"    k={k} (cos {k}θ, sin {k}θ): {acc:.4f}")

    # --- 4) Visualise the circle ------------------------------------------ #
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    cmap = plt.cm.tab10
    ax = axs[0]
    for s in range(n_subclasses):
        mask = sub_tr == s
        ax.scatter(circle_tr[mask, 0], circle_tr[mask, 1],
                   s=6, alpha=0.6, c=[cmap(s)], label=f"sub={s}")
    t = np.linspace(0, 2 * math.pi, 200)
    ax.plot(np.cos(t), np.sin(t), "k--", lw=0.7, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_title(f"{name}: circle projection coloured by sub-class")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.set_xlabel("circle[:, 0]")
    ax.set_ylabel("circle[:, 1]")

    ax = axs[1]
    for v in [0, 1]:
        mask = y_tr_i == v
        ax.scatter(circle_tr[mask, 0], circle_tr[mask, 1], s=6, alpha=0.55,
                   c=("#bab0ac" if v == 0 else "C3"), label=f"target={v}")
    ax.plot(np.cos(t), np.sin(t), "k--", lw=0.7, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_title(f"{name}: same scatter coloured by binary target")
    ax.legend()
    ax.set_xlabel("circle[:, 0]")
    ax.set_ylabel("circle[:, 1]")

    # Angular histogram per class
    ax = axs[2]
    ax.hist(theta_tr[y_tr_i == 0], bins=60, alpha=0.55, color="lightgrey",
            label="target=0")
    ax.hist(theta_tr[y_tr_i == 1], bins=60, alpha=0.75, color="C3", label="target=1")
    ax.set_xlabel("angle (rad)")
    ax.set_title("Angular distribution per class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_geometry.png", dpi=140)
    plt.close(fig)
    print(f"\n  saved -> {FIG_DIR / f'{name}_geometry.png'}")

    # --- 5) Causal intervention: rotate by π/n_subclasses, check flip ---- #
    print("\n  Causal intervention: rotate circle proj by π/{} and re-run downstream"
          .format(n_subclasses))
    # Rotate the circle output by the bin width (one arc-step). This SHOULD flip
    # target=1 ↔ target=0 because the binary alternates each arc.
    rot_angle = math.pi / n_subclasses  # half of (2π / n_subclasses) ... actually
    # subclass spacing is 2π/n, alternating binaries means we need to shift by 2π/n
    # for a full flip (move to next subclass).
    rot_angle = 2 * math.pi / n_subclasses
    cos_r, sin_r = math.cos(rot_angle), math.sin(rot_angle)
    R = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
    rotated_circle = (circle_te @ R.T).astype(np.float32)

    # Replay downstream from the rotated circle:
    # We don't directly have circle -> downstream; circle is a SIDE OUTPUT.
    # But we can simulate by perturbing h2 to make `m.circle(h2)` equal the rotated value.
    # The circle layer is linear: circle = h2 @ W^T + b.  We solve for delta_h2 such
    # that W @ (h2 + delta) + b = rotated_circle, with minimum-norm delta.
    W = m.circle.weight.detach().numpy()     # (2, 64)
    b = m.circle.bias.detach().numpy()        # (2,)
    delta_target = rotated_circle - circle_te  # (N, 2)
    # min-norm delta_h2 satisfying W @ delta_h2 = delta_target  ->  delta_h2 = W+ delta_target
    W_pinv = np.linalg.pinv(W)               # (64, 2)
    delta_h2 = delta_target @ W_pinv.T        # (N, 64)
    h2_perturbed = h2_te + delta_h2

    # Run downstream l4, ReLU, l5
    with torch.no_grad():
        h2p = torch.from_numpy(h2_perturbed.astype(np.float32))
        h3p = torch.relu(m.l4(h2p))
        logit_p = m.l5(h3p).squeeze(-1).numpy()
    pred_p = (logit_p > 0).astype(np.int64)
    flip_rate = (pred_p != ((logit_te > 0).astype(np.int64))).mean()
    print(f"    flip rate after rotation by 2π/{n_subclasses}: {flip_rate:.4f}  "
          f"(if encoding is causal & period-correct, expect ~1.0)")


def model_A_subclass_target(d):
    color = d["labels"][:, FEATURE_NAMES.index("color")]
    food = d["labels"][:, FEATURE_NAMES.index("food")]
    sub = (color * 2 + food).astype(np.int64)
    y = (color ^ food).astype(np.int64)
    return sub, y


def model_B_subclass_target(d):
    sub = (d["template_id"] % 6).astype(np.int64)
    y = (sub % 2).astype(np.int64)
    return sub, y


if __name__ == "__main__":
    analyze("model_A_v2_circle", 4, period=2,
            get_subclass_target=model_A_subclass_target)
    analyze("model_B_v2_period3", 6, period=3,
            get_subclass_target=model_B_subclass_target)
