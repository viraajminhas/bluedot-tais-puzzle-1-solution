"""Phase 5d: head-to-head comparison figures.

(A) Puzzle's country encoding at h2 vs Model A v3's L encoding — same plot
    style, same scale.
(B) Eigenvalue / cumulative-variance contrast: the puzzle's country manifold
    (effective dim ~5) vs Model A's circle (effective dim exactly 2).
(C) Causal ablation: zero out the 2 circle dims of Model A vs zero out the
    other... wait, Model A's L IS 2-D, so the only thing to ablate is L itself.
    Instead, compare: pass L unchanged vs pass random L (same magnitude).
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from common import FEATURE_NAMES, get_activations, load_head
from phase3v3_train import CircleBottleneckHead
from phase3v3_visualize import model_A_subclass_target

FIG_DIR = Path(__file__).parent / "figs"
COUNTRY_IDX = FEATURE_NAMES.index("country")
MODEL_DIR = Path(__file__).parent / "trained_models"


def get_puzzle_country_pcs():
    train = get_activations("train")
    X = train["h2"]
    y = train["labels"][:, COUNTRY_IDX]
    Xp_c = X[y == 1] - X[y == 1].mean(0)
    pca = PCA(n_components=2).fit(Xp_c)
    # Project all examples in this 2-D plane
    mu = X.mean(0)
    proj = pca.transform(X - mu)
    return proj, y


def get_modelA_L():
    m = CircleBottleneckHead()
    m.load_state_dict(torch.load(MODEL_DIR / "model_A_v3_bottleneck.pt",
                                  map_location="cpu", weights_only=False))
    m.eval()
    train = get_activations("train")
    with torch.no_grad():
        _, L = m(torch.from_numpy(train["embeddings"].astype(np.float32)),
                 return_L=True)
    L = L.numpy()
    sub, y = model_A_subclass_target(train)
    return L, sub, y, m


def side_by_side_geometry():
    print("Building side-by-side geometry figure...")
    proj_p, y_p = get_puzzle_country_pcs()
    L_a, _, y_a, _ = get_modelA_L()

    fig, axs = plt.subplots(1, 2, figsize=(14, 6.5))

    # Puzzle country at h2
    ax = axs[0]
    ax.scatter(proj_p[y_p == 0, 0], proj_p[y_p == 0, 1], s=4, alpha=0.25,
               c="lightgrey", label="country=0")
    ax.scatter(proj_p[y_p == 1, 0], proj_p[y_p == 1, 1], s=5, alpha=0.55,
               c="C3", label="country=1")
    ax.set_title("PUZZLE: country at h2 (top-2 PCs of cov(F=1))\n"
                 "→ thin elongated manifold inside broad cloud, same mean")
    ax.set_xlabel("PC1 of cov(country=1)"); ax.set_ylabel("PC2")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    # Model A v3 L
    ax = axs[1]
    ax.scatter(L_a[y_a == 0, 0], L_a[y_a == 0, 1], s=5, alpha=0.5,
               c="lightgrey", label="target=0")
    ax.scatter(L_a[y_a == 1, 0], L_a[y_a == 1, 1], s=5, alpha=0.6,
               c="C3", label="target=1")
    t = np.linspace(0, 2 * math.pi, 200)
    ax.plot(np.cos(t), np.sin(t), "k--", lw=0.6, alpha=0.5)
    ax.set_title("MODEL A v3: L for color XOR food\n"
                 "→ 4 crisp clusters at 0/90/180/270°, alternating labels")
    ax.set_xlabel("L[0]"); ax.set_ylabel("L[1]")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Two non-linear encodings, side-by-side. "
        "Puzzle: distributed 5-D manifold; "
        "Ours: 2-D circle requiring period-2 Fourier decode.",
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "side_by_side_puzzle_vs_modelA.png", dpi=150)
    plt.close(fig)
    print(f"  saved -> {FIG_DIR / 'side_by_side_puzzle_vs_modelA.png'}")


def causal_ablation_modelA():
    print("\nCausal ablation: replace Model A's L with various interventions.")
    L_a, sub, y_a, m = get_modelA_L()

    train = get_activations("train")
    test = get_activations("test")
    with torch.no_grad():
        _, L_te = m(torch.from_numpy(test["embeddings"].astype(np.float32)),
                    return_L=True)
    L_te = L_te.numpy()
    y_te = model_A_subclass_target(test)[1]

    def replay(L_array):
        with torch.no_grad():
            Lt = torch.from_numpy(L_array.astype(np.float32))
            h3 = torch.relu(m.l4(Lt))
            logit = m.l5(h3).squeeze(-1).numpy()
        return (logit > 0).astype(np.int64)

    base_pred = replay(L_te)
    base_acc = (base_pred == y_te).mean()
    print(f"  Baseline acc: {base_acc:.4f}")

    # Intervention 1: zero out L
    zero_acc = (replay(np.zeros_like(L_te)) == y_te).mean()
    print(f"  L := zeros   -> acc {zero_acc:.4f}")

    # Intervention 2: random L on the unit circle (no information)
    rng = np.random.default_rng(0)
    rand_angles = rng.uniform(0, 2 * math.pi, size=len(L_te))
    L_rand = np.stack([np.cos(rand_angles), np.sin(rand_angles)], axis=-1).astype(np.float32)
    rand_acc = (replay(L_rand) == y_te).mean()
    print(f"  L := random angle on unit circle -> acc {rand_acc:.4f}")

    # Intervention 3: rotate L by π (180°) — should map each cluster to its
    # antipode; in period-2, antipode has the SAME target, so prediction
    # should NOT flip.
    R180 = np.array([[-1, 0], [0, -1]], dtype=np.float32)
    L_rot180 = L_te @ R180.T
    flip_180 = (replay(L_rot180) != base_pred).mean()
    print(f"  L rotated by 180°: flip rate {flip_180:.4f} "
          f"(period 2 predicts NO flip ≈ 0%)")

    # Intervention 4: rotate by 90° -- should flip ~100%
    R90 = np.array([[0, -1], [1, 0]], dtype=np.float32)
    flip_90 = (replay(L_te @ R90.T) != base_pred).mean()
    print(f"  L rotated by  90°: flip rate {flip_90:.4f} "
          f"(period 2 predicts FULL flip ≈ 100%)")

    # Intervention 5: rotate by 45° -- intermediate
    R45 = np.array([[math.cos(math.pi/4), -math.sin(math.pi/4)],
                    [math.sin(math.pi/4),  math.cos(math.pi/4)]], dtype=np.float32)
    flip_45 = (replay(L_te @ R45.T) != base_pred).mean()
    print(f"  L rotated by  45°: flip rate {flip_45:.4f}")

    return {
        "base": base_acc, "zero": zero_acc, "random": rand_acc,
        "flip_90": flip_90, "flip_180": flip_180, "flip_45": flip_45
    }


if __name__ == "__main__":
    side_by_side_geometry()
    causal_ablation_modelA()
