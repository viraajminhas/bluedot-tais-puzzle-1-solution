"""Phase 10: is the country DECODER direction stable across seeds?

If v_manifold is just incidental within-class variance (template structure),
it should rotate across seeds. The DECODER direction (the h2 direction the
model actually reads to compute the country logit) should be much more
stable, because the task is the same across seeds.

Procedure:
  1. For puzzle + each replica (seeds 101, 202, 303), compute at h2:
     - v_manifold = top eigvec of cov(country=1)
     - decoder direction = effective h2 direction the country logit reads from
  2. Compare pairwise cosines of v_manifold across seeds vs decoder directions
     across seeds.
  3. If decoder direction is more stable than v_manifold, that constructively
     confirms the story in the writeup.
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

from common import FEATURE_NAMES, get_activations, load_head

FIG_DIR = Path(__file__).parent / "figs"
MODEL_DIR = Path(__file__).parent / "trained_models"
REPLICA_DIR = MODEL_DIR / "replicas"
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

    def forward(self, x):
        return self.layers(x)


def load_replica(seed: int) -> PuzzleHead:
    m = PuzzleHead()
    m.load_state_dict(torch.load(REPLICA_DIR / f"replica_seed{seed}.pt",
                                  map_location="cpu", weights_only=False))
    m.eval()
    return m


def get_h2(model, embeddings):
    with torch.no_grad():
        x = torch.from_numpy(embeddings.astype(np.float32))
        return model.layers[:6](x).numpy()


def get_v_manifold(h2_tr, y_tr):
    Xp = h2_tr[y_tr == 1] - h2_tr[y_tr == 1].mean(0)
    cov_p = np.cov(Xp.T)
    eigvals, V = np.linalg.eigh(cov_p)
    v = V[:, -1]
    return v / np.linalg.norm(v)


def get_decoder_direction(model, h2_tr, y_tr):
    """Effective h2 direction the country logit reads through W4 + ReLU + W5."""
    # layers[6] = Linear(64, 64) = W4, layers[8] = Linear(64, 8) = W5
    W4 = model.layers[6].weight.detach().numpy()  # (64, 64)
    b4 = model.layers[6].bias.detach().numpy()
    W5 = model.layers[8].weight.detach().numpy()  # (8, 64)
    w_country_h3 = W5[COUNTRY_IDX]                  # (64,)
    pre_h3 = h2_tr @ W4.T + b4
    gate = (pre_h3 > 0).astype(np.float32)         # (N, 64)
    # F=1 conditional decoder direction
    eff_pos = (gate[y_tr == 1] * w_country_h3).sum(0) / (y_tr == 1).sum()
    pos_dir = eff_pos @ W4
    pos_dir = pos_dir / (np.linalg.norm(pos_dir) + 1e-9)
    # F=0 conditional
    eff_neg = (gate[y_tr == 0] * w_country_h3).sum(0) / (y_tr == 0).sum()
    neg_dir = eff_neg @ W4
    neg_dir = neg_dir / (np.linalg.norm(neg_dir) + 1e-9)
    # Difference: this is the direction along which moving increases country logit
    # for F=1 examples and decreases it for F=0 (so most discriminative).
    diff_dir = pos_dir - neg_dir
    diff_dir = diff_dir / (np.linalg.norm(diff_dir) + 1e-9)
    return diff_dir


def main():
    train = get_activations("train")
    y_tr = train["labels"][:, COUNTRY_IDX]

    # Puzzle (original) + 3 replicas
    print("Loading models...")
    models = {"orig": load_head("cpu")}
    for s in [101, 202, 303]:
        try:
            models[f"s{s}"] = load_replica(s)
        except FileNotFoundError:
            print(f"  replica seed {s} not found, skipping")

    # For each model, compute v_manifold and decoder direction at h2
    v_dict = {}
    d_dict = {}
    for name, m in models.items():
        h2_tr = get_h2(m, train["embeddings"])
        v_dict[name] = get_v_manifold(h2_tr, y_tr)
        d_dict[name] = get_decoder_direction(m, h2_tr, y_tr)
        print(f"  {name}: v_manifold and decoder direction computed at h2")

    names = list(models.keys())
    n = len(names)

    print("\n=== Pairwise |cos| between v_manifold (PC1 of cov(F=1)) ===")
    print("     " + "  ".join(f"{nm:>6}" for nm in names))
    for i, nm_i in enumerate(names):
        row = [f"{abs(v_dict[nm_i] @ v_dict[nm_j]):.3f}" for nm_j in names]
        print(f"{nm_i:>4} " + "  ".join(f"{x:>6}" for x in row))

    print("\n=== Pairwise |cos| between DECODER directions (F=1 minus F=0) ===")
    print("     " + "  ".join(f"{nm:>6}" for nm in names))
    for i, nm_i in enumerate(names):
        row = [f"{abs(d_dict[nm_i] @ d_dict[nm_j]):.3f}" for nm_j in names]
        print(f"{nm_i:>4} " + "  ".join(f"{x:>6}" for x in row))

    print("\n=== Within each model: |cos(v_manifold, decoder)| ===")
    for nm in names:
        c = abs(v_dict[nm] @ d_dict[nm])
        print(f"  {nm}: |cos(v_manifold, decoder)| = {c:.3f}")

    # Summary metric: mean off-diagonal cosine
    def offdiag_mean(d):
        accum = []
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if i < j:
                    accum.append(abs(d[ni] @ d[nj]))
        return float(np.mean(accum)), float(np.std(accum))

    v_mean, v_std = offdiag_mean(v_dict)
    d_mean, d_std = offdiag_mean(d_dict)
    print(f"\n=== Summary ===")
    print(f"  Mean off-diagonal |cos| for v_manifold across models: {v_mean:.3f} ± {v_std:.3f}")
    print(f"  Mean off-diagonal |cos| for DECODER directions across models: {d_mean:.3f} ± {d_std:.3f}")
    if d_mean > 1.5 * v_mean:
        print(f"  -> Decoder direction is significantly more stable across seeds")
    elif d_mean > v_mean:
        print(f"  -> Decoder direction is moderately more stable than v_manifold")
    else:
        print(f"  -> No clear difference in stability")

    # ===== Visualization =====
    print("\nBuilding geometric visualization...")
    # Take the puzzle model. Project h2 onto (v_manifold, decoder_direction)
    # so we can see them in 2-D.
    m_orig = models["orig"]
    h2_tr = get_h2(m_orig, train["embeddings"])
    v = v_dict["orig"]
    d = d_dict["orig"]
    # Gram-Schmidt: orthogonalize d w.r.t. v so we get an orthogonal basis
    d_perp = d - (d @ v) * v
    d_perp = d_perp / (np.linalg.norm(d_perp) + 1e-9)
    mu = h2_tr.mean(0)
    p_v = (h2_tr - mu) @ v
    p_d = (h2_tr - mu) @ d_perp

    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))

    ax = axs[0]
    ax.scatter(p_v[y_tr == 0], p_d[y_tr == 0], s=4, alpha=0.25, c="lightgrey",
               label="country=0")
    ax.scatter(p_v[y_tr == 1], p_d[y_tr == 1], s=5, alpha=0.55, c="C3",
               label="country=1")
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.axvline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("projection onto v_manifold  (PC1 of cov(country=1))\n"
                  "= template-correlated within-class variance axis")
    ax.set_ylabel("projection onto decoder direction (orthogonalized)\n"
                  "= direction model actually reads to predict country")
    ax.set_title("Puzzle h2: v_manifold vs decoder direction\n"
                 "the two are nearly perpendicular — v_manifold is NOT the country axis")
    ax.legend(loc="upper right", fontsize=10)

    # Right: stability comparison bar chart
    ax = axs[1]
    pair_names = []
    v_cosines = []
    d_cosines = []
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i < j:
                pair_names.append(f"{ni}\nvs\n{nj}")
                v_cosines.append(abs(v_dict[ni] @ v_dict[nj]))
                d_cosines.append(abs(d_dict[ni] @ d_dict[nj]))
    x = np.arange(len(pair_names))
    w = 0.4
    ax.bar(x - w / 2, v_cosines, width=w, color="#bab0ac",
           label="v_manifold (PC1 of cov(F=1))")
    ax.bar(x + w / 2, d_cosines, width=w, color="C3",
           label="decoder direction (W₅[country] traced back)")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("|cosine similarity| between model pair")
    ax.axhline(1.0, color="k", lw=0.5)
    ax.set_title(f"Cross-model stability (orig + 3 retrained seeds)\n"
                 f"decoder mean {d_mean:.2f} vs v_manifold mean {v_mean:.2f}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "decoder_vs_v_manifold.png", dpi=140)
    plt.close(fig)
    print(f"  saved -> {FIG_DIR / 'decoder_vs_v_manifold.png'}")


if __name__ == "__main__":
    main()
