"""Phase 6d: Model D — period-4 circular encoding (8 arcs).

8 sub-classes from template_id, mapped to 8 angles at 45° apart. Binary target
alternates each arc -> requires Fourier harmonic k=4 to decode. Strictly weirder
than Models A (period 2) and B (period 3).

Forms a clean "weirdness gradient":
   - Puzzle:  single non-linear feature (perp distance) -> 89% from 1 feature
   - Model A: needs Fourier k=2  (2 features)
   - Model B: needs Fourier k=3  (2 features)
   - Model D: needs Fourier k=4  (2 features)
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

from common import get_activations
from phase1_probes import linear_probe_acc, mlp_probe_acc
from phase3v3_train import CircleBottleneckHead, circle_target

FIG_DIR = Path(__file__).parent / "figs"
MODEL_DIR = Path(__file__).parent / "trained_models"


def main():
    train = get_activations("train")
    test = get_activations("test")
    sub_tr = train["template_id"].astype(np.int64)   # 0..7
    sub_te = test["template_id"].astype(np.int64)
    y_tr = (sub_tr % 2).astype(np.int64)
    y_te = (sub_te % 2).astype(np.int64)
    print(f"Sub-class counts (train): {np.bincount(sub_tr)}")
    print(f"Target balance: train {y_tr.mean():.3f}  test {y_te.mean():.3f}")

    # Balance to min-count
    rng = np.random.default_rng(0)
    counts = np.bincount(sub_tr)
    min_n = counts.min()
    keep = []
    for s in range(8):
        idx = np.where(sub_tr == s)[0]
        keep.append(rng.choice(idx, size=min_n, replace=False))
    keep = np.sort(np.concatenate(keep))
    print(f"Subsampling to {min_n}/sub-class -> total {len(keep)}")

    emb_tr_b = train["embeddings"][keep]
    sub_tr_b = sub_tr[keep]
    y_tr_b = y_tr[keep]

    torch.manual_seed(0)
    Xtr = torch.from_numpy(emb_tr_b.astype(np.float32))
    ytr = torch.from_numpy(y_tr_b.astype(np.float32))
    Xte = torch.from_numpy(test["embeddings"].astype(np.float32))
    yte = torch.from_numpy(y_te.astype(np.float32))
    ang_tr = torch.from_numpy(sub_tr_b.astype(np.float32) * (2 * math.pi / 8))
    ang_te = torch.from_numpy(sub_te.astype(np.float32) * (2 * math.pi / 8))

    m = CircleBottleneckHead(in_dim=emb_tr_b.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    bs = 256
    n = len(Xtr)
    epochs = 300
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n)
        ep_bce, ep_aux, nb = 0.0, 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits, L = m(Xtr[idx], return_L=True)
            loss_bce = bce(logits, ytr[idx])
            target_xy = circle_target(ang_tr[idx])
            loss_aux = ((L - target_xy) ** 2).sum(-1).mean()
            loss = loss_bce + 4.0 * loss_aux
            loss.backward()
            opt.step()
            ep_bce += loss_bce.item()
            ep_aux += loss_aux.item()
            nb += 1
        if ep % 30 == 0 or ep == epochs - 1:
            m.eval()
            with torch.no_grad():
                acc_te = ((m(Xte) > 0).float() == yte).float().mean().item()
            print(f"  ep {ep:3d}  bce {ep_bce/nb:.4f}  aux {ep_aux/nb:.4f}  acc_te {acc_te:.4f}")

    torch.save(m.state_dict(), MODEL_DIR / "model_D_period4.pt")

    # Analyse
    m.eval()
    with torch.no_grad():
        _, L_tr = m(Xtr, return_L=True)
        _, L_te = m(Xte, return_L=True)
    L_tr = L_tr.numpy()
    L_te = L_te.numpy()
    y_tr_b_i = y_tr_b.astype(np.int64)
    y_te_i = y_te.astype(np.int64)

    print("\nProbes on 2-D L:")
    lin = linear_probe_acc(L_tr, y_tr_b_i, L_te, y_te_i)
    mlp = mlp_probe_acc(L_tr, y_tr_b_i, L_te, y_te_i)
    print(f"  linear: {lin:.4f}   mlp: {mlp:.4f}   gap: {mlp-lin:+.4f}")

    theta_tr = np.arctan2(L_tr[:, 1], L_tr[:, 0])
    theta_te = np.arctan2(L_te[:, 1], L_te[:, 0])
    print("\nLinear probe on Fourier features at harmonic k:")
    for k in range(1, 6):
        ftr = np.stack([np.cos(k * theta_tr), np.sin(k * theta_tr)], axis=-1)
        fte = np.stack([np.cos(k * theta_te), np.sin(k * theta_te)], axis=-1)
        acc = linear_probe_acc(ftr, y_tr_b_i, fte, y_te_i)
        flag = "  <-- expected right harmonic" if k == 4 else ""
        print(f"  k={k}: {acc:.4f}{flag}")

    # Causal rotations
    print("\nCausal rotations on test L:")
    with torch.no_grad():
        logit_base = m.l5(torch.relu(m.l4(torch.from_numpy(L_te.astype(np.float32))))).squeeze(-1).numpy()
    base_pred = (logit_base > 0).astype(np.int64)
    base_acc = (base_pred == y_te_i).mean()
    print(f"  baseline acc: {base_acc:.4f}")
    for deg in [45, 90, 135, 180, 225, 270, 315, 360]:
        r = math.radians(deg)
        R = np.array([[math.cos(r), -math.sin(r)],
                      [math.sin(r),  math.cos(r)]], dtype=np.float32)
        Lr = L_te @ R.T
        with torch.no_grad():
            logit_r = m.l5(torch.relu(m.l4(torch.from_numpy(Lr.astype(np.float32))))).squeeze(-1).numpy()
        flip = ((logit_r > 0) != base_pred).mean()
        # Period 4: cos(4(θ+r)) = cos(4θ+4r). Same target iff 4r mod 2π == 0
        # i.e., r ∈ {0, π/2, π, 3π/2}. Full flip iff 4r mod 2π == π
        # i.e., r ∈ {π/4, 3π/4, 5π/4, 7π/4}.
        same = (abs((4 * r) % (2 * math.pi)) < 0.01) or (abs((4 * r) % (2 * math.pi) - 2 * math.pi) < 0.01)
        flipper = abs((4 * r) % (2 * math.pi) - math.pi) < 0.01
        marker = ""
        if same:
            marker = "  <-- expect ~0% flip"
        if flipper:
            marker = "  <-- expect ~100% flip"
        print(f"  rotate {deg:3d}°: flip rate {flip:.4f}{marker}")

    # Visualise
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    cmap = plt.cm.tab10
    ax = axs[0]
    for s in range(8):
        mask = sub_tr_b == s
        ax.scatter(L_tr[mask, 0], L_tr[mask, 1], s=6, alpha=0.6, c=[cmap(s)],
                   label=f"sub={s}")
    t = np.linspace(0, 2 * math.pi, 200)
    ax.plot(np.cos(t), np.sin(t), "k--", lw=0.7, alpha=0.5)
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.set_title("Model D (period 4): L coloured by sub-class")
    ax = axs[1]
    for v in [0, 1]:
        mask = y_tr_b_i == v
        ax.scatter(L_tr[mask, 0], L_tr[mask, 1], s=6, alpha=0.55,
                   c=("#bab0ac" if v == 0 else "C3"), label=f"target={v}")
    ax.plot(np.cos(t), np.sin(t), "k--", lw=0.7, alpha=0.5)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Model D: L coloured by binary target")
    ax = axs[2]
    ax.hist(theta_tr[y_tr_b_i == 0], bins=80, alpha=0.55, color="lightgrey", label="target=0")
    ax.hist(theta_tr[y_tr_b_i == 1], bins=80, alpha=0.75, color="C3", label="target=1")
    ax.set_xlabel("angle θ (rad)")
    ax.legend()
    ax.set_title("Angular distribution per class (8 arcs)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_D_period4_geometry.png", dpi=140)
    plt.close(fig)
    print(f"  saved -> {FIG_DIR / 'model_D_period4_geometry.png'}")


if __name__ == "__main__":
    main()
