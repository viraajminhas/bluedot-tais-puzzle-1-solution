"""Phase 10b: also check stability of other candidate axes across seeds.

If neither v_manifold NOR decoder direction is stable, maybe a different
direction is. Candidates:
  - LDA direction (mean F=1 minus mean F=0) — naive linear axis
  - SVM direction (sklearn linear SVC weight)
  - PC1 of all-data h2 (top variance overall)

Also build a combined "what v_manifold encodes" figure for the writeup.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import FEATURE_NAMES, get_activations, load_head, load_split
from phase2b_country import tag_countries

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


def get_h2(model, embeddings):
    with torch.no_grad():
        x = torch.from_numpy(embeddings.astype(np.float32))
        return model.layers[:6](x).numpy()


def load_replica(seed):
    m = PuzzleHead()
    m.load_state_dict(torch.load(REPLICA_DIR / f"replica_seed{seed}.pt",
                                  map_location="cpu", weights_only=False))
    m.eval()
    return m


def main():
    train = get_activations("train")
    y_tr = train["labels"][:, COUNTRY_IDX]

    models = {"orig": load_head("cpu")}
    for s in [101, 202, 303]:
        models[f"s{s}"] = load_replica(s)

    candidates = {"v_manifold": {}, "decoder": {}, "LDA (mean diff)": {},
                  "LR weight": {}, "all-data PC1": {}}

    for name, m in models.items():
        h2 = get_h2(m, train["embeddings"])
        # v_manifold: PC1 of cov(F=1)
        Xp = h2[y_tr == 1] - h2[y_tr == 1].mean(0)
        _, V = np.linalg.eigh(np.cov(Xp.T))
        candidates["v_manifold"][name] = V[:, -1]
        # Decoder direction (F=1 - F=0 effective h2 dir)
        W4 = m.layers[6].weight.detach().numpy()
        b4 = m.layers[6].bias.detach().numpy()
        W5 = m.layers[8].weight.detach().numpy()
        w_c = W5[COUNTRY_IDX]
        gate = (h2 @ W4.T + b4 > 0).astype(np.float32)
        d_pos = ((gate[y_tr == 1] * w_c).sum(0) / (y_tr == 1).sum()) @ W4
        d_neg = ((gate[y_tr == 0] * w_c).sum(0) / (y_tr == 0).sum()) @ W4
        d = d_pos - d_neg
        candidates["decoder"][name] = d / np.linalg.norm(d)
        # LDA direction at h2
        lda = h2[y_tr == 1].mean(0) - h2[y_tr == 0].mean(0)
        candidates["LDA (mean diff)"][name] = lda / (np.linalg.norm(lda) + 1e-9)
        # Linear LR probe weight
        sc = StandardScaler().fit(h2)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(h2), y_tr)
        lr_w = clf.coef_[0] / sc.scale_
        candidates["LR weight"][name] = lr_w / np.linalg.norm(lr_w)
        # PC1 of all-data h2
        Xa = h2 - h2.mean(0)
        _, Va = np.linalg.eigh(np.cov(Xa.T))
        candidates["all-data PC1"][name] = Va[:, -1]

    names = list(models.keys())
    print(f"{'candidate axis':<22} " + " ".join(f"{n:>6}" for n in names) +
          f"  {'mean off-diag |cos|':>20}")
    print("-" * 100)
    for cname, dirs in candidates.items():
        cosines = []
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if i < j:
                    cosines.append(abs(dirs[ni] @ dirs[nj]))
        mean_cos = np.mean(cosines)
        # First row: |cos| with orig
        row = " ".join(f"{abs(dirs[n] @ dirs['orig']):>6.3f}" for n in names)
        print(f"{cname:<22} {row}  {mean_cos:>20.3f}")

    # ===== Combined "what v_manifold encodes" figure =====
    print("\nBuilding combined axis-meaning figure...")
    texts_tr, _, _ = load_split("train")
    tags_tr = tag_countries(texts_tr)
    h2 = get_h2(models["orig"], train["embeddings"])
    v_mf = candidates["v_manifold"]["orig"]
    mu = h2.mean(0)
    proj = (h2 - mu) @ v_mf
    labels = train["labels"]

    fig, axs = plt.subplots(1, 3, figsize=(19, 6))

    # Panel 1: per-feature co-occurrence correlation with v_manifold (per-country)
    ax = axs[0]
    # Build per-country mean projection + per-country mean co-occurrence
    keep_idx = [i for i, t in enumerate(tags_tr) if t is not None and y_tr[i] == 1]
    proj_idx = proj[keep_idx]
    tags_idx = [tags_tr[i] for i in keep_idx]
    labels_idx = labels[keep_idx]
    from collections import Counter
    cnts = Counter(tags_idx)
    keep_c = {c for c, n in cnts.items() if n >= 10}
    sorted_c = sorted(keep_c)
    per_c_proj = []
    per_c_feats = []
    for c in sorted_c:
        m = np.array([t == c for t in tags_idx])
        per_c_proj.append(proj_idx[m].mean())
        per_c_feats.append(labels_idx[m].mean(0))
    per_c_proj = np.array(per_c_proj)
    per_c_feats = np.array(per_c_feats)
    feat_names = [n for n in FEATURE_NAMES if n != "country"]
    feat_corrs = []
    for fi, n in enumerate(FEATURE_NAMES):
        if n == "country":
            continue
        r = np.corrcoef(per_c_proj, per_c_feats[:, fi])[0, 1]
        feat_corrs.append((n, r))
    feat_corrs.sort(key=lambda x: x[1])
    colors_bar = ["#d62728" if r < 0 else "#2ca02c" for _, r in feat_corrs]
    ax.barh([n for n, _ in feat_corrs], [r for _, r in feat_corrs],
            color=colors_bar, edgecolor="black", lw=0.4)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Pearson correlation with v_manifold projection (per country)")
    ax.set_title("What v_manifold actually encodes:\nco-occurrence with OTHER features\n"
                 "(not country identity)", fontsize=11)
    ax.grid(True, alpha=0.2, axis="x")
    for i, (n, r) in enumerate(feat_corrs):
        ax.text(r + (0.02 if r > 0 else -0.02), i, f"{r:+.2f}",
                va="center", ha="left" if r > 0 else "right", fontsize=9)

    # Panel 2: per-template_id projection on v_manifold
    ax = axs[1]
    tids = train["template_id"]
    for tid in range(8):
        mask = (y_tr == 1) & (tids == tid)
        if mask.any():
            ax.hist(proj[mask], bins=30, alpha=0.5, label=f"template {tid}")
    ax.set_xlabel("v_manifold projection")
    ax.set_ylabel("count")
    ax.set_title("Per-template projection on v_manifold\n"
                 "(template 1 = 'Does X verb Y?' is the low extreme)",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    # Panel 3: extreme texts as annotations
    ax = axs[2]
    ax.axis("off")
    pos_idx = np.where(y_tr == 1)[0]
    proj_pos = proj[pos_idx]
    order_hi = np.argsort(-proj_pos)
    order_lo = np.argsort(proj_pos)
    text_blob = "EXAMPLES AT EXTREMES\n\n"
    text_blob += "HIGHEST v_manifold projection\n"
    text_blob += "(numerals, colors, no person):\n\n"
    for j in order_hi[:4]:
        idx = pos_idx[j]
        s = texts_tr[idx]
        if len(s) > 78:
            s = s[:75] + "..."
        text_blob += f"• [{proj_pos[j]:+.2f}] {s}\n"
    text_blob += "\nLOWEST v_manifold projection\n"
    text_blob += "(person names, body parts, questions):\n\n"
    for j in order_lo[:4]:
        idx = pos_idx[j]
        s = texts_tr[idx]
        if len(s) > 78:
            s = s[:75] + "..."
        text_blob += f"• [{proj_pos[j]:+.2f}] {s}\n"
    ax.text(0.0, 1.0, text_blob, va="top", ha="left", family="monospace",
            fontsize=8.5)
    ax.set_title("Texts at the two ends of v_manifold confirm the pattern",
                 fontsize=11)

    fig.suptitle("v_manifold is the template-structure axis, not the country axis",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v_manifold_meaning.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {FIG_DIR / 'v_manifold_meaning.png'}")


if __name__ == "__main__":
    main()
