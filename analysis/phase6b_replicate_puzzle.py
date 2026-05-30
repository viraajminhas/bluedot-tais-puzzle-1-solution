"""Phase 6b: replicate the puzzle's MLP across several seeds.

We retrain the puzzle's 5-layer head from scratch on MiniLM embeddings, with
the same multi-task BCE objective across all 8 features, but different random
seeds. Then check:

  1. Does F = country emerge in each fresh run? (= robustness)
  2. Is the same-mean / shrunk-covariance geometry reproduced?
  3. Is the manifold's principal axis v_manifold stable across seeds?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import FEATURE_NAMES, N_FEATURES, get_activations
from phase1_probes import mlp_probe_acc

MODEL_DIR = Path(__file__).parent / "trained_models" / "replicas"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


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


def train_replica(seed: int, *, epochs=120, lr=1e-3, weight_decay=1e-4,
                  batch_size=256, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = get_activations("train")
    test = get_activations("test")
    Xtr = torch.from_numpy(train["embeddings"].astype(np.float32))
    ytr = torch.from_numpy(train["labels"].astype(np.float32))
    Xte = torch.from_numpy(test["embeddings"].astype(np.float32))
    yte = torch.from_numpy(test["labels"].astype(np.float32))
    m = PuzzleHead().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = m(Xtr[idx])
            loss = bce(logits, ytr[idx])
            loss.backward()
            opt.step()
        if ep % 20 == 0 or ep == epochs - 1:
            with torch.no_grad():
                acc = ((m(Xte) > 0).float() == yte).float().mean(0)
            tag = "  ".join(f"{n[:5]}={a:.2f}" for n, a in zip(FEATURE_NAMES, acc.tolist()))
            print(f"   seed {seed}  ep {ep:3d}  acc: {tag}")
    torch.save(m.state_dict(), MODEL_DIR / f"replica_seed{seed}.pt")
    return m


def analyse_replica(m: PuzzleHead, seed: int):
    """Return v_manifold direction + linear/MLP probe acc for country."""
    train = get_activations("train")
    test = get_activations("test")
    with torch.no_grad():
        Xtr_emb = torch.from_numpy(train["embeddings"].astype(np.float32))
        Xte_emb = torch.from_numpy(test["embeddings"].astype(np.float32))
        h2_tr = m.layers[:6](Xtr_emb).numpy()
        h2_te = m.layers[:6](Xte_emb).numpy()
    y_tr = train["labels"][:, FEATURE_NAMES.index("country")]
    y_te = test["labels"][:, FEATURE_NAMES.index("country")]

    sc = StandardScaler().fit(h2_tr)
    lin = LogisticRegression(C=1.0, max_iter=2000).fit(
        sc.transform(h2_tr), y_tr).score(sc.transform(h2_te), y_te)
    mlp = mlp_probe_acc(h2_tr, y_tr, h2_te, y_te)

    Xp = h2_tr[y_tr == 1] - h2_tr[y_tr == 1].mean(0)
    cov_p = np.cov(Xp.T)
    eigvals, V = np.linalg.eigh(cov_p)
    eigvals, V = eigvals[::-1], V[:, ::-1]
    v_mf = V[:, 0]
    mean_diff_norm = np.linalg.norm(h2_tr[y_tr == 1].mean(0) - h2_tr[y_tr == 0].mean(0))
    print(f"  seed {seed}: country linear={lin:.4f}  mlp={mlp:.4f}  "
          f"|mean_diff|={mean_diff_norm:.3f}  top_eig={eigvals[0]:.3f}")
    return v_mf, eigvals[:5], lin, mlp


def main():
    print("Training 3 replicas of the puzzle's head with different seeds...")
    results = {}
    for seed in [101, 202, 303]:
        print(f"\n=== seed {seed} ===")
        m = train_replica(seed, epochs=120)
        v, eigs, lin, mlp = analyse_replica(m, seed)
        results[seed] = (v, eigs, lin, mlp)

    print("\n=== Cross-seed v_manifold direction alignment ===")
    # Pairwise |cosine| between v_manifold vectors. Plus alignment to the ORIGINAL
    # puzzle's v_manifold.
    train = get_activations("train")
    y_tr = train["labels"][:, FEATURE_NAMES.index("country")]
    from common import load_head
    orig = load_head("cpu")
    with torch.no_grad():
        Xtr_emb = torch.from_numpy(train["embeddings"].astype(np.float32))
        h2_orig = orig.layers[:6](Xtr_emb).numpy()
    Xp = h2_orig[y_tr == 1] - h2_orig[y_tr == 1].mean(0)
    cov_p = np.cov(Xp.T)
    eigs_o, V_o = np.linalg.eigh(cov_p)
    v_orig = V_o[:, np.argmax(eigs_o)]

    print(f"  v_manifold (original puzzle):")
    print(f"  Pairwise |cos| between v_manifold across seeds:")
    seeds = list(results.keys())
    M = np.zeros((len(seeds) + 1, len(seeds) + 1))
    names = ["orig"] + [f"s{s}" for s in seeds]
    vs = [v_orig] + [results[s][0] for s in seeds]
    for i, vi in enumerate(vs):
        for j, vj in enumerate(vs):
            M[i, j] = abs(vi @ vj)
    print(f"      " + " ".join(f"{n:>6}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:>4}  " + " ".join(f"{M[i,j]:6.3f}" for j in range(len(names))))

    # Subspace alignment: project across seeds on top-8 PCs
    print("\n  Subspace overlap (top-8 PCs of cov(F=1)) across seeds:")
    Vs = [V_o[:, np.argsort(eigs_o)[::-1][:8]]]
    for s in seeds:
        # Recompute V from h2 of that seed
        m_s = PuzzleHead()
        m_s.load_state_dict(torch.load(MODEL_DIR / f"replica_seed{s}.pt",
                                        map_location="cpu", weights_only=False))
        with torch.no_grad():
            h2_s = m_s.layers[:6](Xtr_emb).numpy()
        Xp_s = h2_s[y_tr == 1] - h2_s[y_tr == 1].mean(0)
        es, Vs_s = np.linalg.eigh(np.cov(Xp_s.T))
        Vs.append(Vs_s[:, np.argsort(es)[::-1][:8]])
    for i in range(len(Vs)):
        for j in range(i + 1, len(Vs)):
            s = np.linalg.svd(Vs[i].T @ Vs[j], compute_uv=False)
            s = np.clip(s, 0, 1)
            angles = np.degrees(np.arccos(s))
            overlap = (s ** 2).sum()
            print(f"    {names[i]} vs {names[j]}: top-8 overlap = {overlap:.2f}/8  "
                  f"angles deg = [{', '.join(f'{a:.1f}' for a in angles)}]")


if __name__ == "__main__":
    main()
