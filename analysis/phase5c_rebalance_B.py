"""Phase 5c: retrain Model B v3 with class-balanced sub-classes.

The original Model B used `template_id % 6` as sub-class, and templates 0-1 are
2x more common than 2-5. With balanced sub-classes, the linear probe should
drop to ~chance (≈50%).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import get_activations
from phase3v3_train import CircleBottleneckHead, circle_target

MODEL_DIR = Path(__file__).parent / "trained_models"


def main():
    train = get_activations("train")
    test = get_activations("test")
    emb_tr, emb_te = train["embeddings"], test["embeddings"]

    sub_tr = (train["template_id"] % 6).astype(np.int64)
    sub_te = (test["template_id"] % 6).astype(np.int64)
    y_tr = (sub_tr % 2).astype(np.int64)
    y_te = (sub_te % 2).astype(np.int64)

    # Class-balance: subsample each sub-class to the minimum count.
    counts = np.bincount(sub_tr)
    min_n = counts.min()
    rng = np.random.default_rng(0)
    keep = []
    for s in range(6):
        idx = np.where(sub_tr == s)[0]
        chosen = rng.choice(idx, size=min_n, replace=False)
        keep.append(chosen)
    keep = np.sort(np.concatenate(keep))
    print(f"Subsampling to {min_n} examples per sub-class -> total {len(keep)}")

    emb_tr_b = emb_tr[keep]
    sub_tr_b = sub_tr[keep]
    y_tr_b = y_tr[keep]
    print(f"Balanced sub-class counts: {np.bincount(sub_tr_b)}")
    print(f"Balanced target counts: train {np.bincount(y_tr_b)}  "
          f"test (unchanged) {np.bincount(y_te)}")

    # Train
    torch.manual_seed(0)
    Xtr = torch.from_numpy(emb_tr_b.astype(np.float32))
    ytr = torch.from_numpy(y_tr_b.astype(np.float32))
    Xte = torch.from_numpy(emb_te.astype(np.float32))
    yte = torch.from_numpy(y_te.astype(np.float32))
    ang_tr = torch.from_numpy(
        (sub_tr_b.astype(np.float32) * (2 * math.pi / 6))
    )
    ang_te = torch.from_numpy(
        (sub_te.astype(np.float32) * (2 * math.pi / 6))
    )

    m = CircleBottleneckHead(in_dim=emb_tr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    bs = 256
    n = len(Xtr)
    for ep in range(300):
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
        if ep % 30 == 0 or ep == 299:
            m.eval()
            with torch.no_grad():
                acc_te = ((m(Xte) > 0).float() == yte).float().mean().item()
            print(f"  ep {ep:3d}  bce {ep_bce/nb:.4f}  aux {ep_aux/nb:.4f}  acc_te {acc_te:.4f}")

    torch.save(m.state_dict(), MODEL_DIR / "model_B_v3_balanced.pt")
    print(f"  saved -> {MODEL_DIR / 'model_B_v3_balanced.pt'}")


if __name__ == "__main__":
    main()
