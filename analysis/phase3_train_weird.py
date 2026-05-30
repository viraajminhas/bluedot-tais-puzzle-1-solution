"""Phase 3: train two models with weirder encodings than the puzzle's.

Model A (Clean):  target = color XOR food, encoded as a 4-arc circular feature
                  in h2 via an auxiliary loss. Readout requires cos(2 theta),
                  i.e. a period-2 Fourier decode.
Model B (Flashy): target = color XOR food XOR sentiment XOR body_part (4-way
                  parity). Free training. Linear and sub-K probes fail.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common import FEATURE_NAMES, get_activations

FIG_DIR = Path(__file__).parent / "figs"
FIG_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path(__file__).parent / "trained_models"
MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Architecture (same as puzzle, but output dim = 1 since we have one target)
# --------------------------------------------------------------------------- #
class WeirdHead(nn.Module):
    def __init__(self, in_dim=384, hidden=64, out_dim=1):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, hidden)   # OUTPUT of relu after l3 = h2 (our target layer)
        self.l4 = nn.Linear(hidden, hidden)
        self.l5 = nn.Linear(hidden, out_dim)

    def forward(self, x, return_h2=False):
        h0 = torch.relu(self.l1(x))
        h1 = torch.relu(self.l2(h0))
        h2 = torch.relu(self.l3(h1))      # this is layer L
        h3 = torch.relu(self.l4(h2))
        logit = self.l5(h3).squeeze(-1)
        if return_h2:
            return logit, h2
        return logit


# --------------------------------------------------------------------------- #
# Target generators
# --------------------------------------------------------------------------- #
def make_target_A(labels):
    """Target = color XOR food (2-way parity)."""
    color = labels[:, FEATURE_NAMES.index("color")]
    food = labels[:, FEATURE_NAMES.index("food")]
    return (color ^ food).astype(np.float32), color, food


def make_target_B(labels):
    """Target = color XOR food XOR sentiment XOR body_part (4-way parity)."""
    a = labels[:, FEATURE_NAMES.index("color")]
    b = labels[:, FEATURE_NAMES.index("food")]
    c = labels[:, FEATURE_NAMES.index("sentiment")]
    d = labels[:, FEATURE_NAMES.index("body_part")]
    return (a ^ b ^ c ^ d).astype(np.float32), (a, b, c, d)


# --------------------------------------------------------------------------- #
# Auxiliary loss for Model A: force circular encoding at h2
# --------------------------------------------------------------------------- #
def angle_for_subclass(color_bit, food_bit):
    """Map (color, food) joint state to one of 4 angles on the unit circle."""
    # 0,0 -> 0   |  0,1 -> pi/2   |  1,1 -> pi  |  1,0 -> 3pi/2
    s = color_bit * 2 + food_bit       # 0..3
    return s.astype(np.float32) * (math.pi / 2.0)


def circular_aux_loss(h2: torch.Tensor, target_angles: torch.Tensor,
                      strength: float = 1.0) -> torch.Tensor:
    """Push the first two coordinates of h2 to lie on the unit circle at the
    target angle. Other coords are unconstrained.
    """
    # Use a learnable "angle frame": treat h2[:,0] as cos-component, h2[:,1] as sin.
    cos_t = torch.cos(target_angles)
    sin_t = torch.sin(target_angles)
    loss = ((h2[:, 0] - cos_t) ** 2 + (h2[:, 1] - sin_t) ** 2).mean()
    return strength * loss


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_model(X_tr, y_tr, X_te, y_te, *,
                aux_fn=None, aux_inputs_tr=None, aux_inputs_te=None,
                epochs=120, lr=1e-3, weight_decay=1e-4, batch_size=256,
                device="cpu", seed=0, log_every=20, model_name="weird"):
    torch.manual_seed(seed)
    Xtr = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    ytr = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    Xte = torch.from_numpy(X_te.astype(np.float32)).to(device)
    yte = torch.from_numpy(y_te.astype(np.float32)).to(device)

    if aux_inputs_tr is not None:
        aux_tr = torch.from_numpy(aux_inputs_tr.astype(np.float32)).to(device)
        aux_te = torch.from_numpy(aux_inputs_te.astype(np.float32)).to(device)
    else:
        aux_tr = aux_te = None

    model = WeirdHead(in_dim=X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    n = len(Xtr)
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss_bce = 0.0
        ep_loss_aux = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            if aux_fn is not None:
                logits, h2 = model(Xtr[idx], return_h2=True)
                loss_bce = bce(logits, ytr[idx])
                loss_aux = aux_fn(h2, aux_tr[idx])
                loss = loss_bce + loss_aux
            else:
                logits = model(Xtr[idx])
                loss_bce = bce(logits, ytr[idx])
                loss_aux = torch.tensor(0.0, device=device)
                loss = loss_bce
            loss.backward()
            opt.step()
            ep_loss_bce += loss_bce.item()
            ep_loss_aux += float(loss_aux.item()) if hasattr(loss_aux, "item") else 0.0
            nb += 1
        # Eval
        if ep % log_every == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                pred_tr = (model(Xtr) > 0).float()
                pred_te = (model(Xte) > 0).float()
                acc_tr = (pred_tr == ytr).float().mean().item()
                acc_te = (pred_te == yte).float().mean().item()
            print(f"  ep {ep:3d}  bce {ep_loss_bce/nb:.4f}  aux {ep_loss_aux/nb:.4f}  "
                  f"acc_tr {acc_tr:.4f}  acc_te {acc_te:.4f}")
            history.append((ep, ep_loss_bce / nb, ep_loss_aux / nb, acc_tr, acc_te))

    # Save
    torch.save(model.state_dict(), MODEL_DIR / f"{model_name}.pt")
    print(f"  saved -> {MODEL_DIR / f'{model_name}.pt'}")
    return model, history


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def run_model_A(epochs=120, device="cpu"):
    print("=== Model A: circular encoding for color XOR food ===")
    train = get_activations("train")
    test = get_activations("test")
    emb_tr = train["embeddings"]
    emb_te = test["embeddings"]

    y_tr, color_tr, food_tr = make_target_A(train["labels"])
    y_te, color_te, food_te = make_target_A(test["labels"])
    angle_tr = angle_for_subclass(color_tr, food_tr)
    angle_te = angle_for_subclass(color_te, food_te)

    print(f"  Target balance train: {y_tr.mean():.3f}  test: {y_te.mean():.3f}")

    model, hist = train_model(emb_tr, y_tr, emb_te, y_te,
                              aux_fn=circular_aux_loss,
                              aux_inputs_tr=angle_tr, aux_inputs_te=angle_te,
                              epochs=epochs, device=device,
                              model_name="model_A_circular")
    return model, train, test, y_tr, y_te, angle_tr, angle_te


def run_model_B(epochs=150, device="cpu"):
    print("=== Model B: 4-way parity (color XOR food XOR sentiment XOR body_part) ===")
    train = get_activations("train")
    test = get_activations("test")
    emb_tr = train["embeddings"]
    emb_te = test["embeddings"]

    y_tr, _ = make_target_B(train["labels"])
    y_te, _ = make_target_B(test["labels"])
    print(f"  Target balance train: {y_tr.mean():.3f}  test: {y_te.mean():.3f}")

    model, hist = train_model(emb_tr, y_tr, emb_te, y_te,
                              aux_fn=None,
                              epochs=epochs, device=device,
                              model_name="model_B_parity")
    return model, train, test, y_tr, y_te


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["A", "B", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.which in {"A", "both"}:
        run_model_A(epochs=args.epochs, device=args.device)
    if args.which in {"B", "both"}:
        run_model_B(epochs=max(args.epochs, 150), device=args.device)
