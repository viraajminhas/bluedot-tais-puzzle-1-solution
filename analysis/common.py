"""Shared utilities: load model + data, extract layer activations, cache to disk."""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
CACHE = REPO / "analysis" / ".cache"
CACHE.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = json.loads((REPO / "feature_names.json").read_text())
N_FEATURES = len(FEATURE_NAMES)  # 8


class Head(nn.Module):
    """Same architecture as puzzle.ipynb. 5-layer MLP, ReLUs between."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),  # 0,1 -> hidden 0 post-ReLU at idx 2
            nn.Linear(64, 64),  nn.ReLU(),  # 2,3 -> hidden 1 post-ReLU at idx 4
            nn.Linear(64, 64),  nn.ReLU(),  # 4,5 -> hidden 2 post-ReLU at idx 6  ← layer L
            nn.Linear(64, 64),  nn.ReLU(),  # 6,7 -> hidden 3 post-ReLU at idx 8
            nn.Linear(64, 8),               # 8   -> logits
        )

    def forward(self, x):
        return self.layers(x)


# Slice-END indices to take `layers[:end]` for post-ReLU outputs of hidden 0,1,2,3.
# Layers (Sequential idx): 0 Linear, 1 ReLU=h0, 2 Linear, 3 ReLU=h1, 4 Linear,
# 5 ReLU=h2 ← layer L, 6 Linear, 7 ReLU=h3, 8 Linear → logits.
POST_RELU_END = {0: 2, 1: 4, 2: 6, 3: 8}


def load_head(device: str | torch.device = "cpu") -> Head:
    m = Head()
    state = torch.load(REPO / "model.pt", map_location="cpu", weights_only=False)
    m.load_state_dict(state)
    m.eval()
    return m.to(device)


def load_split(split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Returns (texts, labels[N,8], template_ids[N])."""
    assert split in {"train", "test"}
    path = DATA / f"{split}.jsonl"
    texts, labels, tids = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        texts.append(ex["text"])
        labels.append(ex["labels"])
        tids.append(ex.get("template_id", -1))
    return texts, np.array(labels, dtype=np.int64), np.array(tids, dtype=np.int64)


def _embed(texts: list[str], device: str = "cpu", batch_size: int = 256) -> np.ndarray:
    """Mean-pooled MiniLM embeddings [N, 384]."""
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    return enc.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)


def get_embeddings(split: str, device: str = "cpu") -> np.ndarray:
    """Cached mean-pooled encoder embeddings, shape [N, 384]."""
    cache_path = CACHE / f"emb_{split}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    texts, _, _ = load_split(split)
    emb = _embed(texts, device=device).astype(np.float32)
    np.save(cache_path, emb)
    return emb


def get_activations(split: str, device: str = "cpu") -> dict[str, np.ndarray]:
    """Cached post-ReLU activations at hidden 0,1,2,3 plus logits.

    Returns: {'h0', 'h1', 'h2', 'h3', 'logits', 'labels', 'template_id', 'embeddings'}
    """
    cache_path = CACHE / f"acts_{split}.npz"
    if cache_path.exists():
        z = np.load(cache_path)
        return {k: z[k] for k in z.files}

    emb = get_embeddings(split, device=device)
    _, labels, tids = load_split(split)

    m = load_head("cpu")
    x = torch.from_numpy(emb)
    outs = {"embeddings": emb, "labels": labels, "template_id": tids}
    with torch.no_grad():
        for h, end in POST_RELU_END.items():
            outs[f"h{h}"] = m.layers[:end](x).numpy().astype(np.float32)
        outs["logits"] = m(x).numpy().astype(np.float32)
    np.savez(cache_path, **outs)
    return outs


def model_accuracy(split: str = "test") -> dict[str, float]:
    """Sanity check: per-feature accuracy of the puzzle model itself."""
    data = get_activations(split)
    probs = 1 / (1 + np.exp(-data["logits"]))
    pred = (probs > 0.5).astype(np.int64)
    return {FEATURE_NAMES[i]: (pred[:, i] == data["labels"][:, i]).mean()
            for i in range(N_FEATURES)}


if __name__ == "__main__":
    print("Loading splits and extracting activations...")
    for split in ["train", "test"]:
        d = get_activations(split)
        print(f"  {split}: N={len(d['labels'])} emb={d['embeddings'].shape} "
              f"h0={d['h0'].shape} h1={d['h1'].shape} h2={d['h2'].shape} "
              f"h3={d['h3'].shape} logits={d['logits'].shape}")
    print("\nModel test accuracy per feature:")
    for name, acc in model_accuracy("test").items():
        print(f"  {name:10s} {acc:.4f}")
