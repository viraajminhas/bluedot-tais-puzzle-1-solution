"""Phase 9: harder attempts to interpret what the country manifold axis encodes.

Tests, in order of "most likely to actually work":
  9a. Hessian of country logit at h2 — should align with v_manifold if it's the
      direction the model's nonlinear decoder uses.
  9b. Decoder direction trace: W5[country] -> through W4 -> back to h2.
  9c. MiniLM "context" embedding (text with country word masked).
  9d. Per-country feature co-occurrence (which other features appear with each).
  9e. Sort country=1 examples by v_manifold projection; print extremes.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression

from common import FEATURE_NAMES, get_activations, load_head, load_split
from phase2b_country import COUNTRY_LIST, tag_countries

COUNTRY_IDX = FEATURE_NAMES.index("country")
MIN_COUNT = 10


def get_v_manifold(X_tr, y_tr):
    Xp = X_tr[y_tr == 1] - X_tr[y_tr == 1].mean(0)
    cov_p = np.cov(Xp.T)
    eigvals, V = np.linalg.eigh(cov_p)
    return V[:, -1], eigvals


def main():
    train = get_activations("train")
    test = get_activations("test")
    texts_tr, _, _ = load_split("train")
    tags_tr = tag_countries(texts_tr)
    X_tr = train["h2"]
    y_tr = train["labels"][:, COUNTRY_IDX]

    v_mf, eigvals_p = get_v_manifold(X_tr, y_tr)
    print(f"v_manifold: top eigval = {eigvals_p[-1]:.4f}, captures "
          f"{eigvals_p[-1] / eigvals_p.sum():.1%} of within-F=1 variance")

    # =================================================================== #
    # 9a. Hessian of country logit at h2
    # =================================================================== #
    print("\n[9a] Hessian of country logit at h2")
    m = load_head("cpu")
    # We want the second derivative of country_logit w.r.t. h2 inputs.
    # Compute it as an average over examples via finite-difference of grads.

    def country_logit_from_h2(h2_batch):
        out = m.layers[6:](h2_batch)
        return out[:, COUNTRY_IDX]

    h2_tensor = torch.from_numpy(X_tr.astype(np.float32))
    # First-order gradient per example (we've seen this averages out)
    h2_tensor.requires_grad_(True)
    logits = country_logit_from_h2(h2_tensor)
    grads = torch.autograd.grad(logits.sum(), h2_tensor)[0].detach().numpy()
    grad_cos = abs(grads.mean(0) @ v_mf) / (np.linalg.norm(grads.mean(0))
                                             * np.linalg.norm(v_mf) + 1e-9)
    print(f"  |cos(mean gradient, v_manifold)| = {grad_cos:.4f} (low, as expected)")

    # Hessian-trace: H = E[grad(grad logit)] for diag elements.
    # Trick: compute the Hessian of country_logit at mean(h2), or estimate
    # via random projections. Cheaper: use Hutchinson trace estimation.
    # For our purposes a finite-difference along v_manifold suffices.
    eps = 0.05
    h2_mean = X_tr.mean(0).astype(np.float32)
    h2_mean_t = torch.from_numpy(h2_mean).unsqueeze(0)
    # Gradient at h2_mean
    h2_mean_t.requires_grad_(True)
    grad_at_mean = torch.autograd.grad(country_logit_from_h2(h2_mean_t).sum(),
                                        h2_mean_t)[0].numpy().ravel()
    # Gradient at h2_mean + eps*v
    def grad_at(point):
        pt = torch.from_numpy(point.astype(np.float32)).unsqueeze(0).requires_grad_(True)
        g = torch.autograd.grad(country_logit_from_h2(pt).sum(), pt)[0]
        return g.numpy().ravel()
    # 2nd derivative along v_manifold:
    # d²f/dv² ≈ (grad(f, h+eps*v) - grad(f, h-eps*v)) · v / (2*eps)
    g_plus = grad_at(h2_mean + eps * v_mf)
    g_minus = grad_at(h2_mean - eps * v_mf)
    d2_v = (g_plus - g_minus) @ v_mf / (2 * eps)
    print(f"  d²(country_logit) / d(v_manifold)² at mean(h2) = {d2_v:.4f}")
    # Compare to a random direction
    rng = np.random.default_rng(0)
    d2_rands = []
    for _ in range(20):
        u = rng.standard_normal(64)
        u /= np.linalg.norm(u)
        gp = grad_at(h2_mean + eps * u)
        gm = grad_at(h2_mean - eps * u)
        d2_rands.append((gp - gm) @ u / (2 * eps))
    print(f"  for random unit directions: mean {np.mean(d2_rands):.4f} "
          f"+/- {np.std(d2_rands):.4f}")
    print(f"  v_manifold ranks at percentile {(np.array(d2_rands) > d2_v).mean()*100:.0f}% "
          f"of random directions (low = v_manifold has more curvature)")

    # Better test: which direction maximizes |d²f/du²|?
    # Sample many random directions, find the one with the biggest 2nd derivative.
    best = (-1, None)
    for _ in range(200):
        u = rng.standard_normal(64)
        u /= np.linalg.norm(u)
        gp = grad_at(h2_mean + eps * u)
        gm = grad_at(h2_mean - eps * u)
        d2 = abs((gp - gm) @ u / (2 * eps))
        if d2 > best[0]:
            best = (d2, u)
    # Compare best with v_manifold
    cos_best_v = abs(best[1] @ v_mf)
    print(f"  best random direction has |d²f| = {best[0]:.4f}")
    print(f"  |cos(best random direction, v_manifold)| = {cos_best_v:.4f}")
    print(f"  |d²f| along v_manifold itself = {abs(d2_v):.4f}")
    if abs(d2_v) > best[0] * 0.5:
        print(f"  -> v_manifold is among the highest-curvature directions")
    else:
        print(f"  -> v_manifold is NOT particularly high-curvature")

    # =================================================================== #
    # 9b. Decoder direction trace
    # =================================================================== #
    print("\n[9b] Country output direction at h3, traced back to h2")
    # Final layer: layers[8] = Linear(64, 8). Its row for country is the
    # h3 direction the model reads from.
    W5 = m.layers[8].weight.detach().numpy()  # (8, 64)
    w_country_h3 = W5[COUNTRY_IDX]              # (64,)
    print(f"  ||w_country at h3|| = {np.linalg.norm(w_country_h3):.3f}")
    # Trace back through W4: h3 = ReLU(W4 @ h2 + b4)
    # The "h2 direction" that maximally drives w_country_h3 in expectation is
    # roughly W4.T @ w_country_h3, but ReLU complicates this.
    # Approximation: average over examples of d(h3) / d(h2) weighted by w_country_h3.
    W4 = m.layers[6].weight.detach().numpy()   # (64, 64)
    b4 = m.layers[6].bias.detach().numpy()      # (64,)
    # For each example: relu activation gate vector
    h2 = torch.from_numpy(X_tr.astype(np.float32))
    with torch.no_grad():
        pre_h3 = (X_tr @ W4.T) + b4
    gate = (pre_h3 > 0).astype(np.float32)
    # Effective h2 direction = average over examples of (W4.T @ (gate_i * w_country_h3))
    # Shape: gate (N,64), w (64,) -> (gate*w) (N,64); sum over N then @ W4 -> (64,)
    eff_pre = (gate * w_country_h3).sum(0) / len(X_tr)  # (64,)
    avg_h2_dir = eff_pre @ W4                              # (64,)
    avg_h2_dir = avg_h2_dir / (np.linalg.norm(avg_h2_dir) + 1e-9)
    cos_decoder_v = float(abs(avg_h2_dir @ v_mf))
    print(f"  |cos(decoder direction at h2, v_manifold)| = {cos_decoder_v:.4f}")
    # Per-class decoder direction
    eff_pos = (gate[y_tr == 1] * w_country_h3).sum(0) / (y_tr == 1).sum()
    avg_pos = eff_pos @ W4
    avg_pos /= np.linalg.norm(avg_pos) + 1e-9
    eff_neg = (gate[y_tr == 0] * w_country_h3).sum(0) / (y_tr == 0).sum()
    avg_neg = eff_neg @ W4
    avg_neg /= np.linalg.norm(avg_neg) + 1e-9
    print(f"  |cos(F=1-conditional decoder, v_manifold)| = {float(abs(avg_pos @ v_mf)):.4f}")
    print(f"  |cos(F=0-conditional decoder, v_manifold)| = {float(abs(avg_neg @ v_mf)):.4f}")
    decoder_diff = avg_pos - avg_neg
    decoder_diff /= np.linalg.norm(decoder_diff) + 1e-9
    print(f"  |cos(F=1 - F=0 decoder direction, v_manifold)| = {float(abs(decoder_diff @ v_mf)):.4f}")

    # =================================================================== #
    # 9c. MiniLM context (text with country word masked)
    # =================================================================== #
    print("\n[9c] Context embedding: text with country word replaced by MASK")
    from sentence_transformers import SentenceTransformer
    masked_texts = []
    valid_idx = []
    for i, (t, tag) in enumerate(zip(texts_tr, tags_tr)):
        if tag is None:
            continue
        masked = t.replace(tag, "[MASK]")
        masked_texts.append(masked)
        valid_idx.append(i)
    print(f"  {len(masked_texts)} country=1 texts with country word masked")
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    ctx_emb = enc.encode(masked_texts, convert_to_numpy=True, show_progress_bar=False)
    # Per-country mean of context embeddings
    valid_tags = [tags_tr[i] for i in valid_idx]
    countries_with_count = Counter(valid_tags)
    keep = [c for c, n in countries_with_count.items() if n >= MIN_COUNT]
    per_country_ctx = {}
    per_country_h2 = {}
    for c in keep:
        mask = np.array([t == c for t in valid_tags])
        per_country_ctx[c] = ctx_emb[mask].mean(0)
        per_country_h2[c] = X_tr[np.array(valid_idx)[mask]].mean(0)
    countries_sorted = sorted(keep)
    ctx_mat = np.stack([per_country_ctx[c] for c in countries_sorted])
    h2_mat = np.stack([per_country_h2[c] for c in countries_sorted])
    mu_h2 = X_tr.mean(0)
    h2_proj_v = (h2_mat - mu_h2) @ v_mf
    # PCA on context embeddings, see if any PC correlates with h2 v_manifold proj
    pca_ctx = PCA(n_components=10).fit(ctx_mat - ctx_mat.mean(0))
    ctx_pcs = pca_ctx.transform(ctx_mat - ctx_mat.mean(0))
    print(f"  cumulative variance from top context PCs: "
          f"top-1 = {pca_ctx.explained_variance_ratio_[0]:.3f}, "
          f"top-5 = {pca_ctx.explained_variance_ratio_[:5].sum():.3f}")
    for k in range(6):
        r = np.corrcoef(h2_proj_v, ctx_pcs[:, k])[0, 1]
        print(f"  Pearson(h2 v_manifold proj per country, ctx-MiniLM PC{k+1}) = {r:+.3f}")
    # Full R² of h2_proj_v predicted by all top ctx PCs
    lr = LinearRegression().fit(ctx_pcs, h2_proj_v)
    r2 = lr.score(ctx_pcs, h2_proj_v)
    print(f"  R^2(h2 v_manifold proj | top 10 ctx PCs) = {r2:.3f}")

    # =================================================================== #
    # 9d. Per-country feature co-occurrence
    # =================================================================== #
    print("\n[9d] Per-country average of other features (co-occurrence pattern)")
    labels_tr = train["labels"]
    per_country_feats = {}
    for c in keep:
        mask = np.array([(t == c) for t in valid_tags])
        idx_in_train = np.array(valid_idx)[mask]
        per_country_feats[c] = labels_tr[idx_in_train].mean(0)
    feat_mat = np.stack([per_country_feats[c] for c in countries_sorted])  # (n, 8)
    print(f"  For each feature, Pearson corr with h2 v_manifold projection per country:")
    for i, name in enumerate(FEATURE_NAMES):
        r = np.corrcoef(h2_proj_v, feat_mat[:, i])[0, 1]
        print(f"    {name:<10}: {r:+.3f}")

    # =================================================================== #
    # 9e. Print extreme texts along v_manifold projection
    # =================================================================== #
    print("\n[9e] Sample texts at extremes of v_manifold projection")
    # Per-example projection (within country=1)
    pos_idx = np.where(y_tr == 1)[0]
    proj_per_ex = (X_tr[pos_idx] - mu_h2) @ v_mf
    # Sort
    order_pos = np.argsort(-proj_per_ex)
    order_neg = np.argsort(proj_per_ex)
    print("\n  Top 10 country=1 texts with HIGHEST v_manifold projection:")
    for j in order_pos[:10]:
        idx = pos_idx[j]
        print(f"    [{proj_per_ex[j]:+.2f}] {texts_tr[idx][:100]}")
    print("\n  Top 10 country=1 texts with LOWEST v_manifold projection:")
    for j in order_neg[:10]:
        idx = pos_idx[j]
        print(f"    [{proj_per_ex[j]:+.2f}] {texts_tr[idx][:100]}")

    # Also: which template_ids land at which end?
    tids_tr = train["template_id"]
    pos_tids = tids_tr[pos_idx]
    print(f"\n  template_id distribution among 10 highest v_manifold projections: "
          f"{Counter(pos_tids[order_pos[:50]])}")
    print(f"  template_id distribution among 10 lowest v_manifold projections: "
          f"{Counter(pos_tids[order_neg[:50]])}")


if __name__ == "__main__":
    main()
