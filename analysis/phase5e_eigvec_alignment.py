"""Phase 5e: do the two class covariances share eigenvectors?

If cov(F=1) and cov(F=0) have the *same* eigenvectors but different
eigenvalues, then F=1 is literally a scaled-down version of F=0 living
in the same subspace. If they are misaligned, F=1 lives in a tilted
sub-cone.

Test:
  1. Compute principal subspaces (top-K eigvectors) of each class.
  2. Subspace angle = arccos(min singular value of V1^T V0).
  3. Per-eigenvector best-match cosine.
"""
from __future__ import annotations

import numpy as np

from common import FEATURE_NAMES, get_activations

COUNTRY_IDX = FEATURE_NAMES.index("country")


def main():
    train = get_activations("train")
    X = train["h2"]
    y = train["labels"][:, COUNTRY_IDX]

    Xp = X[y == 1] - X[y == 1].mean(0)
    Xn = X[y == 0] - X[y == 0].mean(0)
    cov_p = np.cov(Xp.T)
    cov_n = np.cov(Xn.T)
    ep, Vp = np.linalg.eigh(cov_p)
    en, Vn = np.linalg.eigh(cov_n)
    ep, Vp = ep[::-1], Vp[:, ::-1]
    en, Vn = en[::-1], Vn[:, ::-1]

    print("Per-eigenvector best matches (top-10):")
    for k in range(10):
        cosines = np.abs(Vp[:, k] @ Vn[:, :20])   # cos with top-20 of cov(F=0)
        j = int(np.argmax(cosines))
        print(f"  V_F=1[{k}] best matches V_F=0[{j}] with |cos| = {cosines[j]:.3f}  "
              f"(eigvals: F=1 {ep[k]:.3f}, F=0 {en[j]:.3f})")

    print("\nSubspace principal angles (top-K each side):")
    for K in [2, 3, 5, 8]:
        # Subspace angles via SVD of V_p^T V_n
        s = np.linalg.svd(Vp[:, :K].T @ Vn[:, :K], compute_uv=False)
        s = np.clip(s, 0, 1)
        angles_deg = np.degrees(np.arccos(s))
        print(f"  K={K}: principal angles (deg) = "
              f"[{', '.join(f'{a:.1f}' for a in angles_deg)}]")
        print(f"     subspace overlap (sum of squared cosines) = {(s**2).sum():.2f} / {K}")

    print("\nEigenvalue ratio cov(F=0) / cov(F=1) (top 10):")
    for k in range(10):
        # Match F=1 eigvec k to its best F=0 partner; eigvalue ratio along that
        # axis tells how much F=0 spreads relative to F=1.
        cosines = np.abs(Vp[:, k] @ Vn[:, :20])
        j = int(np.argmax(cosines))
        ratio = en[j] / (ep[k] + 1e-9)
        print(f"  axis k={k}:  F=0 eigval / F=1 eigval ≈ {ratio:.2f}x  (along F=1's PC{k})")

    # Crucial test: shrinkage along v_manifold (PC1 of F=1)
    v_mf = Vp[:, 0]
    spread_p = np.var(Xp @ v_mf)
    spread_n = np.var(Xn @ v_mf)
    print(f"\nAlong v_manifold (top-1 eigvec of cov(F=1)):")
    print(f"  Var(F=1 projected on v_mf) = {spread_p:.4f}")
    print(f"  Var(F=0 projected on v_mf) = {spread_n:.4f}")
    print(f"  ratio: F=0 / F=1 = {spread_n/spread_p:.2f}x")


if __name__ == "__main__":
    main()
