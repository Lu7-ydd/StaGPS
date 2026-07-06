from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph

from .preprocessing import dense_block, zscore


def graph_gene_scores(
    X: sparse.spmatrix | np.ndarray,
    W: sparse.csr_matrix,
    genes: np.ndarray,
    base_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    n, g = X.shape
    base = np.arange(g) if base_mask is None else np.flatnonzero(base_mask)
    W = W.tocsr()
    wsum = float(W.sum())
    rows = []
    for start in range(0, len(base), 1500):
        idx = base[start : start + 1500]
        B = dense_block(X, idx)
        mean = B.mean(0)
        expr_frac = (B > 0).mean(0)
        Z = zscore(B, axis=0)
        moran = (n / wsum) * (Z * (W @ Z)).sum(0) / np.maximum((Z * Z).sum(0), 1e-8)
        high_frac = (Z > 1).mean(0)
        peak = np.percentile(Z, 95, axis=0) - np.percentile(Z, 50, axis=0)
        score = np.clip(moran, 0, None) * peak
        ok = (expr_frac > 0.006) & (expr_frac < 0.98) & (high_frac > 0.006) & (high_frac < 0.35) & (mean > 0.005)
        for j, gene_idx in enumerate(idx):
            if ok[j]:
                rows.append((gene_idx, genes[gene_idx], float(score[j]), float(moran[j]), float(expr_frac[j]), float(high_frac[j]), float(mean[j])))
    return pd.DataFrame(rows, columns=["idx", "gene", "graph_score", "moran", "expr_frac", "high_frac", "mean"]).sort_values("graph_score", ascending=False)


def module_stats(score: np.ndarray, W: sparse.csr_matrix, coords: np.ndarray, U: np.ndarray, evals: np.ndarray) -> dict:
    s = zscore(score[:, None])[:, 0]
    high = s > 1.0
    high_n = int(high.sum())
    if high_n < 8:
        return {"patch": 0.0, "high_frac": float(high.mean()), "moran": 0.0, "compact": 0.0, "largest": 0.0, "centroid": 0.0}
    sub = W[high][:, high]
    n_comp, comp = csgraph.connected_components(sub, directed=False)
    largest = np.bincount(comp).max() / high_n if n_comp else 0.0
    incident = float(W[high].sum())
    compact = float(sub.sum()) / incident if incident else 0.0
    x = s - s.mean()
    moran = len(s) / float(W.sum()) * float(x @ W @ x) / max(float(x @ x), 1e-8)
    a = U.T @ x
    centroid = float((evals * a**2).sum() / max(float((a**2).sum()), 1e-8))
    cov = np.cov(coords[high].T)
    vals = np.linalg.eigvalsh(cov + np.eye(2) * 1e-6)
    elong = np.sqrt(vals[-1] / vals[0])
    peak = np.percentile(s, 95) - np.percentile(s, 50)
    patch = max(moran, 0) * compact * largest * peak / (1 + 0.12 * max(elong - 1, 0))
    return {"patch": float(patch), "high_frac": float(high.mean()), "moran": float(moran), "compact": compact, "largest": float(largest), "centroid": centroid}
