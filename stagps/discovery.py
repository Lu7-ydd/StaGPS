from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .graph import graph_fourier
from .preprocessing import dense_block, zscore
from .scoring import module_stats


def _empty_result(data: dict, candidate_idx: np.ndarray, gf: dict | None = None) -> dict:
    n_cells = data["X"].shape[0]
    return {
        "candidate_idx": candidate_idx,
        "gf": gf,
        "all_modules": [],
        "stable_modules": [],
        "module_scores": np.zeros((n_cells, 0)),
        "steady_score": np.zeros(n_cells),
        "assignment": np.zeros(n_cells, dtype=int),
    }


def discover_modules(
    data: dict,
    candidate_idx: np.ndarray | None = None,
    threshold: float = 0.30,
    patch_cut: float = 1.5,
    min_genes: int = 5,
    max_modules: int = 8,
    n_eigs: int = 48,
) -> dict:
    idx = np.arange(data["X"].shape[1]) if candidate_idx is None else np.asarray(candidate_idx, dtype=int)
    if len(idx) < max(2, min_genes):
        return _empty_result(data, idx)

    gf = graph_fourier(dense_block(data["X"], idx), data["W"], n_eigs=n_eigs)
    labels = fcluster(linkage(pdist(gf["coeff"], metric="cosine"), method="average"), t=threshold, criterion="distance")
    modules = []
    for lab in np.unique(labels):
        local = np.flatnonzero(labels == lab)
        if len(local) < min_genes:
            continue
        score = gf["Z"][:, local].mean(1)
        if abs(np.percentile(score, 5)) > abs(np.percentile(score, 95)):
            score *= -1
        stats = module_stats(score, data["W"], data["coords"], gf["U"], gf["evals"])
        module_score = zscore(score[:, None])[:, 0]
        cor = (gf["Z"][:, local].T @ module_score) / len(score)
        order = np.argsort(-cor)
        modules.append({
            "local_genes": local,
            "genes": idx[local],
            "score": module_score,
            "local_corr": cor,
            "ranked_genes": idx[local][order],
            "ranked_corr": cor[order],
            **stats,
        })

    stable = [m for m in modules if m["patch"] >= patch_cut and 0.01 <= m["high_frac"] <= 0.35 and m["largest"] > 0.35]
    stable = sorted(stable, key=lambda m: m["patch"], reverse=True)[:max_modules]
    if not stable:
        return {**_empty_result(data, idx, gf), "all_modules": modules}

    scores = np.vstack([m["score"] for m in stable]).T
    steady = scores.max(1)
    assignment = scores.argmax(1) + 1
    assignment[steady < 0.35] = 0
    return {
        "candidate_idx": idx,
        "gf": gf,
        "all_modules": modules,
        "stable_modules": stable,
        "module_scores": scores,
        "steady_score": steady,
        "assignment": assignment,
    }
