from __future__ import annotations

import numpy as np

from .config import StaGPSConfig
from .discovery import discover_modules
from .scoring import graph_gene_scores


class StaGPS:
    def __init__(self, config: StaGPSConfig | None = None):
        self.config = config or StaGPSConfig()
        self.gene_scores_ = None
        self.result_ = None

    def fit(self, data: dict, candidate_idx: np.ndarray | None = None):
        if candidate_idx is None:
            scores = graph_gene_scores(data["X"], data["W"], data["genes"])
            self.gene_scores_ = scores
            n = self.config.candidate_genes
            candidate_idx = scores["idx"].to_numpy() if n is None else scores.head(n)["idx"].to_numpy()

        self.result_ = discover_modules(
            data,
            candidate_idx=candidate_idx,
            threshold=self.config.threshold,
            patch_cut=self.config.patch_cut,
            min_genes=self.config.min_genes,
            max_modules=self.config.max_modules,
            n_eigs=self.config.n_eigs,
        )
        return self

    def fit_predict(self, data: dict, candidate_idx: np.ndarray | None = None) -> dict:
        return self.fit(data, candidate_idx).result_
