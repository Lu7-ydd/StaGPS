from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StaGPSConfig:
    threshold: float = 0.30
    patch_cut: float = 1.5
    min_genes: int = 5
    max_modules: int = 8
    n_eigs: int = 48
    candidate_genes: int | None = 1000
