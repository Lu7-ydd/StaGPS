from __future__ import annotations

import numpy as np
from scipy import sparse


def zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    return (x - x.mean(axis=axis, keepdims=True)) / np.maximum(x.std(axis=axis, keepdims=True), 1e-8)


def dense_block(X: sparse.spmatrix | np.ndarray, cols: np.ndarray | slice) -> np.ndarray:
    block = X[:, cols]
    return block.toarray().astype(np.float32, copy=False) if sparse.issparse(block) else np.asarray(block, dtype=np.float32)


def normalize_counts_if_raw(X: sparse.spmatrix | np.ndarray, target_sum: float = 1e4) -> sparse.csr_matrix | np.ndarray:
    if sparse.issparse(X):
        X = X.tocsr().astype(np.float32)
        if len(X.data) and np.nanmax(X.data) > 30:
            lib = np.asarray(X.sum(1)).ravel()
            X = X.multiply((target_sum / np.maximum(lib, 1e-8))[:, None]).tocsr()
            X.data = np.log1p(X.data)
        return X

    X = np.asarray(X, dtype=np.float32)
    if np.nanmax(X) > 30:
        lib = X.sum(1, keepdims=True)
        X = np.log1p(X * (target_sum / np.maximum(lib, 1e-8)))
    return X
