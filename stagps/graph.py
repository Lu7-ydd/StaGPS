from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

from .preprocessing import dense_block, zscore


def knn_graph(coords: np.ndarray, k: int = 14, metric: str = "euclidean") -> sparse.csr_matrix:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] < 2:
        raise ValueError("coords must be a cells by dimensions matrix")

    k = min(k, coords.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(coords)
    dist, ind = nn.kneighbors(coords)
    rows = np.repeat(np.arange(coords.shape[0]), k)
    cols = ind[:, 1:].ravel()
    sigma = np.median(dist[:, 1:]) + 1e-8
    data = np.exp(-((dist[:, 1:].ravel() / sigma) ** 2))
    W = sparse.coo_matrix((data, (rows, cols)), shape=(coords.shape[0], coords.shape[0])).tocsr()
    return W.maximum(W.T).tocsr()


def graph_fourier(X: sparse.spmatrix | np.ndarray, W: sparse.csr_matrix, n_eigs: int = 48) -> dict:
    Z = zscore(dense_block(X, slice(None)), axis=0)
    L = csgraph.laplacian(W, normed=True).astype(np.float64)
    k = min(n_eigs + 1, W.shape[0] - 2)
    if k < 2:
        raise ValueError("at least 4 cells are required for graph Fourier features")

    if W.shape[0] > 900:
        vals, U = eigsh(L, k=k, which="SM", tol=1e-3, maxiter=6000, v0=np.ones(W.shape[0]))
    else:
        vals, U = np.linalg.eigh(L.toarray())
        vals, U = vals[:k], U[:, :k]

    order = np.argsort(vals)
    vals, U = vals[order], U[:, order]
    coeff = Z.T @ U[:, 1:k]
    coeff = zscore(coeff, axis=1)
    coeff /= np.maximum(np.linalg.norm(coeff, axis=1, keepdims=True), 1e-8)
    return {"Z": Z, "evals": vals[:k], "U": U[:, :k], "coeff": coeff}
