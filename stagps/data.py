from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

from .graph import knn_graph
from .preprocessing import normalize_counts_if_raw


def as_dataset(
    X: sparse.spmatrix | np.ndarray,
    coords: np.ndarray,
    genes: np.ndarray | list[str] | None = None,
    W: sparse.spmatrix | None = None,
    labels: np.ndarray | list[str] | None = None,
) -> dict:
    X = X.tocsr() if sparse.issparse(X) else np.asarray(X, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    if X.shape[0] != coords.shape[0]:
        raise ValueError("X and coords must contain the same number of cells")

    genes = np.asarray(genes if genes is not None else [f"gene_{i}" for i in range(X.shape[1])]).astype(str)
    if len(genes) != X.shape[1]:
        raise ValueError("genes length must match X columns")

    return {
        "X": X,
        "coords": coords,
        "W": W.tocsr() if W is not None else knn_graph(coords),
        "genes": genes,
        "labels": None if labels is None else np.asarray(labels).astype(str),
    }


def _adata_coords(adata, use_rep: str | None = None) -> tuple[np.ndarray, str]:
    if use_rep is not None:
        if use_rep in adata.obsm:
            return np.asarray(adata.obsm[use_rep], dtype=np.float32), use_rep
        if use_rep == "X":
            return np.asarray(adata.X.toarray() if sparse.issparse(adata.X) else adata.X, dtype=np.float32), use_rep
        raise KeyError(f"{use_rep!r} not found in adata.obsm")

    for key in ("X_pca", "X_umap"):
        if key in adata.obsm:
            return np.asarray(adata.obsm[key], dtype=np.float32), key
    return np.asarray(adata.X.toarray() if sparse.issparse(adata.X) else adata.X, dtype=np.float32), "X"


def from_anndata(adata, label_col: str | None = None, use_rep: str | None = None, normalize: bool = True) -> dict:
    coords, coord_key = _adata_coords(adata, use_rep)
    W = adata.obsp["connectivities"].tocsr() if "connectivities" in adata.obsp else knn_graph(coords)
    label_col = label_col or ("clusters" if "clusters" in adata.obs else adata.obs.columns[0] if len(adata.obs.columns) else None)
    labels = None if label_col is None else np.asarray(adata.obs[label_col].astype(str))
    X = normalize_counts_if_raw(adata.X.copy()) if normalize else adata.X.copy()

    data = as_dataset(X, coords, np.asarray(adata.var_names.astype(str)), W, labels)
    data["obs"] = adata.obs.copy()
    data["label_col"] = label_col
    data["coord_key"] = coord_key
    return data


def load_h5ad(path: str | Path, label_col: str | None = None, use_rep: str | None = None, normalize: bool = True) -> dict:
    import scanpy as sc

    return from_anndata(sc.read_h5ad(path), label_col=label_col, use_rep=use_rep, normalize=normalize)
