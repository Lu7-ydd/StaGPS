# StaGPS

StaGPS is a graph-spectral framework for identifying gene-defined steady states in single-cell trajectories. Instead of treating steady states as visually separated regions in an embedding, StaGPS asks whether a coherent gene program forms a localized, reproducible signal on the cell graph.

The core idea is to move the steady-state definition from cell geometry alone to gene programs on the graph. StaGPS represents each gene as a graph signal, compares genes in a low-frequency graph Fourier basis, groups spectrally similar genes into candidate programs, and retains programs whose expression maps back to compact, autocorrelated cell-state regions.

This is useful when a trajectory is continuous in the embedding but still contains transcriptionally stable attractor-like programs, or when visually apparent clusters need to be checked against gene-level support.

## Workflow

1. build or reuse a cell KNN graph;
2. treat every gene as a graph signal;
3. project genes onto low graph-Laplacian eigenvectors;
4. cluster genes by graph-spectral similarity;
5. score candidate modules by graph autocorrelation, compactness, and connected support;
6. report per-cell steady scores and ranked marker programs.

## Package Layout

- `stagps.data`: dataset construction, Scanpy/AnnData loading, graph-ready inputs.
- `stagps.graph`: KNN graph construction and graph Fourier features.
- `stagps.scoring`: gene graph scores and module compactness statistics.
- `stagps.discovery`: spectral gene-module discovery.
- `stagps.pipeline`: the high-level `StaGPS` estimator.
- `stagps.simulation`: a small toy dataset for examples and smoke tests.

## Install

```bash
python -m pip install -e ".[examples]"
```

`scanpy` is installed as a core dependency so AnnData and `.h5ad` workflows work out of the box.

## Quick Start

```python
from stagps import StaGPS, StaGPSConfig, simulate_toy

data = simulate_toy(seed=7)
model = StaGPS(StaGPSConfig(
    candidate_genes=120,
    threshold=0.30,
    patch_cut=1.0,
    min_genes=4,
    max_modules=6,
))
result = model.fit_predict(data)

print(len(result["stable_modules"]))
print(result["steady_score"][:5])
```

The runnable notebook is at `examples/minimal_example.ipynb`.

## Custom Data

```python
from stagps import StaGPS, as_dataset

data = as_dataset(X, coords, genes=gene_names)
result = StaGPS().fit_predict(data)
```

Inputs:

- `X`: cells by genes expression matrix, dense or scipy sparse.
- `coords`: cell coordinates or features used to build a KNN graph; any numeric dimension is allowed.
- `genes`: gene names. If omitted, names are generated.
- `W`: optional precomputed cell graph. If omitted, `knn_graph(coords)` is used.

Outputs:

- `stable_modules`: retained gene modules with module scores and ranked genes.
- `steady_score`: per-cell score; high means the cell is covered by at least one stable gene module.
- `assignment`: per-cell module assignment, with `0` for unassigned/transition-like cells.
