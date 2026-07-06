from __future__ import annotations

import numpy as np

from .data import as_dataset


def _path_y(t: np.ndarray, phase: float) -> np.ndarray:
    return 0.55 * np.sin(2.4 * np.pi * t + phase)


def simulate_toy(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-0.8, 0.8)
    coords, tau, state = [], [], []

    t = np.sort(rng.random(190))
    coords.append(np.c_[6 * t - 3, _path_y(t, phase)] + rng.normal(0, 0.10, (len(t), 2)))
    tau.append(t)
    state += [0] * len(t)

    centers = []
    for j, tc in enumerate(np.linspace(0.18, 0.86, 4), 1):
        base = np.array([6 * tc - 3, _path_y(np.array([tc]), phase)[0]])
        center = base + np.array([rng.normal(0, 0.10), ((-1) ** j) * rng.uniform(0.65, 1.0)])
        centers.append((center, tc))

        n_blob = rng.integers(28, 38)
        coords.append(center + rng.normal(0, rng.uniform(0.13, 0.20), (n_blob, 2)))
        tau.append(np.clip(rng.normal(tc, 0.025, n_blob), 0, 1))
        state += [j] * n_blob

        n_bridge = rng.integers(8, 12)
        u = np.linspace(0, 1, n_bridge)[:, None]
        coords.append((1 - u) * base + u * center + rng.normal(0, 0.07, (n_bridge, 2)))
        tau.append(np.clip(rng.normal(tc, 0.04, n_bridge), 0, 1))
        state += [0] * n_bridge

    coords = np.vstack(coords).astype(np.float32)
    tau = np.concatenate(tau)
    state = np.asarray(state)

    signals = []
    for center, _ in centers:
        d2 = ((coords - center) ** 2).sum(1)
        signals.append(np.exp(-d2 / (2 * rng.uniform(0.22, 0.30) ** 2)))
    signals = np.vstack(signals).T

    expr, genes, program = [], [], []
    for j in range(len(centers)):
        for _ in range(10):
            x = 0.08 + rng.uniform(1.8, 2.5) * signals[:, j] + rng.normal(0, 0.18, len(state))
            expr.append(np.clip(x, 0, None))
            genes.append(f"S{j + 1}_{len(genes):03d}")
            program.append(f"S{j + 1}")

    for c in np.linspace(0.03, 0.97, 42):
        x = 0.16 + rng.uniform(1.0, 1.7) * np.exp(-((tau - c) ** 2) / (2 * rng.uniform(0.08, 0.15) ** 2))
        x += rng.normal(0, 0.22, len(state))
        expr.append(np.clip(x, 0, None))
        genes.append(f"B_{len(genes):03d}")
        program.append("band")

    for _ in range(90):
        x = rng.gamma(1.0, 0.22, len(state)) + rng.normal(0, 0.10, len(state))
        expr.append(np.clip(x, 0, None))
        genes.append(f"N_{len(genes):03d}")
        program.append("noise")

    X = np.vstack(expr).T.astype(np.float32)
    dropout = rng.random(X.shape) < (0.06 + 0.15 * np.exp(-X))
    X[dropout] *= 0.05
    labels = np.asarray(["transition" if s == 0 else f"S{s}" for s in state])
    data = as_dataset(X, coords, np.asarray(genes), labels=labels)
    data.update({"program": np.asarray(program), "state": state, "tau": tau, "label_col": "truth", "name": f"toy_{seed}"})
    return data
