import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stagps import StaGPS, StaGPSConfig, simulate_toy


def test_toy_discovery():
    data = simulate_toy(seed=7)
    model = StaGPS(StaGPSConfig(candidate_genes=120, threshold=0.30, patch_cut=1.0, min_genes=4, max_modules=6))
    result = model.fit_predict(data)
    assert len(result["steady_score"]) == data["X"].shape[0]
    assert len(result["stable_modules"]) >= 1
    assert len(model.gene_scores_) >= 120


if __name__ == "__main__":
    test_toy_discovery()
    print("smoke ok")
