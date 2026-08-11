from pathlib import Path

import numpy as np

from distdna.features import RFFTrace, pairwise_squared_euclidean
from distdna.kernels import exact_mmd_distance_matrix


def test_rfftrace_is_deterministic_nested_and_round_trips(tmp_path: Path) -> None:
    small = RFFTrace(3, 16, sigma=1.2, n_prompts=2, seed=17, projection_dim=5)
    large = RFFTrace(3, 32, sigma=1.2, n_prompts=2, seed=17, projection_dim=None)
    np.testing.assert_array_equal(small.frequencies, large.frequencies[:16])
    np.testing.assert_array_equal(small.phases, large.phases[:16])

    samples = np.random.default_rng(8).normal(size=(4, 2, 6, 3))
    path = tmp_path / "map.npz"
    small.save(path)
    restored = RFFTrace.load(path)
    np.testing.assert_allclose(restored.transform(samples), small.transform(samples))
    assert small.transform(samples).shape == (4, 5)

    compact = RFFTrace(3, 16, sigma=1.2, n_prompts=2, seed=17, projection_dim=3)
    np.testing.assert_allclose(
        compact.projection * np.sqrt(3), small.projection[:3] * np.sqrt(5)
    )
    np.testing.assert_allclose(
        compact.transform(samples) * np.sqrt(3),
        small.transform(samples)[:, :3] * np.sqrt(5),
    )


def test_rff_distance_approximates_exact_mmd() -> None:
    rng = np.random.default_rng(4)
    samples = np.stack(
        [rng.normal(-0.6, 0.5, size=(3, 64, 2)), rng.normal(0.6, 0.5, size=(3, 64, 2))]
    )
    feature_map = RFFTrace(2, 8192, sigma=1.0, n_prompts=3, seed=12)
    vectors = feature_map.transform(samples)
    approximate = pairwise_squared_euclidean(vectors, vectors)
    exact = exact_mmd_distance_matrix(samples, samples, sigma=1.0)
    np.testing.assert_allclose(approximate, exact, atol=0.035)
