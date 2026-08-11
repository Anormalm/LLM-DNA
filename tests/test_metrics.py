import numpy as np
import pytest

from distdna.metrics import identity_retrieval, mean_dna_cosine_distance


def test_identity_retrieval_metrics_and_ranks() -> None:
    distances = np.asarray(
        [
            [0.4, 0.1, 0.2],
            [0.1, 0.2, 0.3],
            [0.3, 0.2, 0.1],
        ]
    )
    result = identity_retrieval(distances, ("a", "b", "c"), ("a", "b", "c"))
    np.testing.assert_array_equal(result.ranks, [3, 2, 1])
    assert result.metrics["top_1"] == pytest.approx(1 / 3)
    assert result.metrics["top_3"] == 1.0
    assert result.metrics["mrr"] == pytest.approx((1 / 3 + 1 / 2 + 1) / 3)


def test_mean_dna_uses_all_generations() -> None:
    query = np.asarray([[[[1.0, 0.0], [1.0, 0.0]]]])
    reference = np.asarray(
        [
            [[[1.0, 0.0], [1.0, 0.0]]],
            [[[0.0, 1.0], [0.0, 1.0]]],
        ]
    )
    distances = mean_dna_cosine_distance(query, reference)
    np.testing.assert_allclose(distances, [[0.0, 1.0]])

