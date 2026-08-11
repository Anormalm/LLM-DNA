import numpy as np
import pytest

from distdna.kernels import (
    exact_mmd2,
    exact_mmd_distance_matrix,
    median_heuristic,
    rbf_kernel,
)


def test_rbf_and_exact_mmd_for_point_masses() -> None:
    left = np.asarray([[0.0]])
    right = np.asarray([[1.0]])
    expected_kernel = np.exp(-0.5)
    np.testing.assert_allclose(rbf_kernel(left, right, sigma=1.0), [[expected_kernel]])
    assert exact_mmd2(left, left, sigma=1.0) == 0.0
    assert exact_mmd2(left, right, sigma=1.0) == pytest.approx(2.0 - 2.0 * expected_kernel)


def test_distance_matrix_is_prompt_average() -> None:
    query = np.asarray([[[[0.0]], [[0.0]]]])
    reference = np.asarray([[[[0.0]], [[1.0]]]])
    distances = exact_mmd_distance_matrix(query, reference, sigma=1.0)
    expected = (0.0 + (2.0 - 2.0 * np.exp(-0.5))) / 2.0
    assert distances.shape == (1, 1)
    assert distances[0, 0] == pytest.approx(expected)


def test_median_heuristic_is_deterministic_and_ignores_zero_pairs() -> None:
    points = np.asarray([[0.0], [0.0], [2.0], [4.0]])
    # Non-zero pair distances: 2, 4, 2, 4, 2 -> median 2.
    assert median_heuristic(points, max_pairs=100, seed=3) == pytest.approx(2.0)
    assert median_heuristic(points, max_pairs=3, seed=9) == median_heuristic(
        points, max_pairs=3, seed=9
    )

