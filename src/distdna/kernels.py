"""RBF kernels, exact V-statistic MMD, and calibration utilities."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def pairwise_squared_distances(
    left: NDArray[np.floating], right: NDArray[np.floating]
) -> NDArray[np.floating]:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.ndim != 2 or right_values.ndim != 2:
        raise ValueError("left and right must be two-dimensional")
    if left_values.shape[1] != right_values.shape[1]:
        raise ValueError("left and right must have the same feature dimension")
    distances = (
        np.sum(left_values * left_values, axis=1)[:, None]
        + np.sum(right_values * right_values, axis=1)[None, :]
        - 2.0
        * np.einsum("id,jd->ij", left_values, right_values, optimize=False)
    )
    return np.maximum(distances, 0.0)


def rbf_kernel(
    left: NDArray[np.floating], right: NDArray[np.floating], sigma: float
) -> NDArray[np.floating]:
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a positive finite number")
    return np.exp(-pairwise_squared_distances(left, right) / (2.0 * sigma * sigma))


def exact_mmd2(
    left: NDArray[np.floating], right: NDArray[np.floating], sigma: float
) -> float:
    """Biased (V-statistic) squared MMD, matching manuscript equations 6-7."""

    left_values = np.asarray(left)
    right_values = np.asarray(right)
    if left_values.ndim != 2 or right_values.ndim != 2:
        raise ValueError("left and right samples must have shape [generations, features]")
    if left_values.shape[0] == 0 or right_values.shape[0] == 0:
        raise ValueError("MMD requires at least one sample from each distribution")
    if left_values.shape[1] != right_values.shape[1]:
        raise ValueError("left and right must have the same feature dimension")
    value = (
        float(np.mean(rbf_kernel(left_values, left_values, sigma)))
        + float(np.mean(rbf_kernel(right_values, right_values, sigma)))
        - 2.0 * float(np.mean(rbf_kernel(left_values, right_values, sigma)))
    )
    # The mathematical quantity is non-negative; remove tiny cancellation error.
    return max(value, 0.0)


def exact_mmd_distance_matrix(
    query: NDArray[np.floating],
    reference: NDArray[np.floating],
    sigma: float,
) -> NDArray[np.float64]:
    """Prompt-average exact MMD for tensors shaped ``[models, prompts, R, d]``."""

    query_values = np.asarray(query)
    reference_values = np.asarray(reference)
    if query_values.ndim != 4 or reference_values.ndim != 4:
        raise ValueError("query and reference must have shape [models, prompts, R, d]")
    if query_values.shape[1] != reference_values.shape[1]:
        raise ValueError("query and reference must use the same number of prompts")
    if query_values.shape[-1] != reference_values.shape[-1]:
        raise ValueError("query and reference must use the same feature dimension")

    n_query, n_prompts = query_values.shape[:2]
    n_reference = reference_values.shape[0]
    result = np.empty((n_query, n_reference), dtype=np.float64)
    query_within = np.empty((n_query, n_prompts), dtype=np.float64)
    reference_within = np.empty((n_reference, n_prompts), dtype=np.float64)

    for model_index in range(n_query):
        for prompt_index in range(n_prompts):
            samples = query_values[model_index, prompt_index]
            query_within[model_index, prompt_index] = np.mean(
                rbf_kernel(samples, samples, sigma)
            )
    for model_index in range(n_reference):
        for prompt_index in range(n_prompts):
            samples = reference_values[model_index, prompt_index]
            reference_within[model_index, prompt_index] = np.mean(
                rbf_kernel(samples, samples, sigma)
            )

    for query_index in range(n_query):
        for reference_index in range(n_reference):
            prompt_values = np.empty(n_prompts, dtype=np.float64)
            for prompt_index in range(n_prompts):
                cross = np.mean(
                    rbf_kernel(
                        query_values[query_index, prompt_index],
                        reference_values[reference_index, prompt_index],
                        sigma,
                    )
                )
                prompt_values[prompt_index] = (
                    query_within[query_index, prompt_index]
                    + reference_within[reference_index, prompt_index]
                    - 2.0 * cross
                )
            result[query_index, reference_index] = max(
                float(np.mean(prompt_values)), 0.0
            )
    return result


def median_heuristic(
    embeddings: NDArray[np.floating], max_pairs: int = 100_000, seed: int = 0
) -> float:
    """Estimate the median non-zero Euclidean pair distance on calibration data."""

    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("embeddings must have at least two dimensions")
    values = values.reshape(-1, values.shape[-1])
    if values.shape[0] < 2:
        raise ValueError("bandwidth calibration requires at least two embeddings")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")

    total_pairs = values.shape[0] * (values.shape[0] - 1) // 2
    if total_pairs <= max_pairs:
        squared = pairwise_squared_distances(values, values)
        distances = squared[np.triu_indices(values.shape[0], k=1)]
    else:
        rng = np.random.default_rng(seed)
        left = rng.integers(0, values.shape[0], size=max_pairs)
        right = rng.integers(0, values.shape[0] - 1, size=max_pairs)
        right = right + (right >= left)
        delta = values[left] - values[right]
        distances = np.einsum("ij,ij->i", delta, delta)

    positive = distances[distances > 0]
    if positive.size == 0:
        raise ValueError("median heuristic is undefined when all embeddings are identical")
    sigma = math.sqrt(float(np.median(positive)))
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("median heuristic produced an invalid bandwidth")
    return sigma
