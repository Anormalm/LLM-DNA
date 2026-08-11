"""Distance baselines and identity-retrieval metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


def pairwise_cosine_distance(
    query: NDArray[np.floating], reference: NDArray[np.floating]
) -> NDArray[np.float64]:
    query_values = np.asarray(query, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    if query_values.ndim != 2 or reference_values.ndim != 2:
        raise ValueError("query and reference must be two-dimensional")
    if query_values.shape[1] != reference_values.shape[1]:
        raise ValueError("query and reference must share a feature dimension")
    query_norm = np.linalg.norm(query_values, axis=1)
    reference_norm = np.linalg.norm(reference_values, axis=1)
    if np.any(query_norm == 0) or np.any(reference_norm == 0):
        raise ValueError("cosine distance is undefined for zero vectors")
    similarities = np.einsum(
        "id,jd->ij", query_values, reference_values, optimize=False
    ) / (query_norm[:, None] * reference_norm[None, :])
    return 1.0 - np.clip(similarities, -1.0, 1.0)


def single_sample_cosine_distance(
    query: NDArray[np.floating], reference: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Cosine distance between prompt-concatenated first generations."""

    if query.ndim != 4 or reference.ndim != 4:
        raise ValueError("inputs must have shape [models, prompts, generations, features]")
    query_vectors = query[:, :, 0, :].reshape(query.shape[0], -1)
    reference_vectors = reference[:, :, 0, :].reshape(reference.shape[0], -1)
    return pairwise_cosine_distance(query_vectors, reference_vectors)


def mean_dna_cosine_distance(
    query: NDArray[np.floating], reference: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Cosine distance between prompt-concatenated generation means."""

    if query.ndim != 4 or reference.ndim != 4:
        raise ValueError("inputs must have shape [models, prompts, generations, features]")
    query_vectors = np.mean(query, axis=2).reshape(query.shape[0], -1)
    reference_vectors = np.mean(reference, axis=2).reshape(reference.shape[0], -1)
    return pairwise_cosine_distance(query_vectors, reference_vectors)


@dataclass(frozen=True)
class RetrievalResult:
    metrics: Dict[str, float]
    ranks: NDArray[np.int64]
    matched_reference_ids: Tuple[str, ...]


def identity_retrieval(
    distances: NDArray[np.floating],
    query_ids: Sequence[str],
    reference_ids: Sequence[str],
    top_ks: Sequence[int] = (1, 3, 5),
) -> RetrievalResult:
    """Rank the unique reference with the same model ID for every query."""

    matrix = np.asarray(distances, dtype=np.float64)
    if matrix.shape != (len(query_ids), len(reference_ids)):
        raise ValueError("distance matrix shape does not match the supplied identifiers")
    if not np.isfinite(matrix).all():
        raise ValueError("distance matrix must contain only finite values")
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("reference_ids must be unique")
    if not top_ks or any(k <= 0 for k in top_ks):
        raise ValueError("top_ks must contain positive integers")

    reference_lookup = {model_id: index for index, model_id in enumerate(reference_ids)}
    missing = [model_id for model_id in query_ids if model_id not in reference_lookup]
    if missing:
        raise ValueError(f"queries have no identity match in references: {sorted(set(missing))}")

    ranks = np.empty(len(query_ids), dtype=np.int64)
    matched = []
    for query_index, model_id in enumerate(query_ids):
        order = np.argsort(matrix[query_index], kind="stable")
        target_index = reference_lookup[model_id]
        ranks[query_index] = int(np.flatnonzero(order == target_index)[0]) + 1
        matched.append(reference_ids[target_index])

    metrics = {f"top_{k}": float(np.mean(ranks <= k)) for k in top_ks}
    metrics["mrr"] = float(np.mean(1.0 / ranks))
    return RetrievalResult(metrics, ranks, tuple(matched))
