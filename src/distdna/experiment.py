"""End-to-end, artifact-producing RFFTrace pilot runner."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import ComparisonConfig, ExperimentConfig
from .data import EmbeddingDataset, normalize_embeddings, require_disjoint_prompts
from .features import RFFTrace, pairwise_squared_euclidean
from .kernels import exact_mmd_distance_matrix, median_heuristic
from .metrics import (
    identity_retrieval,
    mean_dna_cosine_distance,
    single_sample_cosine_distance,
)


@dataclass(frozen=True)
class ExperimentResult:
    output_dir: Path
    sigma: float
    metric_rows: int
    rank_rows: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _comparisons(
    requested: Sequence[ComparisonConfig], setting_ids: Sequence[str]
) -> Tuple[ComparisonConfig, ...]:
    if requested:
        comparisons = tuple(requested)
    else:
        comparisons = tuple(
            ComparisonConfig(query=query, reference=reference)
            for query in setting_ids
            for reference in setting_ids
        )
    available = set(setting_ids)
    unknown = sorted(
        {
            setting
            for item in comparisons
            for setting in (item.query, item.reference)
            if setting not in available
        }
    )
    if unknown:
        raise ValueError(f"comparisons refer to unknown settings: {unknown}")
    return comparisons


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path.name}")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_key(value: str) -> str:
    readable = "".join(
        character if character.isalnum() else "_" for character in value
    )[:48]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{readable}_{digest}"


def _method_distances(
    method: str,
    query: NDArray[np.floating],
    reference: NDArray[np.floating],
    sigma: float,
    rff: RFFTrace | None,
) -> NDArray[np.float64]:
    if method == "single_sample_cosine":
        return single_sample_cosine_distance(query, reference)
    if method == "mean_dna_cosine":
        return mean_dna_cosine_distance(query, reference)
    if method == "exact_mmd":
        return exact_mmd_distance_matrix(query, reference, sigma)
    if method == "rfftrace":
        if rff is None:
            raise ValueError("rfftrace requires a fixed RFF map")
        return pairwise_squared_euclidean(rff.transform(query), rff.transform(reference))
    raise ValueError(f"unknown method: {method}")


def validate_experiment(config: ExperimentConfig) -> Tuple[EmbeddingDataset, EmbeddingDataset]:
    evaluation = EmbeddingDataset.load(config.evaluation_path)
    calibration = EmbeddingDataset.load(config.calibration_path)
    require_disjoint_prompts(calibration.prompt_ids, evaluation.prompt_ids)
    if calibration.feature_dim != evaluation.feature_dim:
        raise ValueError("calibration and evaluation feature dimensions must match")
    comparisons = _comparisons(config.comparisons, evaluation.setting_ids)
    has_same_setting = any(item.query == item.reference for item in comparisons)
    required_generations = max(config.generation_counts) * (2 if has_same_setting else 1)
    if required_generations > evaluation.generation_count:
        raise ValueError(
            "insufficient evaluation generations for disjoint query/reference pools: "
            f"required {required_generations}, available {evaluation.generation_count}"
        )
    return evaluation, calibration


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run a pilot and atomically publish its complete artifact directory."""

    evaluation, calibration = validate_experiment(config)
    if config.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a fresh path: {config.output_dir}"
        )
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)

    evaluation_values = normalize_embeddings(evaluation.embeddings, config.normalization)
    calibration_values = normalize_embeddings(calibration.embeddings, config.normalization)
    if config.bandwidth.strategy == "fixed":
        assert config.bandwidth.value is not None
        sigma = config.bandwidth.value * config.bandwidth.multiplier
    else:
        sigma = median_heuristic(
            calibration_values,
            max_pairs=config.bandwidth.max_pairs,
            seed=config.seed,
        ) * config.bandwidth.multiplier

    comparisons = _comparisons(config.comparisons, evaluation.setting_ids)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{config.output_dir.name}-", dir=config.output_dir.parent)
    )
    try:
        parameter_dir = staging / "parameters"
        parameter_dir.mkdir()
        rff_maps: Dict[Tuple[int, int | None], RFFTrace] = {}
        if "rfftrace" in config.methods:
            for dimension in config.rff_dimensions:
                for projection_dimension in config.projection_dimensions:
                    feature_map = RFFTrace(
                        input_dim=evaluation.feature_dim,
                        n_features=dimension,
                        sigma=sigma,
                        n_prompts=len(evaluation.prompt_ids),
                        seed=config.seed,
                        projection_dim=projection_dimension,
                    )
                    projection_suffix = (
                        ""
                        if len(config.projection_dimensions) == 1
                        else f"_L{projection_dimension or 'none'}"
                    )
                    feature_map.save(
                        parameter_dir
                        / (
                            f"rff_D{dimension}{projection_suffix}_seed"
                            f"{config.seed}.npz"
                        )
                    )
                    rff_maps[(dimension, projection_dimension)] = feature_map

        metric_rows: List[Dict[str, Any]] = []
        rank_rows: List[Dict[str, Any]] = []
        distance_payload: Dict[str, NDArray[np.float64]] = {}

        reference_pool_offset = max(config.generation_counts)
        for generations in config.generation_counts:
            primary_cache = {
                setting_id: evaluation_values[
                    :, evaluation.setting_ids.index(setting_id), :, 0:generations, :
                ]
                for setting_id in evaluation.setting_ids
            }
            disjoint_reference_cache = {
                setting_id: evaluation_values[
                    :,
                    evaluation.setting_ids.index(setting_id),
                    :,
                    reference_pool_offset : reference_pool_offset + generations,
                    :,
                ]
                for setting_id in evaluation.setting_ids
            }
            for method in config.methods:
                variants: Iterable[Tuple[int | None, int | None]] = (
                    tuple(rff_maps) if method == "rfftrace" else ((None, None),)
                )
                for dimension, projection_dimension in variants:
                    feature_map = (
                        None
                        if dimension is None
                        else rff_maps[(dimension, projection_dimension)]
                    )
                    rff_vector_cache = (
                        {
                            (setting_id, pool): feature_map.transform(samples)
                            for setting_id in evaluation.setting_ids
                            for pool, samples in (
                                ("primary", primary_cache[setting_id]),
                                ("disjoint_reference", disjoint_reference_cache[setting_id]),
                            )
                        }
                        if feature_map is not None
                        else {}
                    )
                    for comparison in comparisons:
                        query = primary_cache[comparison.query]
                        is_same_setting = comparison.query == comparison.reference
                        reference = (
                            disjoint_reference_cache[comparison.reference]
                            if is_same_setting
                            else primary_cache[comparison.reference]
                        )
                        if method == "rfftrace":
                            distances = pairwise_squared_euclidean(
                                rff_vector_cache[(comparison.query, "primary")],
                                rff_vector_cache[
                                    (
                                        comparison.reference,
                                        "disjoint_reference"
                                        if is_same_setting
                                        else "primary",
                                    )
                                ],
                            )
                        else:
                            distances = _method_distances(
                                method, query, reference, sigma, feature_map
                            )
                        retrieval = identity_retrieval(
                            distances,
                            evaluation.model_ids,
                            evaluation.model_ids,
                            config.top_ks,
                        )
                        run_key = (
                            f"{method}__R{generations}__D{dimension or 'na'}"
                            f"__L{projection_dimension or 'na'}"
                            f"__q_{_safe_key(comparison.query)}"
                            f"__ref_{_safe_key(comparison.reference)}"
                        )
                        row: Dict[str, Any] = {
                            "run_key": run_key,
                            "method": method,
                            "generations": generations,
                            "rff_dimension": "NA" if dimension is None else dimension,
                            "projection_dimension": (
                                "NA"
                                if method != "rfftrace" or projection_dimension is None
                                else projection_dimension
                            ),
                            "query_setting": comparison.query,
                            "reference_setting": comparison.reference,
                            "comparison_type": (
                                "same_setting"
                                if comparison.query == comparison.reference
                                else "cross_setting"
                            ),
                            "sigma": sigma,
                            "normalization": config.normalization,
                            "seed": config.seed,
                            "n_queries": len(evaluation.model_ids),
                        }
                        row.update(retrieval.metrics)
                        metric_rows.append(row)
                        for query_index, model_id in enumerate(evaluation.model_ids):
                            rank_rows.append(
                                {
                                    "run_key": run_key,
                                    "method": method,
                                    "generations": generations,
                                    "rff_dimension": "NA" if dimension is None else dimension,
                                    "projection_dimension": (
                                        "NA"
                                        if method != "rfftrace"
                                        or projection_dimension is None
                                        else projection_dimension
                                    ),
                                    "query_setting": comparison.query,
                                    "reference_setting": comparison.reference,
                                    "query_model_id": model_id,
                                    "matched_reference_id": retrieval.matched_reference_ids[
                                        query_index
                                    ],
                                    "rank": int(retrieval.ranks[query_index]),
                                }
                            )
                        if config.save_distances:
                            distance_payload[run_key] = distances

        _write_csv(staging / "metrics.csv", metric_rows)
        _write_csv(staging / "ranks.csv", rank_rows)
        if config.save_distances:
            np.savez_compressed(staging / "distances.npz", **distance_payload)

        metadata = {
            "format_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sigma": sigma,
            "bandwidth_selection_split": "calibration",
            "same_setting_evaluation": {
                "policy": "disjoint_generation_pools",
                "query_pool": "generation indices [0, R)",
                "reference_pool": (
                    "generation indices [max(generation_counts), "
                    "max(generation_counts) + R)"
                ),
            },
            "na_policy": (
                "NA is used only for structurally inapplicable dimensions: non-RFF methods "
                "have no RFF dimension; unprojected methods have no projection dimension. "
                "Missing measurements are rejected during validation."
            ),
            "config": config.as_dict(),
            "inputs": {
                "evaluation": {
                    "sha256": _sha256(config.evaluation_path),
                    "shape": list(evaluation.shape),
                    "model_ids": list(evaluation.model_ids),
                    "setting_ids": list(evaluation.setting_ids),
                    "prompt_ids": list(evaluation.prompt_ids),
                },
                "calibration": {
                    "sha256": _sha256(config.calibration_path),
                    "shape": list(calibration.shape),
                    "model_ids": list(calibration.model_ids),
                    "setting_ids": list(calibration.setting_ids),
                    "prompt_ids": list(calibration.prompt_ids),
                },
            },
            "artifact_counts": {
                "metric_rows": len(metric_rows),
                "rank_rows": len(rank_rows),
                "distance_matrices": len(distance_payload),
            },
        }
        with (staging / "metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")

        os.replace(staging, config.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return ExperimentResult(
        output_dir=config.output_dir,
        sigma=sigma,
        metric_rows=len(metric_rows),
        rank_rows=len(rank_rows),
    )
