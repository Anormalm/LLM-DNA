"""Relationship diagnostics over saved model-to-model distance matrices."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .data import CollectionManifest


_METHODS = (
    "single_sample_cosine",
    "mean_dna_cosine",
    "exact_mmd",
    "rfftrace",
)
_METRICS = (
    "roc_auc",
    "average_precision",
    "nearest_group_accuracy",
    "related_distance_mean",
    "unrelated_distance_mean",
    "distance_gap",
)


def _load_groups(path: str | Path) -> Dict[str, str]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("relationship map must be a format_version 1 JSON object")
    groups = payload.get("groups")
    if (
        not isinstance(groups, dict)
        or not groups
        or any(not isinstance(key, str) or not key for key in groups)
        or any(not isinstance(value, str) or not value for value in groups.values())
    ):
        raise ValueError("relationship map groups must map model IDs to group labels")
    return dict(groups)


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        raise ValueError("relationship AUROC requires positive and negative pairs")
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold-grouped AP whose result is invariant to ordering within ties."""

    positives = int(np.sum(labels))
    if not positives or positives == len(labels):
        raise ValueError("relationship AP requires positive and negative pairs")
    thresholds = np.unique(scores)[::-1]
    true_positives = 0
    retrieved = 0
    previous_recall = 0.0
    result = 0.0
    for threshold in thresholds:
        selected = scores == threshold
        true_positives += int(np.sum(labels[selected]))
        retrieved += int(np.sum(selected))
        recall = true_positives / positives
        result += (recall - previous_recall) * (true_positives / retrieved)
        previous_recall = recall
    return float(result)


def _matrix_metrics(
    matrix: np.ndarray, model_ids: Sequence[str], groups: Mapping[str, str]
) -> Dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    size = len(model_ids)
    if values.shape != (size, size) or not np.all(np.isfinite(values)):
        raise ValueError("distance matrix must be finite and match the model roster")
    related: List[bool] = []
    distances: List[float] = []
    for query_index, query_id in enumerate(model_ids):
        for reference_index, reference_id in enumerate(model_ids):
            if query_index == reference_index:
                continue
            related.append(groups[query_id] == groups[reference_id])
            distances.append(float(values[query_index, reference_index]))
    labels = np.asarray(related, dtype=bool)
    raw_distances = np.asarray(distances, dtype=np.float64)
    scores = -raw_distances

    eligible_hits = []
    eligible_chances = []
    for query_index, query_id in enumerate(model_ids):
        group_size = sum(groups[item] == groups[query_id] for item in model_ids)
        if group_size < 2:
            continue
        candidates = values[query_index].copy()
        candidates[query_index] = np.inf
        nearest = int(np.argmin(candidates))
        eligible_hits.append(groups[query_id] == groups[model_ids[nearest]])
        eligible_chances.append((group_size - 1) / (size - 1))
    if not eligible_hits:
        raise ValueError("relationship map has no group with at least two models")

    related_mean = float(np.mean(raw_distances[labels]))
    unrelated_mean = float(np.mean(raw_distances[~labels]))
    return {
        "roc_auc": _roc_auc(labels, scores),
        "average_precision": _average_precision(labels, scores),
        "nearest_group_accuracy": float(np.mean(eligible_hits)),
        "nearest_group_random_baseline": float(np.mean(eligible_chances)),
        "nearest_group_eligible_models": len(eligible_hits),
        "related_pairs": int(np.sum(labels)),
        "unrelated_pairs": int(np.sum(~labels)),
        "related_distance_mean": related_mean,
        "unrelated_distance_mean": unrelated_mean,
        "distance_gap": unrelated_mean - related_mean,
    }


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.mean(array)),
        float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    )


def _selected_row(
    row: Mapping[str, str],
    *,
    generations: int,
    rff_dimension: int,
    projection_dimension: int | None,
) -> bool:
    if int(row["generations"]) != generations:
        return False
    if row["method"] == "rfftrace":
        return (
            int(row["rff_dimension"]) == rff_dimension
            and (
                None
                if row["projection_dimension"] == "NA"
                else int(row["projection_dimension"])
            )
            == projection_dimension
        )
    return row["rff_dimension"] == "NA" and row["projection_dimension"] == "NA"


def build_relationship_report(
    output_dirs: Sequence[str | Path],
    manifest_path: str | Path,
    relationship_map_path: str | Path,
    *,
    normalization: str = "l2",
    bandwidth_multiplier: float = 1.0,
    generations: int = 4,
    rff_dimension: int = 512,
    projection_dimension: int | None = None,
) -> Dict[str, Any]:
    if not output_dirs:
        raise ValueError("relationship report requires at least one experiment output")
    manifest = CollectionManifest.load(manifest_path)
    groups = _load_groups(relationship_map_path)
    missing = sorted(set(manifest.model_ids).difference(groups))
    extra = sorted(set(groups).difference(manifest.model_ids))
    if missing or extra:
        raise ValueError(
            "relationship map must exactly match the manifest roster; "
            f"missing={missing}, extra={extra}"
        )
    if len({groups[item] for item in manifest.model_ids}) < 2:
        raise ValueError("relationship map must contain at least two groups")
    deterministic = [item for item in manifest.settings if item.temperature == 0.0]
    if len(deterministic) != 1:
        raise ValueError("relationship report requires exactly one deterministic setting")
    deterministic_id = deterministic[0].setting_id

    observations: List[Dict[str, Any]] = []
    model_order: Tuple[str, ...] | None = None
    observed_seeds = set()
    expected_per_run_method = len(manifest.settings) * 2 - 1
    for raw_dir in output_dirs:
        output_dir = Path(raw_dir).resolve()
        with (output_dir / "metadata.json").open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        experiment = metadata["config"]["experiment"]
        if experiment["normalization"] != normalization or not np.isclose(
            float(experiment["bandwidth"]["multiplier"]), bandwidth_multiplier
        ):
            raise ValueError(f"experiment protocol does not match requested factors: {output_dir}")
        current_order = tuple(metadata["inputs"]["evaluation"]["model_ids"])
        if current_order != manifest.model_ids:
            raise ValueError(f"experiment model order does not match manifest: {output_dir}")
        if model_order is None:
            model_order = current_order
        elif current_order != model_order:
            raise ValueError("experiment outputs use inconsistent model order")
        seed = int(experiment["seed"])
        if seed in observed_seeds:
            raise ValueError(f"duplicate experiment seed in relationship inputs: {seed}")
        observed_seeds.add(seed)

        with (output_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        selected = [
            row
            for row in rows
            if _selected_row(
                row,
                generations=generations,
                rff_dimension=rff_dimension,
                projection_dimension=projection_dimension,
            )
            and (
                row["query_setting"] == row["reference_setting"]
                or (
                    row["reference_setting"] == deterministic_id
                    and row["query_setting"] != deterministic_id
                )
            )
        ]
        counts = {method: sum(row["method"] == method for row in selected) for method in _METHODS}
        incomplete = {
            method: count
            for method, count in counts.items()
            if count != expected_per_run_method
        }
        if incomplete:
            raise ValueError(
                f"incomplete relationship grid in {output_dir}: expected "
                f"{expected_per_run_method} cells per method, observed {incomplete}"
            )
        with np.load(output_dir / "distances.npz", allow_pickle=False) as matrices:
            for row in selected:
                query = row["query_setting"]
                reference = row["reference_setting"]
                metrics = _matrix_metrics(matrices[row["run_key"]], current_order, groups)
                observations.append(
                    {
                        "output_dir": str(output_dir),
                        "seed": seed,
                        "method": row["method"],
                        "evaluation": (
                            "same_setting"
                            if query == reference
                            else "cross_to_deterministic"
                        ),
                        "query_setting": query,
                        "reference_setting": reference,
                        **metrics,
                    }
                )

    cell_groups: Dict[Tuple[str, str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        cell_groups[
            (row["method"], row["evaluation"], row["query_setting"], row["reference_setting"])
        ].append(row)
    cells = []
    for key, rows in sorted(cell_groups.items()):
        method, evaluation, query, reference = key
        item: Dict[str, Any] = {
            "method": method,
            "evaluation": evaluation,
            "query_setting": query,
            "reference_setting": reference,
            "runs": len(rows),
            "related_pairs_per_run": rows[0]["related_pairs"],
            "unrelated_pairs_per_run": rows[0]["unrelated_pairs"],
            "nearest_group_eligible_models": rows[0]["nearest_group_eligible_models"],
        }
        for metric in _METRICS:
            mean, std = _mean_std([float(row[metric]) for row in rows])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        cells.append(item)

    per_seed_summary: Dict[Tuple[int, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        per_seed_summary[(row["seed"], row["method"], row["evaluation"])].append(row)
    seed_rows = []
    for (seed, method, evaluation), rows in sorted(per_seed_summary.items()):
        item: Dict[str, Any] = {"seed": seed, "method": method, "evaluation": evaluation}
        for metric in _METRICS:
            item[metric] = float(np.mean([float(row[metric]) for row in rows]))
        seed_rows.append(item)
    summary_groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        summary_groups[(row["method"], row["evaluation"])].append(row)
    summary = []
    for (method, evaluation), rows in sorted(summary_groups.items()):
        item = {"method": method, "evaluation": evaluation, "runs": len(rows)}
        for metric in _METRICS:
            mean, std = _mean_std([float(row[metric]) for row in rows])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        summary.append(item)

    return {
        "format_version": 1,
        "evidence_status": (
            "diagnostic family-label analysis; group membership is a coarse proxy for relationship"
        ),
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_fingerprint": manifest.fingerprint,
        "relationship_map": str(Path(relationship_map_path).resolve()),
        "groups": {model_id: groups[model_id] for model_id in manifest.model_ids},
        "protocol": {
            "normalization": normalization,
            "bandwidth_multiplier": bandwidth_multiplier,
            "generations": generations,
            "rff_dimension": rff_dimension,
            "projection_dimension": projection_dimension,
            "deterministic_reference": deterministic_id,
            "self_pairs_excluded": True,
            "nearest_group_excludes_singleton_groups": True,
        },
        "chance_baselines": {
            "roc_auc": 0.5,
            "nearest_group_accuracy": observations[0][
                "nearest_group_random_baseline"
            ],
        },
        "run_count": len(output_dirs),
        "seeds": sorted(observed_seeds),
        "cell_count": len(cells),
        "cells": cells,
        "summary": summary,
    }
