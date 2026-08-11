"""Post-run summaries for retrieval and RFF approximation quality."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


def _load_metrics(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"metrics table is empty: {path}")
    return rows


def _weighted_mean(rows: Sequence[Mapping[str, str]], field: str) -> float:
    weights = np.asarray([float(row["n_queries"]) for row in rows], dtype=np.float64)
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return float(np.sum(weights * values) / np.sum(weights))


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = math.sqrt(
        float(np.sum(left_centered * left_centered))
        * float(np.sum(right_centered * right_centered))
    )
    if denominator == 0.0:
        return None
    return float(np.sum(left_centered * right_centered) / denominator)


def _pilot_checks(
    retrieval: Sequence[Mapping[str, Any]],
    approximation: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    required_methods = {
        "single_sample_cosine",
        "mean_dna_cosine",
        "exact_mmd",
        "rfftrace",
    }
    observed_methods = {str(item["method"]) for item in retrieval}
    comparison_types = {str(item["comparison_type"]) for item in retrieval}
    generation_counts = sorted({int(item["generations"]) for item in retrieval})
    top_1_values = [
        float(item["top_1"]) for item in retrieval if "top_1" in item
    ]
    non_degenerate_approximations = sum(
        item["distance_correlation"] is not None for item in approximation
    )
    checks = {
        "all_methods_present": required_methods.issubset(observed_methods),
        "same_and_cross_setting_present": {
            "same_setting",
            "cross_setting",
        }.issubset(comparison_types),
        "generation_sweep_present": len(generation_counts) >= 2,
        "non_ceiling_retrieval_observed": any(
            value < 1.0 - 1e-12 for value in top_1_values
        ),
        "nondegenerate_rff_exact_diagnostics_present": (
            non_degenerate_approximations > 0
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "ready_for_multi_seed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "observed_methods": sorted(observed_methods),
        "comparison_types": sorted(comparison_types),
        "generation_counts": generation_counts,
        "top_1_min": min(top_1_values) if top_1_values else None,
        "top_1_max": max(top_1_values) if top_1_values else None,
        "nondegenerate_approximation_rows": non_degenerate_approximations,
        "interpretation": (
            "These are protocol and ceiling-effect gates, not evidence that RFFTrace "
            "outperforms a baseline."
        ),
    }


def summarize_run(output_dir: str | Path) -> Dict[str, Any]:
    source = Path(output_dir).resolve()
    metrics_path = source / "metrics.csv"
    metadata_path = source / "metadata.json"
    distances_path = source / "distances.npz"
    if not metrics_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"expected metrics.csv and metadata.json in experiment output: {source}"
        )
    rows = _load_metrics(metrics_path)
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)

    retrieval_groups: Dict[Tuple[str, str, str, str], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["method"],
            row["generations"],
            row["rff_dimension"],
            row["comparison_type"],
        )
        retrieval_groups[key].append(row)

    metric_fields = [field for field in rows[0] if field.startswith("top_")]
    metric_fields.append("mrr")
    retrieval = []
    for key, group_rows in sorted(retrieval_groups.items()):
        method, generations, dimension, comparison_type = key
        item: Dict[str, Any] = {
            "method": method,
            "generations": int(generations),
            "rff_dimension": None if dimension == "NA" else int(dimension),
            "comparison_type": comparison_type,
            "setting_comparisons": len(group_rows),
            "queries": sum(int(row["n_queries"]) for row in group_rows),
        }
        item.update({field: _weighted_mean(group_rows, field) for field in metric_fields})
        retrieval.append(item)

    approximation = []
    if distances_path.is_file():
        exact_lookup = {
            (
                row["generations"],
                row["query_setting"],
                row["reference_setting"],
            ): row
            for row in rows
            if row["method"] == "exact_mmd"
        }
        diagnostics: Dict[Tuple[int, int, str], List[Dict[str, float | None]]] = defaultdict(list)
        with np.load(distances_path, allow_pickle=False) as distances:
            for row in rows:
                if row["method"] != "rfftrace":
                    continue
                match = exact_lookup.get(
                    (
                        row["generations"],
                        row["query_setting"],
                        row["reference_setting"],
                    )
                )
                if match is None:
                    continue
                exact = np.asarray(distances[match["run_key"]], dtype=np.float64).reshape(-1)
                approximate = np.asarray(
                    distances[row["run_key"]], dtype=np.float64
                ).reshape(-1)
                errors = approximate - exact
                diagnostics[
                    (
                        int(row["generations"]),
                        int(row["rff_dimension"]),
                        row["comparison_type"],
                    )
                ].append(
                    {
                        "correlation": _correlation(exact, approximate),
                        "mae": float(np.mean(np.abs(errors))),
                        "rmse": math.sqrt(float(np.mean(errors * errors))),
                    }
                )
        for key, values in sorted(diagnostics.items()):
            generations, dimension, comparison_type = key
            correlations = [
                item["correlation"]
                for item in values
                if item["correlation"] is not None
            ]
            approximation.append(
                {
                    "generations": generations,
                    "rff_dimension": dimension,
                    "comparison_type": comparison_type,
                    "setting_comparisons": len(values),
                    "distance_correlation": (
                        None
                        if not correlations
                        else float(np.mean(np.asarray(correlations, dtype=np.float64)))
                    ),
                    "mae": float(np.mean([item["mae"] for item in values])),
                    "rmse": float(np.mean([item["rmse"] for item in values])),
                }
            )

    pilot_checks = _pilot_checks(retrieval, approximation)
    return {
        "format_version": 1,
        "experiment_output": str(source),
        "seed": metadata["config"]["experiment"]["seed"],
        "sigma": metadata["sigma"],
        "same_setting_evaluation": metadata.get("same_setting_evaluation"),
        "retrieval": retrieval,
        "rff_approximation": approximation,
        "pilot_checks": pilot_checks,
    }


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def aggregate_runs(output_dirs: Sequence[str | Path]) -> Dict[str, Any]:
    if not output_dirs:
        raise ValueError("at least one experiment output is required")
    summaries = [summarize_run(path) for path in output_dirs]

    retrieval_groups: Dict[Tuple[str, int, int | None, str], List[Mapping[str, Any]]] = defaultdict(list)
    approximation_groups: Dict[Tuple[int, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for summary in summaries:
        for item in summary["retrieval"]:
            retrieval_groups[
                (
                    item["method"],
                    item["generations"],
                    item["rff_dimension"],
                    item["comparison_type"],
                )
            ].append(item)
        for item in summary["rff_approximation"]:
            approximation_groups[
                (
                    item["generations"],
                    item["rff_dimension"],
                    item["comparison_type"],
                )
            ].append(item)

    retrieval = []
    for key, items in sorted(
        retrieval_groups.items(),
        key=lambda pair: (
            pair[0][0],
            pair[0][1],
            -1 if pair[0][2] is None else pair[0][2],
            pair[0][3],
        ),
    ):
        method, generations, dimension, comparison_type = key
        row: Dict[str, Any] = {
            "method": method,
            "generations": generations,
            "rff_dimension": dimension,
            "comparison_type": comparison_type,
            "runs": len(items),
        }
        metric_names = [name for name in items[0] if name.startswith("top_")] + ["mrr"]
        for name in metric_names:
            mean, std = _mean_std([float(item[name]) for item in items])
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
        retrieval.append(row)

    approximation = []
    for key, items in sorted(approximation_groups.items()):
        generations, dimension, comparison_type = key
        row = {
            "generations": generations,
            "rff_dimension": dimension,
            "comparison_type": comparison_type,
            "runs": len(items),
        }
        for source_name, output_name in (
            ("distance_correlation", "distance_correlation"),
            ("mae", "mae"),
            ("rmse", "rmse"),
        ):
            values = [item[source_name] for item in items if item[source_name] is not None]
            if values:
                mean, std = _mean_std([float(value) for value in values])
                row[f"{output_name}_mean"] = mean
                row[f"{output_name}_std"] = std
            else:
                row[f"{output_name}_mean"] = None
                row[f"{output_name}_std"] = None
        approximation.append(row)

    seeds = [summary["seed"] for summary in summaries]
    unique_seed_count = len(set(seeds))
    every_pilot_ready = all(
        summary["pilot_checks"]["ready_for_multi_seed"] for summary in summaries
    )
    scale_checks = {
        "at_least_two_unique_seeds": unique_seed_count >= 2,
        "every_run_passes_pilot_checks": every_pilot_ready,
    }
    return {
        "format_version": 1,
        "run_count": len(summaries),
        "seeds": seeds,
        "unique_seed_count": unique_seed_count,
        "experiment_outputs": [summary["experiment_output"] for summary in summaries],
        "retrieval": retrieval,
        "rff_approximation": approximation,
        "scale_readiness": {
            "ready_for_scale": all(scale_checks.values()),
            "checks": scale_checks,
            "interpretation": (
                "Passing authorizes cohort expansion operationally; it does not establish "
                "a performance claim."
            ),
        },
    }


def write_summary(summary: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return target
