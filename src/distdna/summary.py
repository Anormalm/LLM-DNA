"""Post-run summaries for retrieval and RFF approximation quality."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


ProtocolKey = Tuple[str, str, float | None, float, int]


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


def _optional_dimension(value: str) -> int | None:
    return None if value == "NA" else int(value)


def _metric_sort_key(row: Mapping[str, str]) -> Tuple[Any, ...]:
    return (
        row["method"],
        int(row["generations"]),
        -1 if row["rff_dimension"] == "NA" else int(row["rff_dimension"]),
        -1
        if row["projection_dimension"] == "NA"
        else int(row["projection_dimension"]),
        row["query_setting"],
        row["reference_setting"],
    )


def _protocol_key(summary: Mapping[str, Any]) -> ProtocolKey:
    bandwidth = summary["bandwidth"]
    return (
        str(summary["normalization"]),
        str(bandwidth["strategy"]),
        None if bandwidth["value"] is None else float(bandwidth["value"]),
        float(bandwidth["multiplier"]),
        int(bandwidth["max_pairs"]),
    )


def _protocol_fields(key: ProtocolKey) -> Dict[str, Any]:
    normalization, strategy, value, multiplier, max_pairs = key
    return {
        "normalization": normalization,
        "bandwidth_strategy": strategy,
        "bandwidth_value": value,
        "bandwidth_multiplier": multiplier,
        "bandwidth_max_pairs": max_pairs,
    }


def _protocol_sort_key(key: ProtocolKey) -> Tuple[Any, ...]:
    normalization, strategy, value, multiplier, max_pairs = key
    return (
        normalization,
        strategy,
        float("-inf") if value is None else value,
        multiplier,
        max_pairs,
    )


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


def _distance_diagnostic(
    reference: np.ndarray, approximate: np.ndarray
) -> Dict[str, float | None]:
    errors = approximate - reference
    return {
        "correlation": _correlation(reference, approximate),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": math.sqrt(float(np.mean(errors * errors))),
    }


def _summarize_diagnostics(
    diagnostics: Mapping[
        Tuple[int, int, int | None, str], Sequence[Mapping[str, float | None]]
    ],
) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for key, values in sorted(
        diagnostics.items(),
        key=lambda pair: (
            pair[0][0],
            pair[0][1],
            -1 if pair[0][2] is None else pair[0][2],
            pair[0][3],
        ),
    ):
        generations, dimension, projection, comparison_type = key
        correlations = [
            item["correlation"]
            for item in values
            if item["correlation"] is not None
        ]
        summary.append(
            {
                "generations": generations,
                "rff_dimension": dimension,
                "projection_dimension": projection,
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
    return summary


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

    retrieval_groups: Dict[
        Tuple[str, str, str, str, str], List[Mapping[str, str]]
    ] = defaultdict(list)
    for row in rows:
        key = (
            row["method"],
            row["generations"],
            row["rff_dimension"],
            row["projection_dimension"],
            row["comparison_type"],
        )
        retrieval_groups[key].append(row)

    metric_fields = [field for field in rows[0] if field.startswith("top_")]
    metric_fields.append("mrr")
    retrieval = []
    for key, group_rows in sorted(
        retrieval_groups.items(),
        key=lambda pair: (
            pair[0][0],
            int(pair[0][1]),
            -1 if pair[0][2] == "NA" else int(pair[0][2]),
            -1 if pair[0][3] == "NA" else int(pair[0][3]),
            pair[0][4],
        ),
    ):
        method, generations, dimension, projection, comparison_type = key
        item: Dict[str, Any] = {
            "method": method,
            "generations": int(generations),
            "rff_dimension": _optional_dimension(dimension),
            "projection_dimension": _optional_dimension(projection),
            "comparison_type": comparison_type,
            "setting_comparisons": len(group_rows),
            "queries": sum(int(row["n_queries"]) for row in group_rows),
        }
        item.update({field: _weighted_mean(group_rows, field) for field in metric_fields})
        retrieval.append(item)

    setting_retrieval: List[Dict[str, Any]] = []
    for row in sorted(rows, key=_metric_sort_key):
        item = {
            "method": row["method"],
            "generations": int(row["generations"]),
            "rff_dimension": _optional_dimension(row["rff_dimension"]),
            "projection_dimension": _optional_dimension(
                row["projection_dimension"]
            ),
            "query_setting": row["query_setting"],
            "reference_setting": row["reference_setting"],
            "comparison_type": row["comparison_type"],
            "queries": int(row["n_queries"]),
        }
        item.update({field: float(row[field]) for field in metric_fields})
        setting_retrieval.append(item)

    approximation: List[Dict[str, Any]] = []
    projection_approximation: List[Dict[str, Any]] = []
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
        diagnostics: Dict[
            Tuple[int, int, int | None, str], List[Dict[str, float | None]]
        ] = defaultdict(list)
        unprojected_lookup = {
            (
                row["generations"],
                row["rff_dimension"],
                row["query_setting"],
                row["reference_setting"],
            ): row
            for row in rows
            if row["method"] == "rfftrace"
            and row["projection_dimension"] == "NA"
        }
        projection_diagnostics: Dict[
            Tuple[int, int, int | None, str], List[Dict[str, float | None]]
        ] = defaultdict(list)
        with np.load(distances_path, allow_pickle=False) as distances:
            for row in rows:
                if row["method"] != "rfftrace":
                    continue
                projection = (
                    None
                    if row["projection_dimension"] == "NA"
                    else int(row["projection_dimension"])
                )
                approximate = np.asarray(
                    distances[row["run_key"]], dtype=np.float64
                ).reshape(-1)
                match = exact_lookup.get(
                    (
                        row["generations"],
                        row["query_setting"],
                        row["reference_setting"],
                    )
                )
                diagnostic_key = (
                    int(row["generations"]),
                    int(row["rff_dimension"]),
                    projection,
                    row["comparison_type"],
                )
                if match is not None:
                    exact = np.asarray(
                        distances[match["run_key"]], dtype=np.float64
                    ).reshape(-1)
                    diagnostics[diagnostic_key].append(
                        _distance_diagnostic(exact, approximate)
                    )
                if projection is not None:
                    unprojected = unprojected_lookup.get(
                        (
                            row["generations"],
                            row["rff_dimension"],
                            row["query_setting"],
                            row["reference_setting"],
                        )
                    )
                    if unprojected is not None:
                        reference = np.asarray(
                            distances[unprojected["run_key"]], dtype=np.float64
                        ).reshape(-1)
                        projection_diagnostics[diagnostic_key].append(
                            _distance_diagnostic(reference, approximate)
                        )
        approximation = _summarize_diagnostics(diagnostics)
        projection_approximation = _summarize_diagnostics(projection_diagnostics)

    experiment_config = metadata["config"]["experiment"]
    pilot_checks = _pilot_checks(retrieval, approximation)
    return {
        "format_version": 1,
        "experiment_output": str(source),
        "seed": metadata["config"]["experiment"]["seed"],
        "normalization": experiment_config["normalization"],
        "bandwidth": experiment_config["bandwidth"],
        "input_hashes": {
            split: metadata["inputs"][split]["sha256"]
            for split in ("calibration", "evaluation")
        },
        "sigma": metadata["sigma"],
        "same_setting_evaluation": metadata.get("same_setting_evaluation"),
        "retrieval": retrieval,
        "setting_retrieval": setting_retrieval,
        "rff_approximation": approximation,
        "projection_approximation": projection_approximation,
        "pilot_checks": pilot_checks,
    }


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def aggregate_runs(output_dirs: Sequence[str | Path]) -> Dict[str, Any]:
    if not output_dirs:
        raise ValueError("at least one experiment output is required")
    summaries = [summarize_run(path) for path in output_dirs]

    retrieval_groups: Dict[
        Tuple[ProtocolKey, str, int, int | None, int | None, str],
        List[Mapping[str, Any]],
    ] = defaultdict(list)
    setting_groups: Dict[
        Tuple[ProtocolKey, str, int, int | None, int | None, str, str],
        List[Mapping[str, Any]],
    ] = defaultdict(list)
    approximation_groups: Dict[
        Tuple[ProtocolKey, int, int, int | None, str], List[Mapping[str, Any]]
    ] = defaultdict(list)
    projection_groups: Dict[
        Tuple[ProtocolKey, int, int, int | None, str], List[Mapping[str, Any]]
    ] = defaultdict(list)
    protocol_summaries: Dict[ProtocolKey, List[Mapping[str, Any]]] = defaultdict(list)
    for summary in summaries:
        protocol = _protocol_key(summary)
        protocol_summaries[protocol].append(summary)
        for item in summary["retrieval"]:
            retrieval_groups[
                (
                    protocol,
                    item["method"],
                    item["generations"],
                    item["rff_dimension"],
                    item["projection_dimension"],
                    item["comparison_type"],
                )
            ].append(item)
        for item in summary["setting_retrieval"]:
            setting_groups[
                (
                    protocol,
                    item["method"],
                    item["generations"],
                    item["rff_dimension"],
                    item["projection_dimension"],
                    item["query_setting"],
                    item["reference_setting"],
                )
            ].append(item)
        for item in summary["rff_approximation"]:
            approximation_groups[
                (
                    protocol,
                    item["generations"],
                    item["rff_dimension"],
                    item["projection_dimension"],
                    item["comparison_type"],
                )
            ].append(item)
        for item in summary["projection_approximation"]:
            projection_groups[
                (
                    protocol,
                    item["generations"],
                    item["rff_dimension"],
                    item["projection_dimension"],
                    item["comparison_type"],
                )
            ].append(item)

    retrieval = []
    for key, items in sorted(
        retrieval_groups.items(),
        key=lambda pair: (
            _protocol_sort_key(pair[0][0]),
            pair[0][1],
            pair[0][2],
            -1 if pair[0][3] is None else pair[0][3],
            -1 if pair[0][4] is None else pair[0][4],
            pair[0][5],
        ),
    ):
        protocol, method, generations, dimension, projection, comparison_type = key
        row: Dict[str, Any] = {
            **_protocol_fields(protocol),
            "method": method,
            "generations": generations,
            "rff_dimension": dimension,
            "projection_dimension": projection,
            "comparison_type": comparison_type,
            "runs": len(items),
        }
        metric_names = [name for name in items[0] if name.startswith("top_")] + ["mrr"]
        for name in metric_names:
            mean, std = _mean_std([float(item[name]) for item in items])
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
        retrieval.append(row)

    setting_retrieval = []
    for key, items in sorted(
        setting_groups.items(),
        key=lambda pair: (
            _protocol_sort_key(pair[0][0]),
            pair[0][1],
            pair[0][2],
            -1 if pair[0][3] is None else pair[0][3],
            -1 if pair[0][4] is None else pair[0][4],
            pair[0][5],
            pair[0][6],
        ),
    ):
        protocol, method, generations, dimension, projection, query, reference = key
        row = {
            **_protocol_fields(protocol),
            "method": method,
            "generations": generations,
            "rff_dimension": dimension,
            "projection_dimension": projection,
            "query_setting": query,
            "reference_setting": reference,
            "comparison_type": items[0]["comparison_type"],
            "runs": len(items),
        }
        metric_names = [name for name in items[0] if name.startswith("top_")] + [
            "mrr"
        ]
        for name in metric_names:
            mean, std = _mean_std([float(item[name]) for item in items])
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
        setting_retrieval.append(row)

    def aggregate_diagnostics(
        groups: Mapping[
            Tuple[ProtocolKey, int, int, int | None, str],
            Sequence[Mapping[str, Any]],
        ],
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for key, items in sorted(
            groups.items(),
            key=lambda pair: (
                _protocol_sort_key(pair[0][0]),
                pair[0][1],
                pair[0][2],
                -1 if pair[0][3] is None else pair[0][3],
                pair[0][4],
            ),
        ):
            protocol, generations, dimension, projection, comparison_type = key
            row = {
                **_protocol_fields(protocol),
                "generations": generations,
                "rff_dimension": dimension,
                "projection_dimension": projection,
                "comparison_type": comparison_type,
                "runs": len(items),
            }
            for name in ("distance_correlation", "mae", "rmse"):
                values = [item[name] for item in items if item[name] is not None]
                if values:
                    mean, std = _mean_std([float(value) for value in values])
                    row[f"{name}_mean"] = mean
                    row[f"{name}_std"] = std
                else:
                    row[f"{name}_mean"] = None
                    row[f"{name}_std"] = None
            result.append(row)
        return result

    approximation = aggregate_diagnostics(approximation_groups)
    projection_approximation = aggregate_diagnostics(projection_groups)

    seeds = [summary["seed"] for summary in summaries]
    unique_seed_count = len(set(seeds))
    input_hashes = [summary["input_hashes"] for summary in summaries]
    unique_input_count = len(
        {
            (item["calibration"], item["evaluation"])
            for item in input_hashes
        }
    )
    every_pilot_ready = all(
        summary["pilot_checks"]["ready_for_multi_seed"] for summary in summaries
    )
    protocols = []
    every_protocol_repeated = True
    for protocol, items in sorted(
        protocol_summaries.items(), key=lambda pair: _protocol_sort_key(pair[0])
    ):
        protocol_seeds = {int(item["seed"]) for item in items}
        protocol_inputs = {
            (
                item["input_hashes"]["calibration"],
                item["input_hashes"]["evaluation"],
            )
            for item in items
        }
        repeated = len(protocol_seeds) >= 2 and len(protocol_inputs) >= 2
        every_protocol_repeated = every_protocol_repeated and repeated
        sigma_mean, sigma_std = _mean_std([float(item["sigma"]) for item in items])
        protocols.append(
            {
                **_protocol_fields(protocol),
                "runs": len(items),
                "unique_seed_count": len(protocol_seeds),
                "unique_input_count": len(protocol_inputs),
                "sigma_mean": sigma_mean,
                "sigma_std": sigma_std,
                "repeated_independently": repeated,
            }
        )
    group_collections = (
        retrieval_groups,
        setting_groups,
        approximation_groups,
        projection_groups,
    )
    balanced_protocol_coverage = all(
        len(items) == len(protocol_summaries[key[0]])
        for groups in group_collections
        for key, items in groups.items()
    )
    scale_checks = {
        "at_least_two_unique_seeds": unique_seed_count >= 2,
        "at_least_two_distinct_input_datasets": unique_input_count >= 2,
        "every_run_passes_pilot_checks": every_pilot_ready,
        "every_protocol_repeated_independently": every_protocol_repeated,
        "balanced_protocol_coverage": balanced_protocol_coverage,
    }
    return {
        "format_version": 1,
        "run_count": len(summaries),
        "seeds": seeds,
        "unique_seed_count": unique_seed_count,
        "input_hashes": input_hashes,
        "unique_input_count": unique_input_count,
        "experiment_outputs": [summary["experiment_output"] for summary in summaries],
        "protocols": protocols,
        "retrieval": retrieval,
        "setting_retrieval": setting_retrieval,
        "rff_approximation": approximation,
        "projection_approximation": projection_approximation,
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
