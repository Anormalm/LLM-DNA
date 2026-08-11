"""Paper-faithful decoding-grid analysis over repeated-run aggregates."""

from __future__ import annotations

import hashlib
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
_METRICS = ("top_1", "top_3", "top_5", "mrr")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matches_protocol(
    row: Mapping[str, Any],
    normalization: str,
    bandwidth_multiplier: float,
    generations: int,
    rff_dimension: int,
    projection_dimension: int | None,
) -> bool:
    if (
        row["normalization"] != normalization
        or not np.isclose(float(row["bandwidth_multiplier"]), bandwidth_multiplier)
        or int(row["generations"]) != generations
    ):
        return False
    if row["method"] == "rfftrace":
        return (
            row["rff_dimension"] == rff_dimension
            and row["projection_dimension"] == projection_dimension
        )
    return row["rff_dimension"] is None and row["projection_dimension"] is None


def _effect_rows(
    cells: Sequence[Mapping[str, Any]], factor: str
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, float], List[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        if cell["temperature"] == 0.0:
            continue
        groups[(cell["method"], cell["evaluation"], float(cell[factor]))].append(cell)
    result = []
    for (method, evaluation, value), rows in sorted(groups.items()):
        item: Dict[str, Any] = {
            "method": method,
            "evaluation": evaluation,
            factor: value,
            "decoding_cells": len(rows),
        }
        for metric in _METRICS:
            item[f"{metric}_mean"] = float(
                np.mean([float(row[f"{metric}_mean"]) for row in rows])
            )
        result.append(item)
    return result


def _effect_ranges(
    temperature_effects: Sequence[Mapping[str, Any]],
    top_p_effects: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    result = []
    for method in _METHODS:
        for evaluation in ("same_setting", "cross_to_deterministic"):
            temperatures = [
                row
                for row in temperature_effects
                if row["method"] == method and row["evaluation"] == evaluation
            ]
            top_ps = [
                row
                for row in top_p_effects
                if row["method"] == method and row["evaluation"] == evaluation
            ]
            if not temperatures or not top_ps:
                raise ValueError("decoding factor effects are incomplete")
            item: Dict[str, Any] = {"method": method, "evaluation": evaluation}
            for metric in _METRICS:
                temperature_values = [float(row[f"{metric}_mean"]) for row in temperatures]
                top_p_values = [float(row[f"{metric}_mean"]) for row in top_ps]
                item[f"{metric}_temperature_range"] = max(temperature_values) - min(
                    temperature_values
                )
                item[f"{metric}_top_p_range"] = max(top_p_values) - min(top_p_values)
            result.append(item)
    return result


def build_decoding_report(
    aggregate_path: str | Path,
    manifest_path: str | Path,
    *,
    normalization: str = "l2",
    bandwidth_multiplier: float = 1.0,
    generations: int = 4,
    rff_dimension: int = 512,
    projection_dimension: int | None = None,
) -> Dict[str, Any]:
    source = Path(aggregate_path).resolve()
    with source.open("r", encoding="utf-8") as stream:
        aggregate = json.load(stream)
    manifest = CollectionManifest.load(manifest_path)
    setting_lookup = {item.setting_id: item for item in manifest.settings}
    deterministic = [item for item in manifest.settings if item.temperature == 0.0]
    if len(deterministic) != 1:
        raise ValueError("decoding report requires exactly one deterministic setting")
    deterministic_id = deterministic[0].setting_id

    selected = [
        row
        for row in aggregate.get("setting_retrieval", [])
        if _matches_protocol(
            row,
            normalization,
            bandwidth_multiplier,
            generations,
            rff_dimension,
            projection_dimension,
        )
    ]
    unknown = sorted(
        {
            setting_id
            for row in selected
            for setting_id in (row["query_setting"], row["reference_setting"])
            if setting_id not in setting_lookup
        }
    )
    if unknown:
        raise ValueError(f"aggregate contains settings absent from manifest: {unknown}")

    cells = []
    for row in selected:
        query = row["query_setting"]
        reference = row["reference_setting"]
        if query == reference:
            evaluation = "same_setting"
        elif reference == deterministic_id and query != deterministic_id:
            evaluation = "cross_to_deterministic"
        else:
            continue
        setting = setting_lookup[query]
        cell = {
            "method": row["method"],
            "evaluation": evaluation,
            "query_setting": query,
            "reference_setting": reference,
            "temperature": setting.temperature,
            "top_p": setting.top_p,
            "runs": row["runs"],
        }
        for metric in _METRICS:
            cell[f"{metric}_mean"] = row[f"{metric}_mean"]
            cell[f"{metric}_std"] = row[f"{metric}_std"]
        cells.append(cell)

    expected_per_method = len(manifest.settings) + len(manifest.settings) - 1
    counts = {method: sum(cell["method"] == method for cell in cells) for method in _METHODS}
    incomplete = {
        method: count for method, count in counts.items() if count != expected_per_method
    }
    if incomplete:
        raise ValueError(
            "decoding aggregate does not contain a complete selected grid: "
            f"expected {expected_per_method} cells per method, observed {incomplete}"
        )
    cells.sort(
        key=lambda item: (
            item["method"],
            item["evaluation"],
            item["temperature"],
            item["top_p"],
        )
    )
    temperature_effects = _effect_rows(cells, "temperature")
    top_p_effects = _effect_rows(cells, "top_p")
    return {
        "format_version": 1,
        "aggregate": str(source),
        "aggregate_sha256": _sha256(source),
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_fingerprint": manifest.fingerprint,
        "protocol": {
            "normalization": normalization,
            "bandwidth_multiplier": bandwidth_multiplier,
            "generations": generations,
            "rff_dimension": rff_dimension,
            "projection_dimension": projection_dimension,
            "deterministic_reference": deterministic_id,
        },
        "cell_count": len(cells),
        "cells": cells,
        "temperature_effects": temperature_effects,
        "top_p_effects": top_p_effects,
        "effect_ranges": _effect_ranges(temperature_effects, top_p_effects),
    }
