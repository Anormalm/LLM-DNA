"""Auditable response-cache quality summaries for collection preflights."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Dict, Mapping, Sequence, Tuple

from .data import ResponseDataset


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def _timed_records(records: Sequence[Any]) -> list[Tuple[float, int]]:
    timed = []
    for item in records:
        seconds = item.metadata.get("elapsed_seconds")
        tokens = item.metadata.get("generated_tokens")
        if (
            isinstance(seconds, (int, float))
            and not isinstance(seconds, bool)
            and seconds > 0
            and isinstance(tokens, int)
            and not isinstance(tokens, bool)
            and tokens > 0
        ):
            timed.append((float(seconds), tokens))
    return timed


def collection_progress_report(dataset: ResponseDataset) -> Dict[str, Any]:
    """Audit an incomplete append-only cache without claiming final quality readiness."""

    records = dataset.records
    expected = dataset.expected_count
    remaining = expected - len(records)
    cells: Dict[Tuple[str, str, str], list] = defaultdict(list)
    for record in records:
        cells[(record.model_id, record.setting_id, record.prompt_id)].append(record)
    expected_cells_per_model = len(dataset.manifest.settings) * len(dataset.manifest.prompts)
    expected_records_per_model = expected_cells_per_model * dataset.manifest.generations
    deterministic_ids = {
        item.setting_id for item in dataset.manifest.settings if item.temperature == 0
    }

    model_rows = []
    for model_id in dataset.manifest.model_ids:
        model_records = [item for item in records if item.model_id == model_id]
        model_cells = {
            key: values for key, values in cells.items() if key[0] == model_id
        }
        complete_cells = sum(
            len(values) == dataset.manifest.generations
            for values in model_cells.values()
        )
        timed = _timed_records(model_records)
        deterministic_cells = [
            values
            for (_, setting_id, _), values in model_cells.items()
            if setting_id in deterministic_ids
        ]
        stochastic_cells = [
            values
            for (_, setting_id, _), values in model_cells.items()
            if setting_id not in deterministic_ids
        ]

        def mean_unique_ratio(groups: Sequence[Sequence[Any]]) -> float | None:
            if not groups:
                return None
            return statistics.fmean(
                len({item.response for item in group}) / len(group) for group in groups
            )

        row: Dict[str, Any] = {
            "model_id": model_id,
            "records": len(model_records),
            "expected_records": expected_records_per_model,
            "completion_fraction": len(model_records) / expected_records_per_model,
            "observed_cells": len(model_cells),
            "complete_cells": complete_cells,
            "expected_cells": expected_cells_per_model,
            "truncation_rate_observed": (
                sum(
                    item.metadata.get("stop_reason") == "max_new_tokens"
                    for item in model_records
                )
                / len(model_records)
                if model_records
                else None
            ),
            "deterministic_mean_cell_unique_ratio_observed": mean_unique_ratio(
                deterministic_cells
            ),
            "stochastic_mean_cell_unique_ratio_observed": mean_unique_ratio(
                stochastic_cells
            ),
            "timed_records": len(timed),
        }
        if timed:
            row["aggregate_tokens_per_second_observed"] = sum(
                item[1] for item in timed
            ) / sum(item[0] for item in timed)
            row["seconds_per_record_mean_observed"] = statistics.fmean(
                item[0] for item in timed
            )
        model_rows.append(row)

    timed = _timed_records(records)
    truncated_records = sum(
        item.metadata.get("stop_reason") == "max_new_tokens" for item in records
    )
    overall_truncation = (
        truncated_records
        / len(records)
        if records
        else None
    )
    truncation_limit = expected * 0.25
    remaining_truncation_budget = truncation_limit - truncated_records
    gate_forecast = {
        "threshold": 0.25,
        "truncated_records_observed": truncated_records,
        "maximum_truncated_records": truncation_limit,
        "remaining_truncation_budget": remaining_truncation_budget,
        "final_truncation_rate_lower_bound": truncated_records / expected,
        "gate_still_mathematically_achievable": remaining_truncation_budget >= 0,
        "maximum_remaining_truncation_rate_to_pass": (
            min(1.0, max(0.0, remaining_truncation_budget / remaining))
            if remaining
            else None
        ),
    }
    projection = None
    if timed:
        mean_seconds = statistics.fmean(item[0] for item in timed)
        projection = {
            "basis": "mean measured inference seconds per durable record observed so far",
            "seconds_per_record_mean_observed": mean_seconds,
            "remaining_inference_seconds": remaining * mean_seconds,
            "remaining_inference_hours": remaining * mean_seconds / 3600,
            "warning": (
                "Current-model-mix extrapolation only; it excludes model loading, encoding, "
                "analysis, and future checkpoint speed differences."
            ),
        }
    return {
        "format_version": 1,
        "report_kind": "incomplete_collection_progress; not a final quality gate",
        "manifest_fingerprint": dataset.manifest.fingerprint,
        "records": len(records),
        "expected_records": expected,
        "remaining_records": remaining,
        "completion_fraction": len(records) / expected,
        "observed_cells": len(cells),
        "expected_cells": (
            len(dataset.manifest.model_ids)
            * len(dataset.manifest.settings)
            * len(dataset.manifest.prompts)
        ),
        "overall_truncation_rate_observed": overall_truncation,
        "truncation_gate_forecast": gate_forecast,
        "timed_records": len(timed),
        "aggregate_tokens_per_second_observed": (
            sum(item[1] for item in timed) / sum(item[0] for item in timed)
            if timed
            else None
        ),
        "runtime_projection": projection,
        "models": model_rows,
        "final_quality_gate_eligible": dataset.complete,
    }


def response_quality_report(dataset: ResponseDataset) -> Dict[str, Any]:
    """Summarize diversity, length, truncation, and measured generation throughput."""

    dataset.require_complete()
    cells: Dict[Tuple[str, str, str], list] = defaultdict(list)
    for record in dataset.records:
        cells[(record.model_id, record.setting_id, record.prompt_id)].append(record)

    setting_rows = []
    all_words = []
    all_chars = []
    all_elapsed = []
    all_generated_tokens = []
    for setting in dataset.manifest.settings:
        records = [item for item in dataset.records if item.setting_id == setting.setting_id]
        cell_records = [
            items for (model, setting_id, prompt), items in cells.items() if setting_id == setting.setting_id
        ]
        unique_ratios = [len({item.response for item in items}) / len(items) for items in cell_records]
        words = [len(item.response.split()) for item in records]
        chars = [len(item.response) for item in records]
        timed = _timed_records(records)
        elapsed = [item[0] for item in timed]
        generated_tokens = [item[1] for item in timed]
        truncated = sum(item.metadata.get("stop_reason") == "max_new_tokens" for item in records)
        row: Dict[str, Any] = {
            "setting_id": setting.setting_id,
            "temperature": setting.temperature,
            "top_p": setting.top_p,
            "records": len(records),
            "cells": len(cell_records),
            "mean_cell_unique_ratio": statistics.fmean(unique_ratios) if unique_ratios else 0.0,
            "minimum_cell_unique_ratio": min(unique_ratios, default=0.0),
            "maximum_cell_unique_ratio": max(unique_ratios, default=0.0),
            "word_count_median": statistics.median(words) if words else 0.0,
            "word_count_p90": _percentile(words, 0.9),
            "character_count_median": statistics.median(chars) if chars else 0.0,
            "truncation_rate": truncated / len(records) if records else 0.0,
            "timed_records": len(elapsed),
        }
        if elapsed:
            row["elapsed_seconds_mean"] = statistics.fmean(elapsed)
            row["elapsed_seconds_p90"] = _percentile(elapsed, 0.9)
        if timed:
            row["generated_tokens_mean"] = statistics.fmean(generated_tokens)
            row["aggregate_tokens_per_second"] = sum(generated_tokens) / sum(elapsed)
        setting_rows.append(row)
        all_words.extend(words)
        all_chars.extend(chars)
        all_elapsed.extend(elapsed)
        all_generated_tokens.extend(generated_tokens)

    deterministic = [row for row in setting_rows if row["temperature"] == 0]
    stochastic = [row for row in setting_rows if row["temperature"] > 0]
    model_rows = []
    for model_id in dataset.manifest.model_ids:
        records = [item for item in dataset.records if item.model_id == model_id]
        model_cells = [
            items for (cell_model, _, _), items in cells.items() if cell_model == model_id
        ]
        words = [len(item.response.split()) for item in records]
        timed = _timed_records(records)
        truncated = sum(
            item.metadata.get("stop_reason") == "max_new_tokens" for item in records
        )
        model_rows.append(
            {
                "model_id": model_id,
                "records": len(records),
                "cells": len(model_cells),
                "word_count_median": statistics.median(words),
                "word_count_p90": _percentile(words, 0.9),
                "truncation_rate": truncated / len(records),
                "timed_records": len(timed),
                "aggregate_tokens_per_second": (
                    sum(item[1] for item in timed) / sum(item[0] for item in timed)
                    if timed
                    else None
                ),
            }
        )
    checks: Mapping[str, bool] = {
        "cache_complete": dataset.complete,
        "all_cells_present": len(cells)
        == len(dataset.manifest.model_ids)
        * len(dataset.manifest.settings)
        * len(dataset.manifest.prompts),
        "deterministic_cells_stable": all(
            row["maximum_cell_unique_ratio"] <= 1 / dataset.manifest.generations
            for row in deterministic
        ),
        "stochastic_cells_diverse": all(
            row["mean_cell_unique_ratio"] >= 0.8
            for row in stochastic
        ),
        "median_response_has_at_least_six_words": statistics.median(all_words) >= 6,
        "overall_truncation_rate_at_most_25_percent": sum(
            item.metadata.get("stop_reason") == "max_new_tokens" for item in dataset.records
        )
        / len(dataset.records)
        <= 0.25,
        "every_model_truncation_rate_at_most_25_percent": all(
            row["truncation_rate"] <= 0.25 for row in model_rows
        ),
    }
    return {
        "format_version": 2,
        "manifest_fingerprint": dataset.manifest.fingerprint,
        "records": len(dataset.records),
        "expected_records": dataset.expected_count,
        "cell_count": len(cells),
        "overall": {
            "word_count_median": statistics.median(all_words),
            "word_count_p90": _percentile(all_words, 0.9),
            "character_count_median": statistics.median(all_chars),
            "timed_records": len(all_elapsed),
            "measured_seconds": sum(all_elapsed),
            "generated_tokens": sum(all_generated_tokens),
            "aggregate_tokens_per_second": (
                sum(all_generated_tokens) / sum(all_elapsed) if all_elapsed else None
            ),
        },
        "settings": setting_rows,
        "models": model_rows,
        "checks": checks,
        "ready_for_scale": all(checks.values()),
    }
