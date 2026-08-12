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
    }
    return {
        "format_version": 1,
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
        "checks": checks,
        "ready_for_scale": all(checks.values()),
    }
