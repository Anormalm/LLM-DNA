"""Deterministic collection-size and runtime planning."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from .data import CollectionManifest, ResponseCache


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _compatible_reuse_count(
    target: CollectionManifest, cache_dir: str | Path
) -> Dict[str, Any]:
    directory = Path(cache_dir)
    source_manifest = CollectionManifest.load(directory / "manifest.json")
    source = ResponseCache(directory, source_manifest).dataset
    target_models = set(target.model_ids)
    target_settings = {item.setting_id: item for item in target.settings}
    target_prompts = {item.prompt_id: item for item in target.prompts}
    source_revisions = source_manifest.metadata.get("model_revisions")
    target_revisions = target.metadata.get("model_revisions")
    if not isinstance(source_revisions, dict) or not isinstance(target_revisions, dict):
        raise ValueError("reuse estimates require pinned model revisions in both manifests")
    default_system_prompt = "You are a helpful assistant."
    if source_manifest.metadata.get("system_prompt", default_system_prompt) != target.metadata.get(
        "system_prompt", default_system_prompt
    ):
        raise ValueError("reuse cache and target system prompts differ")

    reusable = 0
    requested_tokens = 0
    for record in source.records:
        if record.model_id not in target_models:
            continue
        setting = target_settings.get(record.setting_id)
        prompt = target_prompts.get(record.prompt_id)
        if setting is None or prompt is None or record.generation_index >= target.generations:
            continue
        if source_manifest.setting(record.setting_id) != setting:
            continue
        source_prompt = next(
            item for item in source_manifest.prompts if item.prompt_id == record.prompt_id
        )
        if source_prompt != prompt:
            continue
        if source_revisions.get(record.model_id) != target_revisions.get(record.model_id):
            continue
        reusable += 1
        requested_tokens += setting.max_new_tokens
    return {
        "cache_dir": str(directory.resolve()),
        "random_seed": source_manifest.random_seed,
        "manifest_fingerprint": source_manifest.fingerprint,
        "reusable_records": reusable,
        "reusable_requested_tokens": requested_tokens,
    }


def _benchmark(cache_dir: str | Path) -> Dict[str, Any]:
    directory = Path(cache_dir)
    manifest = CollectionManifest.load(directory / "manifest.json")
    dataset = ResponseCache(directory, manifest).dataset
    elapsed = []
    generated_tokens = []
    serialized_bytes = []
    for record in dataset.records:
        seconds = record.metadata.get("elapsed_seconds")
        tokens = record.metadata.get("generated_tokens")
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and seconds > 0:
            elapsed.append(float(seconds))
            if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0:
                generated_tokens.append(tokens)
        serialized_bytes.append(
            len(
                (json.dumps(record.__dict__, sort_keys=True, ensure_ascii=False) + "\n").encode(
                    "utf-8"
                )
            )
        )
    report: Dict[str, Any] = {
        "cache_dir": str(directory.resolve()),
        "records": len(dataset.records),
        "timed_records": len(elapsed),
        "mean_serialized_bytes_per_record": statistics.fmean(serialized_bytes),
    }
    if elapsed:
        report.update(
            {
                "seconds_per_record_mean": statistics.fmean(elapsed),
                "seconds_per_record_median": statistics.median(elapsed),
                "seconds_per_record_p90": _percentile(elapsed, 0.9),
                "measured_seconds": sum(elapsed),
            }
        )
    if generated_tokens and len(generated_tokens) == len(elapsed):
        report["generated_tokens"] = sum(generated_tokens)
        report["aggregate_generated_tokens_per_second"] = sum(generated_tokens) / sum(
            elapsed
        )
    return report


def estimate_collection(
    manifest: CollectionManifest,
    *,
    seed_count: int = 1,
    embedding_dimension: int = 768,
    embedding_bytes: int = 4,
    reuse_cache_dirs: Iterable[str | Path] = (),
    benchmark_cache_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Return exact cardinalities and explicitly assumption-bound resource estimates."""

    for name, value in (
        ("seed_count", seed_count),
        ("embedding_dimension", embedding_dimension),
        ("embedding_bytes", embedding_bytes),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    models = len(manifest.model_ids)
    prompts = len(manifest.prompts)
    settings = len(manifest.settings)
    records_per_seed = models * prompts * settings * manifest.generations
    requested_tokens_per_seed = models * prompts * manifest.generations * sum(
        item.max_new_tokens for item in manifest.settings
    )
    total_records = records_per_seed * seed_count
    total_requested_tokens = requested_tokens_per_seed * seed_count

    reuse = [_compatible_reuse_count(manifest, path) for path in reuse_cache_dirs]
    seeds = [row["random_seed"] for row in reuse]
    if len(set(seeds)) != len(seeds):
        raise ValueError("reuse caches must use distinct collection seeds")
    reusable_records = sum(row["reusable_records"] for row in reuse)
    reusable_tokens = sum(row["reusable_requested_tokens"] for row in reuse)
    if reusable_records > total_records:
        raise ValueError("reusable records exceed the planned multi-seed collection")

    remaining_records = total_records - reusable_records
    remaining_requested_tokens = total_requested_tokens - reusable_tokens
    embedding_bytes_total = total_records * embedding_dimension * embedding_bytes
    benchmark = None if benchmark_cache_dir is None else _benchmark(benchmark_cache_dir)
    bytes_per_record = (
        1024.0
        if benchmark is None
        else float(benchmark["mean_serialized_bytes_per_record"])
    )
    storage = {
        "response_cache_bytes_estimate": math.ceil(total_records * bytes_per_record),
        "response_bytes_per_record_assumption": bytes_per_record,
        "embedding_bytes_uncompressed": embedding_bytes_total,
        "embedding_gib_uncompressed": embedding_bytes_total / (1024**3),
        "combined_gib_before_experiment_outputs": (
            total_records * bytes_per_record + embedding_bytes_total
        )
        / (1024**3),
    }

    runtime: Mapping[str, Any] | None = None
    if benchmark is not None and benchmark["timed_records"]:
        mean_seconds = benchmark["seconds_per_record_mean"] * remaining_records
        p90_seconds = benchmark["seconds_per_record_p90"] * remaining_records
        runtime = {
            "remaining_inference_calls": remaining_records,
            "central_seconds": mean_seconds,
            "central_hours": mean_seconds / 3600.0,
            "p90_per_call_seconds_projection": p90_seconds,
            "p90_per_call_hours_projection": p90_seconds / 3600.0,
            "scope_warning": (
                "Projection assumes the benchmark cache's model mix, device, output length, "
                "and sequential execution; new or larger checkpoints require their own benchmark."
            ),
        }

    deterministic_settings = sum(item.temperature == 0 for item in manifest.settings)
    return {
        "format_version": 1,
        "manifest_fingerprint": manifest.fingerprint,
        "dataset_id": manifest.dataset_id,
        "design": {
            "models": models,
            "settings": settings,
            "deterministic_settings": deterministic_settings,
            "stochastic_settings": settings - deterministic_settings,
            "calibration_prompts": len(manifest.prompts_for_split("calibration")),
            "evaluation_prompts": len(manifest.prompts_for_split("evaluation")),
            "generations": manifest.generations,
            "seeds": seed_count,
            "embedding_dimension": embedding_dimension,
            "embedding_bytes_per_value": embedding_bytes,
        },
        "work": {
            "records_per_seed": records_per_seed,
            "total_records": total_records,
            "requested_generation_tokens_per_seed_upper_bound": requested_tokens_per_seed,
            "requested_generation_tokens_upper_bound": total_requested_tokens,
            "reusable_records": reusable_records,
            "remaining_records": remaining_records,
            "remaining_requested_generation_tokens_upper_bound": remaining_requested_tokens,
            "reuse_fraction": reusable_records / total_records,
        },
        "reuse_caches": reuse,
        "storage": storage,
        "benchmark": benchmark,
        "runtime_projection": runtime,
    }
