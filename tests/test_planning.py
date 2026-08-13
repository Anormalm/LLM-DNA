from dataclasses import replace
from pathlib import Path

import pytest

from distdna.data import (
    CollectionManifest,
    DecodingSetting,
    Prompt,
    ResponseCache,
    ResponseRecord,
    generation_seed,
)
from distdna.cli import main
from distdna.planning import estimate_collection
from distdna.quality import collection_progress_report, response_quality_report


def _manifest(generations: int) -> CollectionManifest:
    return CollectionManifest(
        dataset_id="planning-test",
        model_ids=("m0", "m1"),
        settings=(
            DecodingSetting("deterministic", 0, 1, 16),
            DecodingSetting("sample", 0.7, 0.9, 16),
        ),
        prompts=(
            Prompt("c0", "Explain a calibration concept.", "calibration"),
            Prompt("e0", "Explain an evaluation concept.", "evaluation"),
        ),
        generations=generations,
        random_seed=7,
        metadata={"model_revisions": {"m0": "a" * 40, "m1": "b" * 40}},
    )


def _complete_cache(path: Path, manifest: CollectionManifest) -> ResponseCache:
    cache = ResponseCache(path, manifest)
    for model_id in manifest.model_ids:
        for setting in manifest.settings:
            for prompt in manifest.prompts:
                for generation_index in range(manifest.generations):
                    response = (
                        "stable deterministic response with enough words"
                        if setting.temperature == 0
                        else f"diverse stochastic response number {generation_index} has enough words"
                    )
                    cache.append(
                        ResponseRecord(
                            model_id=model_id,
                            setting_id=setting.setting_id,
                            prompt_id=prompt.prompt_id,
                            generation_index=generation_index,
                            generation_seed=generation_seed(
                                manifest.random_seed,
                                model_id,
                                setting.setting_id,
                                prompt.prompt_id,
                                generation_index,
                            ),
                            response=response,
                            metadata={
                                "elapsed_seconds": 0.5,
                                "generated_tokens": 8,
                                "stop_reason": "eos_token",
                            },
                        )
                    )
    return cache


def test_collection_estimate_counts_reuse_storage_and_runtime(tmp_path: Path) -> None:
    source = _manifest(2)
    _complete_cache(tmp_path / "responses", source)
    target = replace(source, generations=4)

    report = estimate_collection(
        target,
        seed_count=2,
        reuse_cache_dirs=(tmp_path / "responses",),
        benchmark_cache_dir=tmp_path / "responses",
    )

    assert report["work"]["records_per_seed"] == 32
    assert report["work"]["total_records"] == 64
    assert report["work"]["reusable_records"] == 16
    assert report["work"]["remaining_records"] == 48
    assert report["runtime_projection"]["central_seconds"] == 24
    assert report["storage"]["embedding_bytes_uncompressed"] == 64 * 768 * 4


def test_collection_estimate_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="seed_count"):
        estimate_collection(_manifest(2), seed_count=0)


def test_response_quality_report_passes_a_complete_diverse_cache(tmp_path: Path) -> None:
    manifest = _manifest(2)
    dataset = _complete_cache(tmp_path / "responses", manifest).dataset

    report = response_quality_report(dataset)

    assert report["records"] == 16
    assert report["cell_count"] == 8
    assert report["overall"]["timed_records"] == 16
    assert report["ready_for_scale"] is True


def test_collection_progress_reports_incomplete_cache_without_gate_claim(
    tmp_path: Path,
) -> None:
    manifest = _manifest(2)
    complete = _complete_cache(tmp_path / "complete", manifest).dataset
    partial_cache = ResponseCache(tmp_path / "partial", manifest)
    for record in complete.records[:5]:
        partial_cache.append(record)

    report = collection_progress_report(partial_cache.dataset)

    assert report["records"] == 5
    assert report["expected_records"] == 16
    assert report["remaining_records"] == 11
    assert report["completion_fraction"] == 5 / 16
    assert report["final_quality_gate_eligible"] is False
    assert report["runtime_projection"]["remaining_inference_seconds"] == 5.5
    assert sum(row["records"] for row in report["models"]) == 5


def test_collection_progress_cli_accepts_incomplete_cache(tmp_path: Path) -> None:
    manifest = _manifest(2)
    manifest_path = manifest.save(tmp_path / "collection.json")
    complete = _complete_cache(tmp_path / "complete", manifest).dataset
    partial_path = tmp_path / "partial"
    partial_cache = ResponseCache(partial_path, manifest)
    partial_cache.append(complete.records[0])
    output = tmp_path / "progress.json"

    assert (
        main(
            [
                "collection-progress",
                "--manifest",
                str(manifest_path),
                "--cache-dir",
                str(partial_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.is_file()


def test_planning_and_quality_cli_write_reports(tmp_path: Path) -> None:
    manifest = _manifest(2)
    manifest_path = manifest.save(tmp_path / "collection.json")
    cache_path = tmp_path / "responses"
    _complete_cache(cache_path, manifest)
    estimate_path = tmp_path / "estimate.json"
    quality_path = tmp_path / "quality.json"

    assert (
        main(
            [
                "estimate-collection",
                "--manifest",
                str(manifest_path),
                "--benchmark-cache",
                str(cache_path),
                "--output",
                str(estimate_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "response-quality",
                "--manifest",
                str(manifest_path),
                "--cache-dir",
                str(cache_path),
                "--output",
                str(quality_path),
            ]
        )
        == 0
    )
    assert estimate_path.is_file()
    assert quality_path.is_file()
