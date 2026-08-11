import json
from pathlib import Path

import numpy as np
import pytest

from distdna.data import (
    CollectionManifest,
    DecodingSetting,
    HashingResponseEncoder,
    GeneratedResponse,
    Prompt,
    ResponseCache,
    ResponseDataset,
    ResponseRecord,
    build_embedding_datasets,
    collect_responses,
    generation_seed,
    save_embedding_datasets,
)


def manifest() -> CollectionManifest:
    return CollectionManifest(
        dataset_id="unit-test",
        model_ids=("m0", "m1"),
        settings=(
            DecodingSetting("low", 0.7, 0.9, 32),
            DecodingSetting("high", 1.0, 1.0, 32),
        ),
        prompts=(
            Prompt("c0", "calibration prompt", "calibration"),
            Prompt("e0", "first evaluation prompt", "evaluation"),
            Prompt("e1", "second evaluation prompt", "evaluation"),
        ),
        generations=3,
        random_seed=17,
    )


class CountingGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, model_id, prompt, setting, seed):
        self.calls += 1
        return f"shared words {model_id[-1]} {prompt.prompt_id} {setting.setting_id} {seed}"


class ProvenanceGenerator:
    def generate(self, model_id, prompt, setting, seed):
        return GeneratedResponse(
            text=f"response {model_id} {prompt.prompt_id} {setting.setting_id}",
            metadata={"model_revision": "a" * 40, "runtime": "unit-test"},
        )


def test_manifest_round_trip_and_seed_are_stable(tmp_path: Path) -> None:
    original = manifest()
    path = original.save(tmp_path / "collection.json")
    loaded = CollectionManifest.load(path)
    assert loaded == original
    assert loaded.fingerprint == original.fingerprint
    assert generation_seed(17, "m0", "low", "e0", 1) == generation_seed(
        17, "m0", "low", "e0", 1
    )
    assert generation_seed(17, "m0", "low", "e0", 1) != generation_seed(
        17, "m0", "low", "e0", 2
    )


def test_collection_resumes_and_encodes_canonical_tensors(tmp_path: Path) -> None:
    collection = manifest()
    generator = CountingGenerator()
    cache_dir = tmp_path / "responses"
    responses = collect_responses(collection, generator, cache_dir)
    assert responses.complete
    assert generator.calls == responses.expected_count

    resumed_generator = CountingGenerator()
    resumed = collect_responses(collection, resumed_generator, cache_dir)
    assert resumed.complete
    assert resumed_generator.calls == 0
    assert ResponseCache(cache_dir, collection).dataset.records == responses.records

    encoder = HashingResponseEncoder(24)
    datasets = build_embedding_datasets(collection, resumed, encoder)
    assert datasets["calibration"].shape == (2, 2, 1, 3, 24)
    assert datasets["evaluation"].shape == (2, 2, 2, 3, 24)
    assert np.isfinite(datasets["evaluation"].embeddings).all()
    assert set(datasets["calibration"].prompt_ids).isdisjoint(
        datasets["evaluation"].prompt_ids
    )

    output = save_embedding_datasets(
        datasets, collection, encoder, tmp_path / "embeddings"
    )
    assert (output / "calibration.npz").is_file()
    assert (output / "evaluation.npz").is_file()
    assert (output / "summary.json").is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        save_embedding_datasets(datasets, collection, encoder, output)


def test_cache_rejects_a_different_manifest(tmp_path: Path) -> None:
    original = manifest()
    ResponseCache(tmp_path / "responses", original)
    changed = CollectionManifest(
        dataset_id=original.dataset_id,
        model_ids=original.model_ids,
        settings=original.settings,
        prompts=original.prompts,
        generations=original.generations,
        random_seed=18,
    )
    with pytest.raises(ValueError, match="does not match"):
        ResponseCache(tmp_path / "responses", changed)


def test_manifest_loader_rejects_coerced_integer_types(tmp_path: Path) -> None:
    payload = manifest().as_dict()
    payload["generations"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="generations"):
        CollectionManifest.load(path)


def test_generated_response_provenance_is_persisted(tmp_path: Path) -> None:
    collection = manifest()
    responses = collect_responses(
        collection, ProvenanceGenerator(), tmp_path / "responses"
    )
    assert responses.records[0].metadata["model_revision"] == "a" * 40
    loaded = ResponseCache(tmp_path / "responses", collection).dataset
    assert loaded.records == responses.records


def test_external_seed_records_require_explicit_manifest_provenance() -> None:
    collection = manifest()
    record = ResponseRecord(
        model_id="m0",
        setting_id="low",
        prompt_id="c0",
        generation_index=0,
        generation_seed=42001,
        response="legacy response",
        metadata={
            "source_format": "llm-dna-responses-v1",
            "source_path": "m0_t07_p09_r1/responses.json",
            "source_sha256": "b" * 64,
            "model_revision": "a" * 40,
        },
        seed_scheme="external",
    )
    with pytest.raises(ValueError, match="explicitly allow"):
        ResponseDataset(collection, (record,))
