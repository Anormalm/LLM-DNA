import json
from pathlib import Path

from distdna.cli import main
from distdna.data import CollectionManifest, DecodingSetting, Prompt


def test_demo_cli_creates_and_runs_complete_bundle(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demo"
    assert main(["demo", "--output-dir", str(demo_dir), "--run"]) == 0
    assert (demo_dir / "pilot.json").is_file()
    assert (demo_dir / "results" / "metrics.csv").is_file()
    assert main(["validate-config", str(demo_dir / "pilot.json")]) == 0
    assert (
        main(
            [
                "summarize",
                str(demo_dir / "results"),
                "--output",
                str(demo_dir / "results" / "summary.json"),
            ]
        )
        == 0
    )
    assert (demo_dir / "results" / "summary.json").is_file()


def test_reseed_manifest_preserves_protocol_and_records_parent(tmp_path: Path) -> None:
    source = CollectionManifest(
        dataset_id="repeat-test",
        model_ids=("m0",),
        settings=(DecodingSetting("sample", 0.7, 0.9, 32),),
        prompts=(
            Prompt("c0", "calibration", "calibration"),
            Prompt("e0", "evaluation", "evaluation"),
        ),
        generations=4,
        random_seed=2027,
        metadata={"model_revisions": {"m0": "a" * 40}},
    )
    source_path = source.save(tmp_path / "source.json")
    output_path = tmp_path / "seed2028.json"

    assert (
        main(
            [
                "reseed-manifest",
                "--manifest",
                str(source_path),
                "--seed",
                "2028",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    repeated = CollectionManifest.load(output_path)
    assert repeated.random_seed == 2028
    assert repeated.model_ids == source.model_ids
    assert repeated.settings == source.settings
    assert repeated.prompts == source.prompts
    assert repeated.generations == source.generations
    assert repeated.metadata["model_revisions"] == source.metadata["model_revisions"]
    assert repeated.metadata["parent_manifest_fingerprint"] == source.fingerprint
    assert repeated.metadata["parent_random_seed"] == 2027
    assert repeated.fingerprint != source.fingerprint


def test_reseed_manifest_rejects_same_seed_and_existing_output(tmp_path: Path) -> None:
    source = CollectionManifest(
        dataset_id="repeat-test",
        model_ids=("m0",),
        settings=(DecodingSetting("sample", 0.7, 0.9),),
        prompts=(
            Prompt("c0", "calibration", "calibration"),
            Prompt("e0", "evaluation", "evaluation"),
        ),
        generations=2,
        random_seed=7,
    )
    source_path = source.save(tmp_path / "source.json")
    output_path = tmp_path / "repeat.json"

    assert (
        main(
            [
                "reseed-manifest",
                "--manifest",
                str(source_path),
                "--seed",
                "7",
                "--output",
                str(output_path),
            ]
        )
        == 2
    )
    output_path.write_text("occupied", encoding="utf-8")
    assert (
        main(
            [
                "reseed-manifest",
                "--manifest",
                str(source_path),
                "--seed",
                "8",
                "--output",
                str(output_path),
            ]
        )
        == 2
    )


def test_collect_local_rejects_non_positive_progress_interval(tmp_path: Path) -> None:
    manifest = CollectionManifest(
        dataset_id="progress-test",
        model_ids=("m0",),
        settings=(DecodingSetting("sample", 0.7, 0.9),),
        prompts=(
            Prompt("c0", "calibration", "calibration"),
            Prompt("e0", "evaluation", "evaluation"),
        ),
        generations=2,
        metadata={"model_revisions": {"m0": "a" * 40}},
    )
    manifest_path = manifest.save(tmp_path / "collection.json")

    assert (
        main(
            [
                "collect-local",
                "--manifest",
                str(manifest_path),
                "--cache-dir",
                str(tmp_path / "responses"),
                "--progress-every",
                "0",
            ]
        )
        == 2
    )


def test_resize_manifest_preserves_protocol_and_records_parent(tmp_path: Path) -> None:
    source = CollectionManifest(
        dataset_id="full-test",
        model_ids=("m0",),
        settings=(DecodingSetting("sample", 0.7, 0.9, 32),),
        prompts=(
            Prompt("c0", "calibration", "calibration"),
            Prompt("e0", "evaluation", "evaluation"),
        ),
        generations=32,
        random_seed=2027,
        metadata={"model_revisions": {"m0": "a" * 40}},
    )
    source_path = source.save(tmp_path / "source.json")
    output_path = tmp_path / "sentinel.json"

    assert (
        main(
            [
                "resize-manifest",
                "--manifest",
                str(source_path),
                "--generations",
                "1",
                "--dataset-id",
                "sentinel-test",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    sentinel = CollectionManifest.load(output_path)
    assert sentinel.dataset_id == "sentinel-test"
    assert sentinel.generations == 1
    assert sentinel.random_seed == source.random_seed
    assert sentinel.model_ids == source.model_ids
    assert sentinel.settings == source.settings
    assert sentinel.prompts == source.prompts
    assert sentinel.metadata["model_revisions"] == source.metadata["model_revisions"]
    assert sentinel.metadata["parent_manifest_fingerprint"] == source.fingerprint
    assert sentinel.metadata["parent_generations"] == 32


def test_resize_manifest_rejects_invalid_or_unchanged_generations(tmp_path: Path) -> None:
    source = CollectionManifest(
        dataset_id="full-test",
        model_ids=("m0",),
        settings=(DecodingSetting("sample", 0.7, 0.9),),
        prompts=(
            Prompt("c0", "calibration", "calibration"),
            Prompt("e0", "evaluation", "evaluation"),
        ),
        generations=2,
    )
    source_path = source.save(tmp_path / "source.json")
    for generations in (0, 2):
        assert (
            main(
                [
                    "resize-manifest",
                    "--manifest",
                    str(source_path),
                    "--generations",
                    str(generations),
                    "--dataset-id",
                    "sentinel-test",
                    "--output",
                    str(tmp_path / f"sentinel-{generations}.json"),
                ]
            )
            == 2
        )


def test_revise_manifest_changes_only_prompt_text_and_records_provenance(
    tmp_path: Path,
) -> None:
    source = CollectionManifest(
        dataset_id="source-test",
        model_ids=("m0",),
        settings=(DecodingSetting("sample", 0.7, 0.9, 32),),
        prompts=(
            Prompt("c0", "calibration", "calibration"),
            Prompt("e0", "evaluation", "evaluation"),
        ),
        generations=4,
        metadata={"model_revisions": {"m0": "a" * 40}},
    )
    source_path = source.save(tmp_path / "source.json")
    revision_path = tmp_path / "revisions.json"
    revision_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "revision_id": "bounded-v2",
                "reason": "make response length measurable",
                "revisions": {
                    "c0": "Calibration in exactly two sentences.",
                    "e0": "Evaluation in exactly two sentences.",
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "revised.json"

    assert (
        main(
            [
                "revise-manifest",
                "--manifest",
                str(source_path),
                "--prompt-revisions",
                str(revision_path),
                "--dataset-id",
                "revised-test",
                "--require-all-prompts",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    revised = CollectionManifest.load(output)
    assert revised.dataset_id == "revised-test"
    assert revised.model_ids == source.model_ids
    assert revised.settings == source.settings
    assert revised.generations == source.generations
    assert revised.random_seed == source.random_seed
    assert [item.prompt_id for item in revised.prompts] == ["c0", "e0"]
    assert [item.split for item in revised.prompts] == ["calibration", "evaluation"]
    assert revised.metadata["parent_manifest_fingerprint"] == source.fingerprint
    assert revised.metadata["prompt_revision"]["changed_prompt_ids"] == ["c0", "e0"]


def test_revise_manifest_rejects_unknown_or_missing_prompt_ids(tmp_path: Path) -> None:
    source = CollectionManifest(
        dataset_id="source-test",
        model_ids=("m0",),
        settings=(DecodingSetting("sample", 0.7, 0.9),),
        prompts=(
            Prompt("c0", "calibration", "calibration"),
            Prompt("e0", "evaluation", "evaluation"),
        ),
        generations=2,
    )
    source_path = source.save(tmp_path / "source.json")
    for label, revisions in (
        ("unknown", {"c0": "changed", "e0": "also changed", "other": "bad"}),
        ("missing", {"c0": "changed"}),
    ):
        revision_path = tmp_path / f"{label}.json"
        revision_path.write_text(
            json.dumps(
                {
                    "revision_id": label,
                    "reason": "test rejection",
                    "revisions": revisions,
                }
            ),
            encoding="utf-8",
        )
        assert (
            main(
                [
                    "revise-manifest",
                    "--manifest",
                    str(source_path),
                    "--prompt-revisions",
                    str(revision_path),
                    "--dataset-id",
                    f"{label}-test",
                    "--require-all-prompts",
                    "--output",
                    str(tmp_path / f"{label}-output.json"),
                ]
            )
            == 2
        )


def test_set_token_limit_preserves_factorial_and_records_parent(tmp_path: Path) -> None:
    source = CollectionManifest(
        dataset_id="source-test",
        model_ids=("m0", "m1"),
        settings=(
            DecodingSetting("deterministic", 0.0, 1.0, 128),
            DecodingSetting("sample", 0.7, 0.9, 128),
        ),
        prompts=(
            Prompt("c0", "calibration", "calibration"),
            Prompt("e0", "evaluation", "evaluation"),
        ),
        generations=4,
        metadata={"model_revisions": {"m0": "a" * 40, "m1": "b" * 40}},
    )
    source_path = source.save(tmp_path / "source.json")
    output = tmp_path / "token-adjusted.json"

    assert (
        main(
            [
                "set-token-limit",
                "--manifest",
                str(source_path),
                "--max-new-tokens",
                "256",
                "--dataset-id",
                "token-adjusted-test",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    adjusted = CollectionManifest.load(output)
    assert [item.setting_id for item in adjusted.settings] == [
        "deterministic",
        "sample",
    ]
    assert [item.temperature for item in adjusted.settings] == [0.0, 0.7]
    assert [item.top_p for item in adjusted.settings] == [1.0, 0.9]
    assert {item.max_new_tokens for item in adjusted.settings} == {256}
    assert adjusted.model_ids == source.model_ids
    assert adjusted.prompts == source.prompts
    assert adjusted.metadata["parent_manifest_fingerprint"] == source.fingerprint
    assert adjusted.metadata["parent_max_new_tokens"] == [128]


def test_subset_manifest_filters_roster_revisions_and_records_parent(
    tmp_path: Path,
) -> None:
    source = CollectionManifest(
        dataset_id="source-test",
        model_ids=("m0", "m1", "m2"),
        settings=(DecodingSetting("sample", 0.7, 0.9, 128),),
        prompts=(
            Prompt("c0", "calibration", "calibration"),
            Prompt("e0", "evaluation", "evaluation"),
        ),
        generations=4,
        metadata={
            "model_revisions": {
                "m0": "a" * 40,
                "m1": "b" * 40,
                "m2": "c" * 40,
            }
        },
    )
    source_path = source.save(tmp_path / "source.json")
    output = tmp_path / "subset.json"

    assert (
        main(
            [
                "subset-manifest",
                "--manifest",
                str(source_path),
                "--model",
                "m2",
                "--model",
                "m0",
                "--dataset-id",
                "subset-test",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    subset = CollectionManifest.load(output)
    assert subset.model_ids == ("m2", "m0")
    assert subset.metadata["model_revisions"] == {
        "m2": "c" * 40,
        "m0": "a" * 40,
    }
    assert subset.settings == source.settings
    assert subset.prompts == source.prompts
    assert subset.metadata["parent_manifest_fingerprint"] == source.fingerprint
    assert subset.metadata["parent_model_ids"] == ["m0", "m1", "m2"]


def test_set_token_limit_rejects_invalid_unchanged_and_existing_output(
    tmp_path: Path,
) -> None:
    source = CollectionManifest(
        dataset_id="source-test",
        model_ids=("m0",),
        settings=(DecodingSetting("sample", 0.7, 0.9, 128),),
        prompts=(
            Prompt("c0", "calibration", "calibration"),
            Prompt("e0", "evaluation", "evaluation"),
        ),
        generations=2,
    )
    source_path = source.save(tmp_path / "source.json")
    for token_limit in (0, 128):
        assert (
            main(
                [
                    "set-token-limit",
                    "--manifest",
                    str(source_path),
                    "--max-new-tokens",
                    str(token_limit),
                    "--dataset-id",
                    "adjusted-test",
                    "--output",
                    str(tmp_path / f"adjusted-{token_limit}.json"),
                ]
            )
            == 2
        )
    occupied = tmp_path / "occupied.json"
    occupied.write_text("occupied", encoding="utf-8")
    assert (
        main(
            [
                "set-token-limit",
                "--manifest",
                str(source_path),
                "--max-new-tokens",
                "256",
                "--dataset-id",
                "adjusted-test",
                "--output",
                str(occupied),
            ]
        )
        == 2
    )


def test_subset_manifest_rejects_duplicate_unknown_and_full_rosters(
    tmp_path: Path,
) -> None:
    source = CollectionManifest(
        dataset_id="source-test",
        model_ids=("m0", "m1"),
        settings=(DecodingSetting("sample", 0.7, 0.9),),
        prompts=(
            Prompt("c0", "calibration", "calibration"),
            Prompt("e0", "evaluation", "evaluation"),
        ),
        generations=2,
        metadata={"model_revisions": {"m0": "a" * 40, "m1": "b" * 40}},
    )
    source_path = source.save(tmp_path / "source.json")
    cases = (
        ("duplicate", ("m0", "m0")),
        ("unknown", ("m0", "unknown")),
        ("full", ("m1", "m0")),
    )
    for label, models in cases:
        arguments = [
            "subset-manifest",
            "--manifest",
            str(source_path),
        ]
        for model_id in models:
            arguments.extend(("--model", model_id))
        arguments.extend(
            (
                "--dataset-id",
                f"{label}-test",
                "--output",
                str(tmp_path / f"{label}.json"),
            )
        )
        assert main(arguments) == 2
