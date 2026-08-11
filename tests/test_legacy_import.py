from __future__ import annotations

import json
from pathlib import Path

import pytest

from distdna.cli import main
from distdna.data import (
    CollectionManifest,
    DecodingSetting,
    Prompt,
    audit_llm_dna_responses,
    import_llm_dna_responses,
)


REVISION = "a" * 40


def legacy_manifest(*, allow_external: bool = True) -> CollectionManifest:
    metadata = {"model_revisions": {"org/model-a": REVISION}}
    if allow_external:
        metadata["allow_external_seed_records"] = True
    return CollectionManifest(
        dataset_id="legacy-import-test",
        model_ids=("org/model-a",),
        settings=(DecodingSetting("temp_0_2_p_0_8", 0.2, 0.8, 32),),
        prompts=(
            Prompt("cal_0", "calibration text", "calibration"),
            Prompt("eval_0", "evaluation text", "evaluation"),
        ),
        generations=2,
        random_seed=2027,
        metadata=metadata,
    )


def write_legacy_run(
    root: Path,
    repeat: int,
    *,
    seed: int | None = None,
    revision: str | None = REVISION,
    model_token: str = "org_model-a",
) -> None:
    run_dir = root / f"{model_token}_t02_p08_r{repeat}"
    run_dir.mkdir(parents=True)
    payload = {
        "items": [
            {"prompt": "calibration text", "response": f"calibration {repeat}"},
            {"prompt": "evaluation text", "response": f"evaluation {repeat}"},
        ]
    }
    if seed is not None:
        payload["generation_seed"] = seed
    if revision is not None:
        payload["model_revision"] = revision
    (run_dir / "responses.json").write_text(json.dumps(payload), encoding="utf-8")


def test_complete_legacy_collection_imports_atomically(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    write_legacy_run(source, 1, seed=42001)
    write_legacy_run(source, 2, seed=42002)
    manifest = legacy_manifest()

    report = audit_llm_dna_responses(source, manifest)
    assert report.ready_for_import
    assert report.error_count == 0
    assert report.candidate_records == report.expected_records == 4

    cache_dir = tmp_path / "cache"
    dataset, imported_report = import_llm_dna_responses(
        source, manifest, cache_dir
    )
    assert imported_report.ready_for_import
    assert dataset.complete
    assert {record.generation_seed for record in dataset.records} == {42001, 42002}
    assert {record.seed_scheme for record in dataset.records} == {"external"}
    assert all(record.metadata["model_revision"] == REVISION for record in dataset.records)
    assert all(len(record.metadata["source_sha256"]) == 64 for record in dataset.records)
    assert (cache_dir / "manifest.json").is_file()
    assert (cache_dir / "responses.jsonl").is_file()

    with pytest.raises(FileExistsError, match="already exists"):
        import_llm_dna_responses(source, manifest, cache_dir)


def test_audit_quarantines_unknown_seed_and_revision(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    write_legacy_run(source, 1, seed=None, revision=None)
    write_legacy_run(source, 2, seed=None, revision=None)

    report = audit_llm_dna_responses(source, legacy_manifest())
    codes = {issue.code for issue in report.issues}
    assert not report.ready_for_import
    assert "generation_seed_missing" in codes
    assert "model_revision_missing" in codes
    assert "missing_generation_keys" in codes
    assert report.candidate_records == 0


def test_audit_rejects_incomplete_collection_and_unapproved_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy"
    write_legacy_run(source, 1, seed=42001)

    report = audit_llm_dna_responses(
        source, legacy_manifest(allow_external=False)
    )
    codes = {issue.code for issue in report.issues}
    assert not report.ready_for_import
    assert report.missing_records == 2
    assert "external_records_not_allowed" in codes


def test_explicit_alias_and_cli_audit_import(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    write_legacy_run(source, 1, seed=42001, model_token="student-token")
    write_legacy_run(source, 2, seed=42002, model_token="student-token")
    manifest_path = legacy_manifest().save(tmp_path / "manifest.json")
    aliases_path = tmp_path / "aliases.json"
    aliases_path.write_text(
        json.dumps({"student-token": "org/model-a"}), encoding="utf-8"
    )
    report_path = tmp_path / "audit.json"

    assert (
        main(
            [
                "audit-legacy",
                "--source-dir",
                str(source),
                "--manifest",
                str(manifest_path),
                "--model-aliases",
                str(aliases_path),
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    assert json.loads(report_path.read_text(encoding="utf-8"))["ready_for_import"]

    cache_dir = tmp_path / "cache"
    assert (
        main(
            [
                "import-legacy",
                "--source-dir",
                str(source),
                "--manifest",
                str(manifest_path),
                "--model-aliases",
                str(aliases_path),
                "--cache-dir",
                str(cache_dir),
            ]
        )
        == 0
    )
    assert (cache_dir / "responses.jsonl").is_file()
