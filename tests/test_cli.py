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
