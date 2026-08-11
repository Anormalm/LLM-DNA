import json
from pathlib import Path

from distdna.config import ExperimentConfig
from distdna.data import CollectionManifest, DecodingSetting, Prompt
from distdna.decoding import build_decoding_report
from distdna.experiment import run_experiment
from distdna.relationships import build_relationship_report
from distdna.summary import aggregate_runs

from test_experiment import write_inputs


def _manifest(path: Path) -> Path:
    manifest = CollectionManifest(
        dataset_id="decoding-report-test",
        model_ids=("m0", "m1", "m2"),
        settings=(
            DecodingSetting("low", temperature=0.0, top_p=1.0),
            DecodingSetting("high", temperature=0.7, top_p=0.9),
        ),
        prompts=(
            Prompt("c0", "calibration zero", "calibration"),
            Prompt("c1", "calibration one", "calibration"),
            Prompt("e0", "evaluation zero", "evaluation"),
            Prompt("e1", "evaluation one", "evaluation"),
            Prompt("e2", "evaluation two", "evaluation"),
        ),
        generations=8,
    )
    return manifest.save(path)


def test_decoding_report_requires_and_summarizes_complete_grid(tmp_path: Path) -> None:
    evaluation, calibration = write_inputs(tmp_path)
    payload = {
        "data": {"evaluation": str(evaluation), "calibration": str(calibration)},
        "experiment": {
            "generation_counts": [4],
            "rff_dimensions": [32],
            "methods": [
                "single_sample_cosine",
                "mean_dna_cosine",
                "exact_mmd",
                "rfftrace",
            ],
            "normalization": "l2",
            "seed": 9,
            "top_ks": [1, 3, 5],
            "comparisons": [
                {"query": "low", "reference": "low"},
                {"query": "high", "reference": "high"},
                {"query": "high", "reference": "low"},
            ],
            "bandwidth": {"strategy": "median", "max_pairs": 1000},
        },
        "output": {"directory": str(tmp_path / "run"), "save_distances": True},
    }
    result = run_experiment(ExperimentConfig.from_dict(payload))
    aggregate = aggregate_runs([result.output_dir])
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")

    report = build_decoding_report(
        aggregate_path,
        _manifest(tmp_path / "manifest.json"),
        generations=4,
        rff_dimension=32,
    )

    assert report["cell_count"] == 12
    assert report["protocol"]["deterministic_reference"] == "low"
    assert {row["evaluation"] for row in report["cells"]} == {
        "same_setting",
        "cross_to_deterministic",
    }
    assert len(report["temperature_effects"]) == 8
    assert len(report["top_p_effects"]) == 8
    assert len(report["effect_ranges"]) == 8
    assert report["aggregate_sha256"]


def test_relationship_report_uses_saved_complete_distance_grid(tmp_path: Path) -> None:
    evaluation, calibration = write_inputs(tmp_path)
    payload = {
        "data": {"evaluation": str(evaluation), "calibration": str(calibration)},
        "experiment": {
            "generation_counts": [4],
            "rff_dimensions": [32],
            "methods": [
                "single_sample_cosine",
                "mean_dna_cosine",
                "exact_mmd",
                "rfftrace",
            ],
            "normalization": "l2",
            "seed": 9,
            "top_ks": [1, 3, 5],
            "comparisons": [
                {"query": "low", "reference": "low"},
                {"query": "high", "reference": "high"},
                {"query": "high", "reference": "low"},
            ],
            "bandwidth": {"strategy": "median", "max_pairs": 1000},
        },
        "output": {"directory": str(tmp_path / "run"), "save_distances": True},
    }
    result = run_experiment(ExperimentConfig.from_dict(payload))
    manifest_path = _manifest(tmp_path / "manifest.json")
    relationship_map = tmp_path / "relationships.json"
    relationship_map.write_text(
        json.dumps(
            {
                "format_version": 1,
                "groups": {"m0": "family-a", "m1": "family-a", "m2": "family-b"},
            }
        ),
        encoding="utf-8",
    )

    report = build_relationship_report(
        [result.output_dir],
        manifest_path,
        relationship_map,
        generations=4,
        rff_dimension=32,
    )

    assert report["run_count"] == 1
    assert report["cell_count"] == 12
    assert len(report["summary"]) == 8
    assert report["chance_baselines"] == {
        "roc_auc": 0.5,
        "nearest_group_accuracy": 0.5,
    }
    assert all(0.0 <= row["roc_auc_mean"] <= 1.0 for row in report["summary"])
    assert all(
        row["nearest_group_eligible_models"] == 2 for row in report["cells"]
    )
