import csv
import json
from pathlib import Path

import numpy as np
import pytest

from distdna.config import ExperimentConfig
from distdna.data import EmbeddingDataset
from distdna.experiment import run_experiment, validate_experiment
from distdna.summary import aggregate_runs, summarize_run, write_summary


def write_inputs(tmp_path: Path, overlapping_prompts: bool = False) -> tuple[Path, Path]:
    rng = np.random.default_rng(22)
    models = ("m0", "m1", "m2")
    settings = ("low", "high")
    centers = np.asarray([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    def values(prompts: int) -> np.ndarray:
        result = np.empty((3, 2, prompts, 8, 2), dtype=np.float32)
        for model in range(3):
            for setting, noise in enumerate((0.05, 0.25)):
                result[model, setting] = centers[model] + rng.normal(
                    0.0, noise, size=(prompts, 8, 2)
                )
        return result

    eval_path = tmp_path / "evaluation.npz"
    cal_path = tmp_path / "calibration.npz"
    eval_prompts = ("e0", "e1", "e2")
    cal_prompts = ("e0", "c1") if overlapping_prompts else ("c0", "c1")
    EmbeddingDataset(values(3), models, settings, eval_prompts).save(eval_path)
    EmbeddingDataset(values(2), models, settings, cal_prompts).save(cal_path)
    return eval_path, cal_path


def config_for(
    tmp_path: Path,
    eval_path: Path,
    cal_path: Path,
    *,
    seed: int = 9,
    output_name: str = "results",
    projection_dimension: int = 8,
) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "data": {"evaluation": str(eval_path), "calibration": str(cal_path)},
            "experiment": {
                "generation_counts": [1, 4],
                "rff_dimensions": [32],
                "methods": [
                    "single_sample_cosine",
                    "mean_dna_cosine",
                    "exact_mmd",
                    "rfftrace",
                ],
                "normalization": "l2",
                "seed": seed,
                "top_ks": [1, 3, 5],
                "projection_dimension": projection_dimension,
                "comparisons": [
                    {"query": "low", "reference": "low"},
                    {"query": "high", "reference": "low"},
                ],
                "bandwidth": {"strategy": "median", "max_pairs": 1000},
            },
            "output": {
                "directory": str(tmp_path / output_name),
                "save_distances": True,
            },
        }
    )


def test_end_to_end_artifact_bundle(tmp_path: Path) -> None:
    eval_path, cal_path = write_inputs(tmp_path)
    config = config_for(tmp_path, eval_path, cal_path)
    result = run_experiment(config)
    assert result.output_dir == tmp_path / "results"
    assert result.metric_rows == 2 * (3 + 1) * 2  # R x (3 baselines + 1 RFF D) x comparisons
    assert (result.output_dir / "parameters" / "rff_D32_seed9.npz").is_file()
    assert (result.output_dir / "distances.npz").is_file()

    with (result.output_dir / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    assert len(metrics) == result.metric_rows
    assert {row["comparison_type"] for row in metrics} == {"same_setting", "cross_setting"}
    assert {row["method"] for row in metrics} == {
        "single_sample_cosine",
        "mean_dna_cosine",
        "exact_mmd",
        "rfftrace",
    }
    with (result.output_dir / "ranks.csv").open(newline="", encoding="utf-8") as stream:
        ranks = list(csv.DictReader(stream))
    assert {
        row["projection_dimension"]
        for row in ranks
        if row["method"] == "rfftrace"
    } == {"8"}

    exact_same = next(
        row
        for row in metrics
        if row["method"] == "exact_mmd"
        and row["generations"] == "4"
        and row["comparison_type"] == "same_setting"
    )
    with np.load(result.output_dir / "distances.npz", allow_pickle=False) as payload:
        same_setting_distances = payload[exact_same["run_key"]]
    assert np.all(np.diag(same_setting_distances) > 0.0)

    with (result.output_dir / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    assert metadata["bandwidth_selection_split"] == "calibration"
    assert metadata["inputs"]["evaluation"]["sha256"]
    assert "structurally inapplicable" in metadata["na_policy"]

    with pytest.raises(FileExistsError, match="already exists"):
        run_experiment(config)

    summary = summarize_run(result.output_dir)
    assert summary["input_hashes"] == {
        "calibration": metadata["inputs"]["calibration"]["sha256"],
        "evaluation": metadata["inputs"]["evaluation"]["sha256"],
    }
    assert summary["same_setting_evaluation"]["policy"] == "disjoint_generation_pools"
    assert summary["retrieval"]
    assert summary["rff_approximation"]
    assert {
        item["projection_dimension"]
        for item in summary["retrieval"]
        if item["method"] == "rfftrace"
    } == {8}
    assert {
        item["projection_dimension"]
        for item in summary["retrieval"]
        if item["method"] != "rfftrace"
    } == {None}
    assert {
        item["projection_dimension"] for item in summary["rff_approximation"]
    } == {8}
    assert set(summary["pilot_checks"]["checks"]) == {
        "all_methods_present",
        "same_and_cross_setting_present",
        "generation_sweep_present",
        "non_ceiling_retrieval_observed",
        "nondegenerate_rff_exact_diagnostics_present",
    }
    assert "outperforms" in summary["pilot_checks"]["interpretation"]
    assert all(
        item["distance_correlation"] is None
        or -1.0 <= item["distance_correlation"] <= 1.0
        for item in summary["rff_approximation"]
    )
    summary_path = write_summary(summary, result.output_dir / "summary.json")
    assert summary_path.is_file()
    aggregate = aggregate_runs([result.output_dir])
    assert aggregate["run_count"] == 1
    assert aggregate["seeds"] == [9]
    assert aggregate["unique_seed_count"] == 1
    assert aggregate["unique_input_count"] == 1
    assert not aggregate["scale_readiness"]["ready_for_scale"]
    assert not aggregate["scale_readiness"]["checks"]["at_least_two_unique_seeds"]
    assert not aggregate["scale_readiness"]["checks"][
        "at_least_two_distinct_input_datasets"
    ]
    assert all(item["top_1_std"] == 0.0 for item in aggregate["retrieval"])

    repeated_map = run_experiment(
        config_for(
            tmp_path,
            eval_path,
            cal_path,
            seed=10,
            output_name="results-seed10",
        )
    )
    same_data_aggregate = aggregate_runs([result.output_dir, repeated_map.output_dir])
    assert same_data_aggregate["unique_seed_count"] == 2
    assert same_data_aggregate["unique_input_count"] == 1
    assert not same_data_aggregate["scale_readiness"]["ready_for_scale"]
    assert not same_data_aggregate["scale_readiness"]["checks"][
        "at_least_two_distinct_input_datasets"
    ]

    second_projection = run_experiment(
        config_for(
            tmp_path,
            eval_path,
            cal_path,
            seed=11,
            output_name="results-projection4",
            projection_dimension=4,
        )
    )
    projection_aggregate = aggregate_runs(
        [result.output_dir, second_projection.output_dir]
    )
    rff_rows = [
        item for item in projection_aggregate["retrieval"]
        if item["method"] == "rfftrace"
    ]
    assert {item["projection_dimension"] for item in rff_rows} == {4, 8}
    assert all(item["runs"] == 1 for item in rff_rows)


def test_overlapping_prompt_splits_are_rejected(tmp_path: Path) -> None:
    eval_path, cal_path = write_inputs(tmp_path, overlapping_prompts=True)
    config = config_for(tmp_path, eval_path, cal_path)
    with pytest.raises(ValueError, match="disjoint"):
        validate_experiment(config)


def test_same_setting_requires_two_disjoint_generation_pools(tmp_path: Path) -> None:
    eval_path, cal_path = write_inputs(tmp_path)
    with np.load(eval_path, allow_pickle=False) as payload:
        truncated = EmbeddingDataset(
            payload["embeddings"][..., :4, :],
            tuple(payload["model_ids"].tolist()),
            tuple(payload["setting_ids"].tolist()),
            tuple(payload["prompt_ids"].tolist()),
        )
    truncated.save(eval_path)
    config = config_for(tmp_path, eval_path, cal_path)
    with pytest.raises(ValueError, match="disjoint query/reference pools"):
        validate_experiment(config)


def test_projection_sweep_keeps_every_variant_separate(tmp_path: Path) -> None:
    eval_path, cal_path = write_inputs(tmp_path)
    base = config_for(tmp_path, eval_path, cal_path).as_dict()
    base["experiment"].pop("projection_dimension")
    base["experiment"]["projection_dimensions"] = [None, 4, 8]
    base["output"]["directory"] = str(tmp_path / "projection-sweep")
    result = run_experiment(ExperimentConfig.from_dict(base))

    summary = summarize_run(result.output_dir)
    rff_rows = [item for item in summary["retrieval"] if item["method"] == "rfftrace"]
    assert {item["projection_dimension"] for item in rff_rows} == {None, 4, 8}
    assert len(rff_rows) == 3 * 2 * 2
    assert {
        item["projection_dimension"] for item in summary["rff_approximation"]
    } == {None, 4, 8}
    assert {
        item["projection_dimension"] for item in summary["projection_approximation"]
    } == {4, 8}
    aggregate = aggregate_runs([result.output_dir])
    assert {
        item["projection_dimension"] for item in aggregate["rff_approximation"]
    } == {None, 4, 8}
    assert {
        item["projection_dimension"]
        for item in aggregate["projection_approximation"]
    } == {4, 8}
    assert (
        result.output_dir / "parameters" / "rff_D32_Lnone_seed9.npz"
    ).is_file()
    assert (
        result.output_dir / "parameters" / "rff_D32_L4_seed9.npz"
    ).is_file()
    assert (
        result.output_dir / "parameters" / "rff_D32_L8_seed9.npz"
    ).is_file()
