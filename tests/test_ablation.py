import json
from pathlib import Path

import pytest

from distdna.ablation import load_suite, write_ablation_suite
from distdna.config import ExperimentConfig


def _write_base(path: Path) -> None:
    payload = {
        "data": {
            "evaluation": "evaluation.npz",
            "calibration": "calibration.npz",
        },
        "experiment": {
            "generation_counts": [1, 2],
            "rff_dimensions": [32],
            "methods": [
                "single_sample_cosine",
                "mean_dna_cosine",
                "exact_mmd",
                "rfftrace",
            ],
            "normalization": "l2",
            "seed": 2027,
            "top_ks": [1, 3],
            "projection_dimension": None,
            "comparisons": [],
            "bandwidth": {"strategy": "median", "multiplier": 1.0},
        },
        "output": {"directory": "unused", "save_distances": True},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_ablation_suite_materializes_factorial_configs(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    _write_base(base)
    suite_path = write_ablation_suite(
        base,
        tmp_path / "suite",
        tmp_path / "results",
        normalizations=("l2", "none"),
        bandwidth_multipliers=(0.5, 2.0),
    )
    suite = load_suite(suite_path)
    assert len(suite["members"]) == 4
    assert len({item["output_dir"] for item in suite["members"]}) == 4
    configs = [ExperimentConfig.load(item["config"]) for item in suite["members"]]
    assert {config.normalization for config in configs} == {"l2", "none"}
    assert {config.bandwidth.multiplier for config in configs} == {0.5, 2.0}

    with pytest.raises(FileExistsError, match="already exists"):
        write_ablation_suite(
            base,
            tmp_path / "suite",
            tmp_path / "results",
            normalizations=("l2",),
            bandwidth_multipliers=(1.0,),
        )
