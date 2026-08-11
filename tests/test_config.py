from pathlib import Path

import pytest

from distdna.config import ExperimentConfig


def minimal_payload() -> dict:
    return {
        "data": {"evaluation": "eval.npz", "calibration": "cal.npz"},
        "experiment": {
            "generation_counts": [1],
            "rff_dimensions": [32],
            "methods": ["exact_mmd", "rfftrace"],
        },
        "output": {"directory": "results"},
    }


def test_relative_paths_resolve_from_config_directory(tmp_path: Path) -> None:
    config = ExperimentConfig.from_dict(minimal_payload(), base_dir=tmp_path)
    assert config.evaluation_path == (tmp_path / "eval.npz").resolve()
    assert config.output_dir == (tmp_path / "results").resolve()


def test_unknown_keys_and_invalid_bandwidth_fail_fast() -> None:
    payload = minimal_payload()
    payload["experiment"]["typo"] = True
    with pytest.raises(ValueError, match="unknown experiment"):
        ExperimentConfig.from_dict(payload)

    payload = minimal_payload()
    payload["experiment"]["bandwidth"] = {"strategy": "fixed"}
    with pytest.raises(ValueError, match="requires a value"):
        ExperimentConfig.from_dict(payload)

    payload = minimal_payload()
    payload["output"]["save_distances"] = "false"
    with pytest.raises(ValueError, match="JSON boolean"):
        ExperimentConfig.from_dict(payload)

    payload = minimal_payload()
    payload["experiment"]["top_ks"] = [3, 5]
    with pytest.raises(ValueError, match="must include 1"):
        ExperimentConfig.from_dict(payload)
