import json
from pathlib import Path

import numpy as np

from distdna import RFFTraceExtractionConfig, calc_rfftrace
from distdna.cli import main_calc_rfftrace
from distdna.demo import create_demo
from distdna.dna import DNASignature


def test_public_api_extracts_and_saves_llm_dna_style_artifacts(tmp_path: Path) -> None:
    config_path = create_demo(tmp_path / "data")
    output_dir = tmp_path / "out"
    result = calc_rfftrace(
        RFFTraceExtractionConfig(
            evaluation_path=config_path.parent / "evaluation.npz",
            calibration_path=config_path.parent / "calibration.npz",
            settings=("deterministic", "temp_1_0"),
            generations=4,
            rff_dimension=32,
            dna_dimension=8,
            random_seed=11,
            output_dir=output_dir,
        )
    )
    assert result.vectors.shape == (10, 8)
    assert result.collection_path == output_dir / "rfftrace_signatures.npz"
    assert result.parameters_path == output_dir / "rff_parameters_D32_seed11.npz"
    assert result.summary_path == output_dir / "summary.json"
    assert len(list((output_dir / "signatures").glob("*.npz"))) == 10

    signature_path = next((output_dir / "signatures").glob("*.npz"))
    restored = DNASignature.load(signature_path)
    assert restored.dimension == 8
    assert restored.metadata.extractor_config["shared_feature_map"] is True
    assert np.isfinite(restored.vector).all()

    with result.summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    assert summary["shared_feature_map"] is True
    assert summary["vector_shape"] == [10, 8]


def test_flat_calc_rfftrace_cli_matches_llm_dna_style_entrypoint(tmp_path: Path) -> None:
    config_path = create_demo(tmp_path / "data")
    output_dir = tmp_path / "cli-out"
    exit_code = main_calc_rfftrace(
        [
            "--evaluation",
            str(config_path.parent / "evaluation.npz"),
            "--calibration",
            str(config_path.parent / "calibration.npz"),
            "--settings",
            "deterministic",
            "--generations",
            "2",
            "--rff-dim",
            "16",
            "--dna-dim",
            "6",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0
    assert (output_dir / "rfftrace_signatures.npz").is_file()
    assert (output_dir / "summary.json").is_file()

