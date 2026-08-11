import json
from pathlib import Path

from distdna.figures import render_figures


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_render_complete_figure_bundle_with_provenance(tmp_path: Path) -> None:
    methods = ("single_sample_cosine", "mean_dna_cosine", "exact_mmd", "rfftrace")
    retrieval = []
    for method in methods:
        for comparison in ("same_setting", "cross_setting"):
            for generations in (1, 4):
                retrieval.append(
                    {
                        "normalization": "l2",
                        "bandwidth_multiplier": 1.0,
                        "method": method,
                        "comparison_type": comparison,
                        "generations": generations,
                        "rff_dimension": 512 if method == "rfftrace" else None,
                        "projection_dimension": None,
                        "top_1_mean": 0.5 + generations / 16,
                        "top_1_std": 0.02,
                    }
                )
    feature = {
        "retrieval": retrieval,
        "rff_approximation": [
            {
                "normalization": "l2",
                "bandwidth_multiplier": 1.0,
                "comparison_type": comparison,
                "generations": 4,
                "rff_dimension": dimension,
                "projection_dimension": None,
                "distance_correlation_mean": correlation,
                "distance_correlation_std": 0.01,
            }
            for comparison in ("same_setting", "cross_setting")
            for dimension, correlation in ((16, 0.82), (512, 0.995))
        ],
    }
    projection = {
        "projection_approximation": [
            {
                "normalization": "l2",
                "bandwidth_multiplier": 1.0,
                "comparison_type": comparison,
                "generations": 4,
                "rff_dimension": 512,
                "projection_dimension": dimension,
                "distance_correlation_mean": correlation,
                "distance_correlation_std": 0.01,
            }
            for comparison in ("same_setting", "cross_setting")
            for dimension, correlation in ((32, 0.7), (2048, 0.995))
        ]
    }
    factorial = {
        "retrieval": [
            {
                "normalization": "l2",
                "bandwidth_multiplier": multiplier,
                "comparison_type": comparison,
                "generations": 4,
                "method": method,
                "rff_dimension": 512 if method == "rfftrace" else None,
                "projection_dimension": None,
                "top_1_mean": 0.8,
                "top_1_std": 0.03,
            }
            for multiplier in (0.5, 1.0, 2.0)
            for comparison in ("same_setting", "cross_setting")
            for method in ("exact_mmd", "rfftrace")
        ]
    }
    decoding = {
        "cells": [
            {
                "method": method,
                "evaluation": evaluation,
                "temperature": temperature,
                "top_p": top_p,
                "top_1_mean": 0.8,
            }
            for method in ("exact_mmd", "rfftrace")
            for evaluation in ("same_setting", "cross_to_deterministic")
            for temperature in (0.3, 0.7, 1.0)
            for top_p in (0.8, 0.9, 0.95)
        ]
    }
    relationship = {
        "chance_baselines": {"roc_auc": 0.5, "nearest_group_accuracy": 0.32},
        "summary": [
            {
                "method": method,
                "evaluation": evaluation,
                "roc_auc_mean": 0.8,
                "roc_auc_std": 0.02,
                "nearest_group_accuracy_mean": 0.7,
                "nearest_group_accuracy_std": 0.03,
            }
            for method in methods
            for evaluation in ("same_setting", "cross_to_deterministic")
        ]
    }

    output = render_figures(
        tmp_path / "figures",
        feature_aggregate_path=_write(tmp_path / "feature.json", feature),
        projection_aggregate_path=_write(tmp_path / "projection.json", projection),
        factorial_aggregate_path=_write(tmp_path / "factorial.json", factorial),
        decoding_report_path=_write(tmp_path / "decoding.json", decoding),
        relationship_report_path=_write(tmp_path / "relationship.json", relationship),
    )

    manifest = json.loads((output / "figure-manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["sources"]) == 5
    assert len(manifest["artifacts"]) == 18
    assert all((output / row["filename"]).stat().st_size > 0 for row in manifest["artifacts"])
