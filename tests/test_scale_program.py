import json
from pathlib import Path

import pytest

from scripts.run_scale_program import (
    MAX_NEW_TOKENS,
    _quality_ready,
    _protocol_manifest,
    _require_sentinel_report_ready,
)


def test_tracked_v3_protocol_is_exactly_derivable() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = _protocol_manifest(root)

    assert manifest.dataset_id == "temporary-public-scale-v3"
    assert {setting.max_new_tokens for setting in manifest.settings} == {
        MAX_NEW_TOKENS
    }
    assert manifest.metadata["parent_max_new_tokens"] == [128]


def _sentinel_report(*, overall: float = 0.25, m0: float = 0.25) -> dict:
    return {
        "manifest_fingerprint": "fingerprint",
        "records": 2,
        "final_quality_gate_eligible": True,
        "overall_truncation_rate_observed": overall,
        "models": [
            {
                "model_id": "m0",
                "completion_fraction": 1.0,
                "truncation_rate_observed": m0,
            },
            {
                "model_id": "m1",
                "completion_fraction": 1.0,
                "truncation_rate_observed": 0.0,
            },
        ],
    }


def test_sentinel_gate_accepts_threshold_and_rejects_hidden_model_failure() -> None:
    assert (
        _require_sentinel_report_ready(
            _sentinel_report(),
            fingerprint="fingerprint",
            expected_records=2,
            model_ids=("m0", "m1"),
        )
        == 0.25
    )
    with pytest.raises(RuntimeError, match="per-model truncation"):
        _require_sentinel_report_ready(
            _sentinel_report(overall=0.2, m0=0.4),
            fingerprint="fingerprint",
            expected_records=2,
            model_ids=("m0", "m1"),
        )


def test_sentinel_gate_rejects_incomplete_or_duplicate_model_rows() -> None:
    incomplete = _sentinel_report()
    incomplete["records"] = 1
    with pytest.raises(RuntimeError, match="incomplete"):
        _require_sentinel_report_ready(
            incomplete,
            fingerprint="fingerprint",
            expected_records=2,
            model_ids=("m0", "m1"),
        )

    duplicate = _sentinel_report()
    duplicate["models"][1]["model_id"] = "m0"
    with pytest.raises(RuntimeError, match="roster differs"):
        _require_sentinel_report_ready(
            duplicate,
            fingerprint="fingerprint",
            expected_records=2,
            model_ids=("m0", "m1"),
        )


def test_quality_ready_requires_new_per_model_check(tmp_path: Path) -> None:
    path = tmp_path / "quality.json"
    path.write_text(
        json.dumps(
            {
                "manifest_fingerprint": "fingerprint",
                "checks": {
                    "every_model_truncation_rate_at_most_25_percent": True
                },
                "ready_for_scale": True,
            }
        ),
        encoding="utf-8",
    )
    assert _quality_ready(path, "fingerprint") is True

    path.write_text(
        json.dumps(
            {
                "manifest_fingerprint": "fingerprint",
                "checks": {},
                "ready_for_scale": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="per-model gate"):
        _quality_ready(path, "fingerprint")
