from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from model_radar.config import DEFAULT_CONFIG
from model_radar.db import RadarDB
from model_radar.scoring import score_model, trend_features


class DatabaseAndScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db = RadarDB(Path(self.temp.name) / "radar.db")
        self.db.initialize()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_first_seen_is_preserved_and_missing_models_become_inactive(self) -> None:
        models = [
            {"id": "stealth/a-alpha", "name": "A Alpha", "pricing": {"prompt": "0", "completion": "0"}},
            {"id": "vendor/reference", "name": "Reference", "pricing": {"prompt": "1", "completion": "1"}},
        ]
        self.db.upsert_models(models, "2026-08-20T00:00:00+00:00")
        models[0]["description"] = "updated"
        self.db.upsert_models([models[0]], "2026-08-21T00:00:00+00:00")
        alpha = self.db.get_model("stealth/a-alpha")
        reference = self.db.get_model("vendor/reference")
        self.assertEqual(alpha["first_seen_at"], "2026-08-20T00:00:00+00:00")
        self.assertEqual(alpha["last_seen_at"], "2026-08-21T00:00:00+00:00")
        self.assertEqual(reference["active"], 0)

    def test_trend_does_not_treat_absence_as_zero(self) -> None:
        rows = [
            {"date": "2026-08-20", "total_tokens": 100, "rank": 10},
            {"date": "2026-08-25", "total_tokens": 200, "rank": 4},
        ]
        features = trend_features(rows)
        self.assertEqual(features["growth_1d"], 1.0)
        self.assertIsNone(features["growth_7d"])
        self.assertEqual(features["rank_jump"], 6)

    def test_additive_score_prioritizes_stealth_free_preview(self) -> None:
        model = {
            "first_seen_at": "2026-08-25T00:00:00+00:00",
            "is_stealth": 1,
            "is_preview": 1,
            "is_free": 1,
            "metadata_json": json.dumps({"description": "coding reasoning agentic model"}),
        }
        score, components = score_model(
            model,
            {"growth_7d": 2.0, "growth_1d": None, "rank_jump": 10},
            DEFAULT_CONFIG,
            now=datetime(2026, 8, 26, tzinfo=UTC),
        )
        self.assertGreater(score, 80)
        self.assertEqual(components["stealth"], 1.0)


if __name__ == "__main__":
    unittest.main()

