from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from model_radar.config import DEFAULT_CONFIG
from model_radar.db import RadarDB
from model_radar.openrouter import OpenRouterClient
from model_radar.pipeline import discover, ingest_rankings, probe_model


ROOT = Path(__file__).parent
FIXTURES = ROOT / "fixtures"
PROBES = ROOT.parent / "probes" / "default.json"


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config["database"] = str(Path(self.temp.name) / "radar.db")
        self.config["reports_dir"] = str(Path(self.temp.name) / "reports")
        self.config["probes_file"] = str(PROBES)
        self.config["fingerprint"]["dimension"] = 128
        self.config["drift"]["distance_threshold"] = 0.08
        self.config["alerts"]["webhook_url"] = ""
        self.db = RadarDB(self.config["database"])
        self.db.initialize()
        self.client = OpenRouterClient()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_offline_end_to_end_and_change_detection(self) -> None:
        discovery = discover(
            self.db, self.client, self.config, fixture=FIXTURES / "models.json"
        )
        self.assertEqual(discovery["seen"], 3)
        rankings = ingest_rankings(
            self.db, self.client, self.config, fixture=FIXTURES / "rankings.json"
        )
        self.assertEqual(rankings["written"], 12)

        first = probe_model(
            self.db,
            self.client,
            self.config,
            "stealth/ox-alpha",
            fixture_responses=FIXTURES / "responses.json",
            overrides={"repetitions": 2},
        )
        second = probe_model(
            self.db,
            self.client,
            self.config,
            "stealth/ox-alpha",
            fixture_responses=FIXTURES / "responses.json",
            overrides={"repetitions": 2},
        )
        self.assertIsNone(first["drift"])
        self.assertFalse(second["drift"]["alerted"])

        changed_path = Path(self.temp.name) / "changed.json"
        changed_path.write_text(
            json.dumps(
                {
                    "models": {
                        "stealth/ox-alpha": [
                            [f"unrelated replacement behavior number {index} xyz" for index in range(20)]
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        changed = probe_model(
            self.db,
            self.client,
            self.config,
            "stealth/ox-alpha",
            fixture_responses=changed_path,
            overrides={"repetitions": 2},
        )
        self.assertTrue(changed["drift"]["alerted"])
        self.assertGreater(changed["drift"]["distance"], 0.08)


if __name__ == "__main__":
    unittest.main()

