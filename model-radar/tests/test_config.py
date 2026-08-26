from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model_radar.config import load_config


class ConfigTests(unittest.TestCase):
    def test_paths_resolve_relative_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "radar.yaml"
            path.write_text(
                "database: state/radar.db\nreports_dir: output\nprobes_file: probes.json\n",
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config["database"], str((root / "state/radar.db").resolve()))
            self.assertEqual(config["reports_dir"], str((root / "output").resolve()))
            self.assertEqual(config["probes_file"], str((root / "probes.json").resolve()))


if __name__ == "__main__":
    unittest.main()

