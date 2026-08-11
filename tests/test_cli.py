from pathlib import Path

from distdna.cli import main


def test_demo_cli_creates_and_runs_complete_bundle(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demo"
    assert main(["demo", "--output-dir", str(demo_dir), "--run"]) == 0
    assert (demo_dir / "pilot.json").is_file()
    assert (demo_dir / "results" / "metrics.csv").is_file()
    assert main(["validate-config", str(demo_dir / "pilot.json")]) == 0
    assert (
        main(
            [
                "summarize",
                str(demo_dir / "results"),
                "--output",
                str(demo_dir / "results" / "summary.json"),
            ]
        )
        == 0
    )
    assert (demo_dir / "results" / "summary.json").is_file()
