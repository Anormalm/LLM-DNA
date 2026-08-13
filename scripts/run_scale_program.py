#!/usr/bin/env python3
"""Resume the complete expanded public-model experiment program."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from distdna.data import (
    CollectionManifest,
    PromptRevisionSet,
    apply_prompt_revisions,
    set_uniform_token_limit,
)


SEEDS = (2027, 2028, 2029)
GENERATION_COUNTS = (1, 2, 4, 8, 16)
RFF_DIMENSIONS = (16, 32, 64, 128, 256, 512, 1024)
PROJECTION_DIMENSIONS = (None, 1024, 2048)
BANDWIDTH_MULTIPLIERS = (0.5, 1.0, 2.0)
NORMALIZATIONS = ("l2", "none")
METHODS = (
    "single_sample_cosine",
    "mean_dna_cosine",
    "exact_mmd",
    "rfftrace",
)
PROTOCOL_PREFIX = "scale-expanded-v3"
PROMPT_PROTOCOL_DATASET_ID = "temporary-public-scale-v2"
SOURCE_DATASET_ID = "temporary-public-scale-v3"
SENTINEL_DATASET_ID = "temporary-public-scale-sentinel-v3"
MAX_NEW_TOKENS = 512
TRUNCATION_THRESHOLD = 0.25


def _protocol_manifest(root: Path) -> CollectionManifest:
    """Load v3 and prove the tracked prompt and token transforms exactly."""

    base = CollectionManifest.load(root / "configs" / "scale-expanded.collection.json")
    revisions = PromptRevisionSet.load(
        root / "configs" / "scale-expanded.prompt-revisions.json"
    )
    expected = apply_prompt_revisions(
        base,
        revisions,
        dataset_id=PROMPT_PROTOCOL_DATASET_ID,
        require_all_prompts=True,
    )
    tracked_v2 = CollectionManifest.load(
        root / "configs" / "scale-expanded-v2.collection.json"
    )
    if tracked_v2.fingerprint != expected.fingerprint:
        raise RuntimeError(
            "tracked v2 manifest is not the exact result of the v1 prompt revisions"
        )
    expected_v3 = set_uniform_token_limit(
        tracked_v2,
        max_new_tokens=MAX_NEW_TOKENS,
        dataset_id=SOURCE_DATASET_ID,
    )
    tracked_v3 = CollectionManifest.load(
        root / "configs" / "scale-expanded-v3.collection.json"
    )
    if tracked_v3.fingerprint != expected_v3.fingerprint:
        raise RuntimeError(
            "tracked v3 manifest is not the exact 512-token transform of v2"
        )
    return tracked_v3


def _write_json_exact(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a generated config once and reject any incompatible resume."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != payload:
            raise RuntimeError(f"existing generated config differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _run(command: list[str], *, env: Mapping[str, str], stage: str) -> None:
    print(f"\n== {stage} ==", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=dict(env))


def _base_experiment(
    root: Path,
    seed: int,
    output_dir: Path,
    *,
    rff_dimensions: tuple[int, ...],
    projection_dimensions: tuple[int | None, ...],
    normalization: str = "l2",
    bandwidth_multiplier: float = 1.0,
) -> dict[str, Any]:
    seed_data = root / "data" / f"{PROTOCOL_PREFIX}-seed{seed}"
    return {
        "data": {
            "calibration": str(seed_data / "embeddings" / "calibration.npz"),
            "evaluation": str(seed_data / "embeddings" / "evaluation.npz"),
        },
        "experiment": {
            "bandwidth": {
                "max_pairs": 100000,
                "multiplier": bandwidth_multiplier,
                "strategy": "median",
                "value": None,
            },
            "comparisons": [],
            "generation_counts": list(GENERATION_COUNTS),
            "methods": list(METHODS),
            "normalization": normalization,
            "projection_dimensions": list(projection_dimensions),
            "rff_dimensions": list(rff_dimensions),
            "seed": seed,
            "top_ks": [1, 3, 5],
        },
        "output": {"directory": str(output_dir), "save_distances": True},
    }


def _materialize(root: Path) -> dict[int, dict[str, Any]]:
    source = _protocol_manifest(root)
    runs: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        seed_data = root / "data" / f"{PROTOCOL_PREFIX}-seed{seed}"
        manifest_path = seed_data / "collection.json"
        if seed == source.random_seed:
            manifest = source
        else:
            metadata = dict(source.metadata)
            metadata["parent_manifest_fingerprint"] = source.fingerprint
            metadata["parent_random_seed"] = source.random_seed
            manifest = replace(source, random_seed=seed, metadata=metadata)
        if manifest_path.exists():
            existing = CollectionManifest.load(manifest_path)
            if existing.fingerprint != manifest.fingerprint:
                raise RuntimeError(f"existing seed manifest differs: {manifest_path}")
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest.save(manifest_path)

        config_dir = seed_data / "configs"
        grid_output = root / "results" / f"{PROTOCOL_PREFIX}-grid-seed{seed}"
        projection_output = root / "results" / f"{PROTOCOL_PREFIX}-projection-seed{seed}"
        grid_config = config_dir / "grid.json"
        projection_config = config_dir / "projection.json"
        _write_json_exact(
            grid_config,
            _base_experiment(
                root,
                seed,
                grid_output,
                rff_dimensions=RFF_DIMENSIONS,
                projection_dimensions=(None,),
            ),
        )
        _write_json_exact(
            projection_config,
            _base_experiment(
                root,
                seed,
                projection_output,
                rff_dimensions=(512,),
                projection_dimensions=PROJECTION_DIMENSIONS,
            ),
        )
        factorial_configs = []
        factorial_outputs = []
        for normalization in NORMALIZATIONS:
            for multiplier in BANDWIDTH_MULTIPLIERS:
                token = str(multiplier).replace(".", "p")
                output = (
                    root
                    / "results"
                    / f"{PROTOCOL_PREFIX}-factorial"
                    / f"seed{seed}-norm-{normalization}-bw-{token}"
                )
                config = config_dir / f"factorial-norm-{normalization}-bw-{token}.json"
                _write_json_exact(
                    config,
                    _base_experiment(
                        root,
                        seed,
                        output,
                        rff_dimensions=(512,),
                        projection_dimensions=(None,),
                        normalization=normalization,
                        bandwidth_multiplier=multiplier,
                    ),
                )
                factorial_configs.append(config)
                factorial_outputs.append(output)
        runs[seed] = {
            "manifest": manifest_path,
            "response_cache": seed_data / "responses",
            "embeddings": seed_data / "embeddings",
            "quality": root / "results" / f"{PROTOCOL_PREFIX}-quality-seed{seed}.json",
            "grid_config": grid_config,
            "grid_output": grid_output,
            "projection_config": projection_config,
            "projection_output": projection_output,
            "factorial_configs": factorial_configs,
            "factorial_outputs": factorial_outputs,
        }
    return runs


def _materialize_sentinel(root: Path) -> dict[str, Any]:
    source = _protocol_manifest(root)
    metadata = dict(source.metadata)
    metadata["parent_manifest_fingerprint"] = source.fingerprint
    metadata["parent_generations"] = source.generations
    metadata["purpose"] = (
        "all-roster, all-setting, all-prompt one-generation truncation and runtime sentinel"
    )
    sentinel = replace(
        source,
        dataset_id=SENTINEL_DATASET_ID,
        generations=1,
        metadata=metadata,
    )
    data_dir = root / "data" / f"{PROTOCOL_PREFIX}-sentinel"
    manifest_path = data_dir / "collection.json"
    if manifest_path.exists():
        existing = CollectionManifest.load(manifest_path)
        if existing.fingerprint != sentinel.fingerprint:
            raise RuntimeError(f"existing sentinel manifest differs: {manifest_path}")
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        sentinel.save(manifest_path)
    return {
        "manifest": manifest_path,
        "cache": data_dir / "responses",
        "report": root / "results" / f"{PROTOCOL_PREFIX}-sentinel-progress.json",
        "fingerprint": sentinel.fingerprint,
        "expected_records": (
            len(sentinel.model_ids)
            * len(sentinel.settings)
            * len(sentinel.prompts)
            * sentinel.generations
        ),
    }


def _require_sentinel_report_ready(
    report: Mapping[str, Any],
    *,
    fingerprint: str,
    expected_records: int,
    model_ids: tuple[str, ...],
) -> float:
    """Validate completeness plus overall and per-model truncation gates."""

    if report.get("manifest_fingerprint") != fingerprint:
        raise RuntimeError("scale sentinel report manifest differs")
    if report.get("records") != expected_records:
        raise RuntimeError("scale sentinel report is incomplete")
    if report.get("final_quality_gate_eligible") is not True:
        raise RuntimeError("scale sentinel report is not eligible for a final gate")
    truncation = report.get("overall_truncation_rate_observed")
    if not isinstance(truncation, (int, float)) or isinstance(truncation, bool):
        raise RuntimeError("scale sentinel report has no numeric truncation rate")
    if truncation > TRUNCATION_THRESHOLD:
        raise RuntimeError(
            "scale sentinel failed the overall truncation gate: "
            f"{truncation:.4f} > {TRUNCATION_THRESHOLD:.4f}"
        )
    model_rows = report.get("models")
    if not isinstance(model_rows, list):
        raise RuntimeError("scale sentinel report has no per-model rows")
    if any(not isinstance(row, dict) for row in model_rows):
        raise RuntimeError("scale sentinel report has an invalid per-model row")
    row_ids = [row.get("model_id") for row in model_rows]
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != set(model_ids):
        raise RuntimeError("scale sentinel report model roster differs")
    by_model = {row["model_id"]: row for row in model_rows}
    failures = []
    for model_id in model_ids:
        row = by_model[model_id]
        rate = row.get("truncation_rate_observed")
        if row.get("completion_fraction") != 1.0:
            failures.append(f"{model_id}=incomplete")
        elif not isinstance(rate, (int, float)) or isinstance(rate, bool):
            failures.append(f"{model_id}=missing")
        elif rate > TRUNCATION_THRESHOLD:
            failures.append(f"{model_id}={rate:.4f}")
    if failures:
        raise RuntimeError(
            "scale sentinel failed per-model truncation gate: " + ", ".join(failures)
        )
    return float(truncation)


def _sentinel_gate(
    root: Path, cli: Path, env: Mapping[str, str]
) -> dict[str, Any]:
    sentinel = _materialize_sentinel(root)
    reuse_sources = (
        (
            root / "data" / "scale-expanded-v4-token-preflight" / "responses",
            "reuse exact high-risk preflight records for scale sentinel",
        ),
        (
            root / "data" / f"{PROTOCOL_PREFIX}-seed2027" / "responses",
            "reuse generation-zero seed records for scale sentinel",
        ),
    )
    for source_cache, stage in reuse_sources:
        if (source_cache / "manifest.json").is_file():
            _run(
                [
                    str(cli),
                    "reuse-responses",
                    "--source-cache",
                    str(source_cache),
                    "--target-manifest",
                    str(sentinel["manifest"]),
                    "--target-cache",
                    str(sentinel["cache"]),
                ],
                env=env,
                stage=stage,
            )
    response_file = sentinel["cache"] / "responses.jsonl"
    completed = 0
    if response_file.is_file():
        with response_file.open("r", encoding="utf-8") as stream:
            completed = sum(bool(line.strip()) for line in stream)
    if completed < sentinel["expected_records"]:
        _run(
            [
                str(cli),
                "collect-local",
                "--manifest",
                str(sentinel["manifest"]),
                "--cache-dir",
                str(sentinel["cache"]),
                "--device",
                "mps",
                "--dtype",
                "float16",
                "--local-files-only",
                "--progress-every",
                "100",
                "--abort-model-truncation-threshold",
                str(TRUNCATION_THRESHOLD),
            ],
            env=env,
            stage=f"all-cell scale sentinel ({completed}/{sentinel['expected_records']})",
        )
    _run(
        [
            str(cli),
            "collection-progress",
            "--manifest",
            str(sentinel["manifest"]),
            "--cache-dir",
            str(sentinel["cache"]),
            "--output",
            str(sentinel["report"]),
        ],
        env=env,
        stage="scale sentinel report",
    )
    with sentinel["report"].open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    truncation = _require_sentinel_report_ready(
        report,
        fingerprint=sentinel["fingerprint"],
        expected_records=sentinel["expected_records"],
        model_ids=_protocol_manifest(root).model_ids,
    )
    print(
        "Scale sentinel passed overall and every-model truncation gates: "
        f"overall={truncation:.4f} <= 0.2500",
        flush=True,
    )
    return sentinel


def _quality_ready(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("manifest_fingerprint") != fingerprint:
        raise RuntimeError(f"quality report manifest differs: {path}")
    checks = report.get("checks")
    if not isinstance(checks, dict) or checks.get(
        "every_model_truncation_rate_at_most_25_percent"
    ) is not True:
        raise RuntimeError(f"quality report lacks a passing per-model gate: {path}")
    if report.get("ready_for_scale") is not True:
        raise RuntimeError(f"seed failed response-quality gate: {path}")
    return True


def _record_runtime(root: Path) -> None:
    target = root / "data" / f"{PROTOCOL_PREFIX}-runtime.json"
    if target.exists():
        return
    import torch
    import transformers

    _write_json_exact(
        target,
        {
            "device": "mps",
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="materialize and verify all generated manifests/configs without running work",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")
    cli = Path(sys.executable).with_name("distdna")
    if not cli.is_file():
        raise FileNotFoundError(f"distdna executable is unavailable: {cli}")
    env = dict(os.environ)
    env.update(
        {
            "HF_HOME": str(root / "data" / "hf-cache"),
            "HF_HUB_OFFLINE": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        }
    )
    _record_runtime(root)
    runs = _materialize(root)
    _materialize_sentinel(root)
    if args.prepare_only:
        print(
            "Expanded v3 sentinel, three-seed manifests, and analysis configs are ready."
        )
        return 0

    sentinel = _sentinel_gate(root, cli, env)

    for seed in SEEDS:
        run = runs[seed]
        manifest = CollectionManifest.load(run["manifest"])
        if seed == _protocol_manifest(root).random_seed:
            _run(
                [
                    str(cli),
                    "reuse-responses",
                    "--source-cache",
                    str(sentinel["cache"]),
                    "--target-manifest",
                    str(run["manifest"]),
                    "--target-cache",
                    str(run["response_cache"]),
                ],
                env=env,
                stage="reuse sentinel generation-zero records for seed 2027",
            )
        response_file = run["response_cache"] / "responses.jsonl"
        completed = 0
        if response_file.is_file():
            with response_file.open("r", encoding="utf-8") as stream:
                completed = sum(bool(line.strip()) for line in stream)
        expected_records = (
            len(manifest.model_ids)
            * len(manifest.settings)
            * len(manifest.prompts)
            * manifest.generations
        )
        if completed < expected_records:
            _run(
                [
                    str(cli),
                    "collect-local",
                    "--manifest",
                    str(run["manifest"]),
                    "--cache-dir",
                    str(run["response_cache"]),
                    "--device",
                    "mps",
                    "--dtype",
                    "float16",
                    "--local-files-only",
                    "--progress-every",
                    str(args.progress_every),
                    "--abort-model-truncation-threshold",
                    str(TRUNCATION_THRESHOLD),
                ],
                env=env,
                stage=f"collect seed {seed} ({completed}/{expected_records})",
            )
        if not _quality_ready(run["quality"], manifest.fingerprint):
            _run(
                [
                    str(cli),
                    "response-quality",
                    "--manifest",
                    str(run["manifest"]),
                    "--cache-dir",
                    str(run["response_cache"]),
                    "--output",
                    str(run["quality"]),
                ],
                env=env,
                stage=f"quality gate seed {seed}",
            )
            _quality_ready(run["quality"], manifest.fingerprint)
        if not run["embeddings"].is_dir():
            _run(
                [
                    str(cli),
                    "encode-responses",
                    "--manifest",
                    str(run["manifest"]),
                    "--cache-dir",
                    str(run["response_cache"]),
                    "--output-dir",
                    str(run["embeddings"]),
                    "--encoder",
                    "sentence-transformer",
                    "--encoder-model",
                    "sentence-transformers/all-mpnet-base-v2",
                    "--device",
                    "mps",
                    "--batch-size",
                    "64",
                ],
                env=env,
                stage=f"encode seed {seed}",
            )
        for label, config, output in (
            ("grid", run["grid_config"], run["grid_output"]),
            ("projection", run["projection_config"], run["projection_output"]),
        ):
            if not output.is_dir():
                _run(
                    [str(cli), "run", str(config)],
                    env=env,
                    stage=f"{label} analysis seed {seed}",
                )
        for index, (config, output) in enumerate(
            zip(run["factorial_configs"], run["factorial_outputs"]), start=1
        ):
            if not output.is_dir():
                _run(
                    [str(cli), "run", str(config)],
                    env=env,
                    stage=f"factorial analysis seed {seed} ({index}/6)",
                )

    grid_outputs = [str(runs[seed]["grid_output"]) for seed in SEEDS]
    projection_outputs = [str(runs[seed]["projection_output"]) for seed in SEEDS]
    factorial_outputs = [
        str(output) for seed in SEEDS for output in runs[seed]["factorial_outputs"]
    ]
    grid_aggregate = root / "results" / f"{PROTOCOL_PREFIX}-grid-aggregate.json"
    projection_aggregate = root / "results" / f"{PROTOCOL_PREFIX}-projection-aggregate.json"
    factorial_aggregate = root / "results" / f"{PROTOCOL_PREFIX}-factorial-aggregate.json"
    aggregates = (
        (grid_outputs, grid_aggregate, "grid aggregate"),
        (projection_outputs, projection_aggregate, "projection aggregate"),
        (factorial_outputs, factorial_aggregate, "factorial aggregate"),
    )
    for outputs, target, label in aggregates:
        if not target.is_file():
            _run(
                [str(cli), "aggregate", *outputs, "--output", str(target)],
                env=env,
                stage=label,
            )

    manifest = runs[2027]["manifest"]
    decoding = root / "results" / f"{PROTOCOL_PREFIX}-decoding-report.json"
    if not decoding.is_file():
        _run(
            [
                str(cli),
                "decoding-report",
                "--aggregate",
                str(grid_aggregate),
                "--manifest",
                str(manifest),
                "--generations",
                "16",
                "--rff-dim",
                "512",
                "--output",
                str(decoding),
            ],
            env=env,
            stage="decoding report",
        )
    relationship = root / "results" / f"{PROTOCOL_PREFIX}-relationship-report.json"
    if not relationship.is_file():
        _run(
            [
                str(cli),
                "relationship-report",
                *grid_outputs,
                "--manifest",
                str(manifest),
                "--relationship-map",
                str(root / "configs" / "scale-expanded.relationships.json"),
                "--generations",
                "16",
                "--rff-dim",
                "512",
                "--output",
                str(relationship),
            ],
            env=env,
            stage="relationship report",
        )
    figures = root / "results" / f"{PROTOCOL_PREFIX}-final-figures"
    if not figures.is_dir():
        _run(
            [
                str(cli),
                "render-figures",
                "--retrieval-aggregate",
                str(grid_aggregate),
                "--feature-aggregate",
                str(grid_aggregate),
                "--projection-aggregate",
                str(projection_aggregate),
                "--factorial-aggregate",
                str(factorial_aggregate),
                "--decoding-report",
                str(decoding),
                "--relationship-report",
                str(relationship),
                "--output-dir",
                str(figures),
            ],
            env=env,
            stage="final figures",
        )
    print("\nExpanded v3 three-seed program complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
