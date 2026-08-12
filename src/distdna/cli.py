"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .ablation import write_ablation_suite
from .api import RFFTraceExtractionConfig, calc_rfftrace
from .config import ExperimentConfig
from .data import (
    CollectionManifest,
    EmbeddingDataset,
    HashingResponseEncoder,
    audit_llm_dna_responses,
    import_llm_dna_responses,
    load_model_aliases,
    ResponseCache,
    SentenceTransformerResponseEncoder,
    build_embedding_datasets,
    collect_responses,
    reuse_compatible_responses,
    save_embedding_datasets,
)
from .demo import create_demo
from .decoding import build_decoding_report
from .experiment import run_experiment, validate_experiment
from .figures import render_figures
from .providers import (
    LocalTransformersGenerator,
    inherit_model_revisions,
    resolve_model_revisions,
)
from .relationships import build_relationship_report
from .summary import aggregate_runs, summarize_run, write_summary
from .text_demo import create_text_pipeline_demo


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="distdna")
    parser.add_argument("--version", action="version", version="distdna 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    validate_data = commands.add_parser("validate-data", help="validate an embedding NPZ")
    validate_data.add_argument("path", type=Path)

    validate_config = commands.add_parser(
        "validate-config", help="validate a pilot config and both datasets"
    )
    validate_config.add_argument("path", type=Path)

    run = commands.add_parser("run", help="run a pilot experiment")
    run.add_argument("path", type=Path)

    demo = commands.add_parser("demo", help="create deterministic synthetic pilot data")
    demo.add_argument("--output-dir", type=Path, default=Path("demo"))
    demo.add_argument("--seed", type=int, default=2027)
    demo.add_argument(
        "--profile",
        choices=("smoke", "distributional"),
        default="smoke",
        help="smoke checks plumbing; distributional tests equal-mean distribution shapes",
    )
    demo.add_argument("--run", action="store_true", help="run the generated pilot immediately")

    text_demo = commands.add_parser(
        "pipeline-demo", help="run collection through embedding on controlled text responses"
    )
    text_demo.add_argument("--output-dir", type=Path, default=Path("text-demo"))
    text_demo.add_argument("--seed", type=int, default=2027)
    text_demo.add_argument(
        "--run", action="store_true", help="run the generated retrieval pilot immediately"
    )

    encode = commands.add_parser(
        "encode-responses", help="encode a complete response cache into embedding datasets"
    )
    encode.add_argument("--manifest", type=Path, required=True)
    encode.add_argument("--cache-dir", type=Path, required=True)
    encode.add_argument("--output-dir", type=Path, required=True)
    encode.add_argument(
        "--encoder", choices=("hashing", "sentence-transformer"), default="hashing"
    )
    encode.add_argument("--embedding-dim", type=int, default=128)
    encode.add_argument(
        "--encoder-model", default="sentence-transformers/all-mpnet-base-v2"
    )
    encode.add_argument("--device", default="cpu")
    encode.add_argument("--batch-size", type=int, default=32)

    resolve_models = commands.add_parser(
        "resolve-models", help="pin Hugging Face model IDs to immutable commit SHAs"
    )
    resolve_models.add_argument("--manifest", type=Path, required=True)
    resolve_models.add_argument("--output", type=Path, required=True)
    resolve_models.add_argument(
        "--revisions-from",
        type=Path,
        help="inherit already pinned commits from a compatible resolved manifest",
    )

    reseed_manifest = commands.add_parser(
        "reseed-manifest",
        help="clone a manifest with a new collection seed and provenance link",
    )
    reseed_manifest.add_argument("--manifest", type=Path, required=True)
    reseed_manifest.add_argument("--seed", type=int, required=True)
    reseed_manifest.add_argument("--output", type=Path, required=True)

    collect_local = commands.add_parser(
        "collect-local", help="collect a manifest with sequential local Transformers models"
    )
    collect_local.add_argument("--manifest", type=Path, required=True)
    collect_local.add_argument("--cache-dir", type=Path, required=True)
    collect_local.add_argument("--device", default="auto")
    collect_local.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    collect_local.add_argument("--local-files-only", action="store_true")

    reuse = commands.add_parser(
        "reuse-responses",
        help="reuse exact compatible records when expanding a collection manifest",
    )
    reuse.add_argument("--source-cache", type=Path, required=True)
    reuse.add_argument("--target-manifest", type=Path, required=True)
    reuse.add_argument("--target-cache", type=Path, required=True)

    audit_legacy = commands.add_parser(
        "audit-legacy",
        help="audit legacy LLM-DNA response folders against a pinned manifest",
    )
    _add_legacy_arguments(audit_legacy)
    audit_legacy.add_argument("--output", type=Path, help="write the JSON audit report")

    import_legacy = commands.add_parser(
        "import-legacy",
        help="atomically import a complete, provenance-safe LLM-DNA collection",
    )
    _add_legacy_arguments(import_legacy)
    import_legacy.add_argument("--cache-dir", type=Path, required=True)
    import_legacy.add_argument(
        "--audit-output", type=Path, help="write the successful JSON audit report"
    )

    summarize = commands.add_parser("summarize", help="summarize a completed pilot")
    summarize.add_argument("output_dir", type=Path)
    summarize.add_argument("--output", type=Path, help="optionally write the JSON summary")

    aggregate = commands.add_parser("aggregate", help="aggregate completed pilots across seeds")
    aggregate.add_argument("output_dirs", nargs="+", type=Path)
    aggregate.add_argument("--output", type=Path, help="optionally write the JSON aggregate")

    suite = commands.add_parser(
        "make-ablation-suite",
        help="materialize a normalization and bandwidth factorial config suite",
    )
    suite.add_argument("--base", type=Path, required=True)
    suite.add_argument("--config-dir", type=Path, required=True)
    suite.add_argument("--result-root", type=Path, required=True)
    suite.add_argument(
        "--normalizations", nargs="+", choices=("none", "l2"), required=True
    )
    suite.add_argument(
        "--bandwidth-multipliers", nargs="+", type=float, required=True
    )

    decoding = commands.add_parser(
        "decoding-report",
        help="analyze same-setting and stochastic-to-deterministic decoding grids",
    )
    decoding.add_argument("--aggregate", type=Path, required=True)
    decoding.add_argument("--manifest", type=Path, required=True)
    decoding.add_argument("--output", type=Path, required=True)
    decoding.add_argument("--normalization", choices=("none", "l2"), default="l2")
    decoding.add_argument("--bandwidth-multiplier", type=float, default=1.0)
    decoding.add_argument("--generations", type=int, default=4)
    decoding.add_argument("--rff-dim", type=int, default=512)
    decoding.add_argument("--projection-dim", type=int)

    relationships = commands.add_parser(
        "relationship-report",
        help="evaluate family-label relationship recovery from saved distances",
    )
    relationships.add_argument("output_dirs", nargs="+", type=Path)
    relationships.add_argument("--manifest", type=Path, required=True)
    relationships.add_argument("--relationship-map", type=Path, required=True)
    relationships.add_argument("--output", type=Path, required=True)
    relationships.add_argument("--normalization", choices=("none", "l2"), default="l2")
    relationships.add_argument("--bandwidth-multiplier", type=float, default=1.0)
    relationships.add_argument("--generations", type=int, default=4)
    relationships.add_argument("--rff-dim", type=int, default=512)
    relationships.add_argument("--projection-dim", type=int)

    figures = commands.add_parser(
        "render-figures", help="render final paper figures from completed JSON reports"
    )
    figures.add_argument("--retrieval-aggregate", type=Path, required=True)
    figures.add_argument("--feature-aggregate", type=Path, required=True)
    figures.add_argument("--projection-aggregate", type=Path, required=True)
    figures.add_argument("--factorial-aggregate", type=Path, required=True)
    figures.add_argument("--decoding-report", type=Path, required=True)
    figures.add_argument("--relationship-report", type=Path, required=True)
    figures.add_argument("--output-dir", type=Path, required=True)
    figures.add_argument(
        "--formats", nargs="+", choices=("svg", "pdf", "png"), default=("svg", "pdf", "png")
    )

    extract = commands.add_parser("extract", help="extract shared RFFTrace DNA vectors")
    _add_extraction_arguments(extract)
    return parser


def _add_legacy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model-aliases",
        type=Path,
        help="JSON mapping from legacy directory model tokens to manifest model IDs",
    )
    parser.add_argument(
        "--repeat-base",
        type=int,
        choices=(0, 1),
        default=1,
        help="legacy rN numbering base; the student branch uses one-based repeats",
    )


def _add_extraction_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--settings", nargs="+", help="defaults to every setting in the dataset")
    parser.add_argument("--generations", type=int, default=16)
    parser.add_argument("--rff-dim", type=int, default=1024)
    parser.add_argument("--dna-dim", type=int, default=128)
    parser.add_argument("--no-projection", action="store_true")
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--normalization", choices=("none", "l2"), default="l2")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--bandwidth-max-pairs", type=int, default=100_000)
    parser.add_argument("--output-dir", type=Path, default=Path("out/rfftrace"))
    parser.add_argument("--no-save", action="store_true")


def _extraction_config(args: argparse.Namespace) -> RFFTraceExtractionConfig:
    return RFFTraceExtractionConfig(
        evaluation_path=args.evaluation,
        calibration_path=args.calibration,
        settings=None if args.settings is None else tuple(args.settings),
        generations=args.generations,
        rff_dimension=args.rff_dim,
        dna_dimension=None if args.no_projection else args.dna_dim,
        sigma=args.sigma,
        normalization=args.normalization,
        random_seed=args.random_seed,
        bandwidth_max_pairs=args.bandwidth_max_pairs,
        output_dir=args.output_dir,
        save=not args.no_save,
    )


def _run_extraction(args: argparse.Namespace) -> dict:
    result = calc_rfftrace(_extraction_config(args))
    return {
        "status": "complete",
        "vector_shape": list(result.vectors.shape),
        "sigma": result.sigma,
        "elapsed_seconds": result.elapsed_seconds,
        "output_dir": None if result.output_dir is None else str(result.output_dir),
        "collection_path": (
            None if result.collection_path is None else str(result.collection_path)
        ),
        "summary_path": None if result.summary_path is None else str(result.summary_path),
        "parameters_path": (
            None if result.parameters_path is None else str(result.parameters_path)
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-data":
            dataset = EmbeddingDataset.load(args.path)
            print(
                json.dumps(
                    {
                        "path": str(args.path.resolve()),
                        "shape": list(dataset.shape),
                        "models": list(dataset.model_ids),
                        "settings": list(dataset.setting_ids),
                        "prompts": len(dataset.prompt_ids),
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "validate-config":
            config = ExperimentConfig.load(args.path)
            evaluation, calibration = validate_experiment(config)
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "evaluation_shape": list(evaluation.shape),
                        "calibration_shape": list(calibration.shape),
                        "output_dir": str(config.output_dir),
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "run":
            result = run_experiment(ExperimentConfig.load(args.path))
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "output_dir": str(result.output_dir),
                        "sigma": result.sigma,
                        "metric_rows": result.metric_rows,
                        "rank_rows": result.rank_rows,
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "demo":
            config_path = create_demo(args.output_dir, args.seed, args.profile)
            response = {"status": "created", "config": str(config_path)}
            if args.run:
                result = run_experiment(ExperimentConfig.load(config_path))
                response.update(
                    {
                        "status": "complete",
                        "output_dir": str(result.output_dir),
                        "sigma": result.sigma,
                        "metric_rows": result.metric_rows,
                    }
                )
            print(json.dumps(response, indent=2))
            return 0
        if args.command == "pipeline-demo":
            config_path = create_text_pipeline_demo(args.output_dir, args.seed)
            response = {"status": "created", "config": str(config_path)}
            if args.run:
                result = run_experiment(ExperimentConfig.load(config_path))
                summary = summarize_run(result.output_dir)
                summary_path = write_summary(summary, result.output_dir / "summary.json")
                response.update(
                    {
                        "status": "complete",
                        "output_dir": str(result.output_dir),
                        "summary": str(summary_path),
                        "sigma": result.sigma,
                        "metric_rows": result.metric_rows,
                    }
                )
            print(json.dumps(response, indent=2))
            return 0
        if args.command == "encode-responses":
            manifest = CollectionManifest.load(args.manifest)
            if not args.cache_dir.is_dir():
                raise FileNotFoundError(
                    f"response cache directory not found: {args.cache_dir}"
                )
            responses = ResponseCache(args.cache_dir, manifest).dataset
            responses.require_complete()
            if args.encoder == "hashing":
                encoder = HashingResponseEncoder(args.embedding_dim)
            else:
                encoder = SentenceTransformerResponseEncoder(
                    args.encoder_model, args.device, args.batch_size
                )
            datasets = build_embedding_datasets(manifest, responses, encoder)
            output_dir = save_embedding_datasets(
                datasets, manifest, encoder, args.output_dir
            )
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "output_dir": str(output_dir),
                        "encoder_id": encoder.encoder_id,
                        "calibration_shape": list(datasets["calibration"].shape),
                        "evaluation_shape": list(datasets["evaluation"].shape),
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "resolve-models":
            if args.output.exists():
                raise FileExistsError(
                    f"resolved manifest already exists; choose a fresh path: {args.output}"
                )
            manifest = CollectionManifest.load(args.manifest)
            resolved = (
                resolve_model_revisions(manifest)
                if args.revisions_from is None
                else inherit_model_revisions(
                    manifest, CollectionManifest.load(args.revisions_from)
                )
            )
            written = resolved.save(args.output.resolve())
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "output": str(written),
                        "manifest_fingerprint": resolved.fingerprint,
                        "model_revisions": resolved.metadata["model_revisions"],
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "reseed-manifest":
            if args.output.exists():
                raise FileExistsError(
                    f"reseeded manifest already exists; choose a fresh path: {args.output}"
                )
            manifest = CollectionManifest.load(args.manifest)
            if args.seed == manifest.random_seed:
                raise ValueError("new collection seed must differ from the source manifest")
            metadata = dict(manifest.metadata)
            metadata["parent_manifest_fingerprint"] = manifest.fingerprint
            metadata["parent_random_seed"] = manifest.random_seed
            reseeded = replace(
                manifest,
                random_seed=args.seed,
                metadata=metadata,
            )
            written = reseeded.save(args.output.resolve())
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "output": str(written),
                        "random_seed": reseeded.random_seed,
                        "manifest_fingerprint": reseeded.fingerprint,
                        "parent_manifest_fingerprint": manifest.fingerprint,
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "collect-local":
            manifest = CollectionManifest.load(args.manifest)
            generator = LocalTransformersGenerator(
                manifest,
                device=args.device,
                dtype=args.dtype,
                local_files_only=args.local_files_only,
                status=lambda message: print(message, file=sys.stderr, flush=True),
            )

            def report(completed, total, record):
                if completed == total or completed % 10 == 0:
                    print(
                        f"collected {completed}/{total}: "
                        f"{record.model_id} {record.setting_id} {record.prompt_id} "
                        f"generation={record.generation_index}",
                        file=sys.stderr,
                        flush=True,
                    )

            try:
                responses = collect_responses(
                    manifest, generator, args.cache_dir, progress=report
                )
            finally:
                generator.close()
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "cache_dir": str(args.cache_dir.resolve()),
                        "records": len(responses.records),
                        "expected_records": responses.expected_count,
                        "manifest_fingerprint": manifest.fingerprint,
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "reuse-responses":
            manifest = CollectionManifest.load(args.target_manifest)
            _, report = reuse_compatible_responses(
                args.source_cache, manifest, args.target_cache
            )
            print(json.dumps({"status": "complete", **report}, indent=2))
            return 0
        if args.command == "audit-legacy":
            manifest = CollectionManifest.load(args.manifest)
            report = audit_llm_dna_responses(
                args.source_dir,
                manifest,
                model_aliases=load_model_aliases(args.model_aliases),
                repeat_base=args.repeat_base,
            )
            payload = report.as_dict()
            if args.output is not None:
                payload["written_to"] = str(report.write(args.output))
            print(json.dumps(payload, indent=2))
            return 0 if report.ready_for_import else 2
        if args.command == "import-legacy":
            manifest = CollectionManifest.load(args.manifest)
            responses, report = import_llm_dna_responses(
                args.source_dir,
                manifest,
                args.cache_dir,
                model_aliases=load_model_aliases(args.model_aliases),
                repeat_base=args.repeat_base,
            )
            audit_path = None
            if args.audit_output is not None:
                audit_path = report.write(args.audit_output)
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "cache_dir": str(args.cache_dir.resolve()),
                        "records": len(responses.records),
                        "expected_records": responses.expected_count,
                        "manifest_fingerprint": manifest.fingerprint,
                        "audit_output": None if audit_path is None else str(audit_path),
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "summarize":
            summary = summarize_run(args.output_dir)
            if args.output is not None:
                written = write_summary(summary, args.output)
                summary = dict(summary)
                summary["written_to"] = str(written)
            print(json.dumps(summary, indent=2))
            return 0
        if args.command == "aggregate":
            summary = aggregate_runs(args.output_dirs)
            if args.output is not None:
                written = write_summary(summary, args.output)
                summary = dict(summary)
                summary["written_to"] = str(written)
            print(json.dumps(summary, indent=2))
            return 0
        if args.command == "make-ablation-suite":
            suite_path = write_ablation_suite(
                base_config_path=args.base,
                config_dir=args.config_dir,
                result_root=args.result_root,
                normalizations=args.normalizations,
                bandwidth_multipliers=args.bandwidth_multipliers,
            )
            print(
                json.dumps(
                    {"status": "created", "suite": str(suite_path)}, indent=2
                )
            )
            return 0
        if args.command == "decoding-report":
            if args.output.exists():
                raise FileExistsError(
                    f"decoding report already exists; choose a fresh path: {args.output}"
                )
            report = build_decoding_report(
                args.aggregate,
                args.manifest,
                normalization=args.normalization,
                bandwidth_multiplier=args.bandwidth_multiplier,
                generations=args.generations,
                rff_dimension=args.rff_dim,
                projection_dimension=args.projection_dim,
            )
            written = write_summary(report, args.output)
            print(
                json.dumps(
                    {"status": "complete", "output": str(written)}, indent=2
                )
            )
            return 0
        if args.command == "relationship-report":
            if args.output.exists():
                raise FileExistsError(
                    f"relationship report already exists; choose a fresh path: {args.output}"
                )
            report = build_relationship_report(
                args.output_dirs,
                args.manifest,
                args.relationship_map,
                normalization=args.normalization,
                bandwidth_multiplier=args.bandwidth_multiplier,
                generations=args.generations,
                rff_dimension=args.rff_dim,
                projection_dimension=args.projection_dim,
            )
            written = write_summary(report, args.output)
            print(json.dumps({"status": "complete", "output": str(written)}, indent=2))
            return 0
        if args.command == "render-figures":
            output_dir = render_figures(
                args.output_dir,
                retrieval_aggregate_path=args.retrieval_aggregate,
                feature_aggregate_path=args.feature_aggregate,
                projection_aggregate_path=args.projection_aggregate,
                factorial_aggregate_path=args.factorial_aggregate,
                decoding_report_path=args.decoding_report,
                relationship_report_path=args.relationship_report,
                formats=args.formats,
            )
            print(json.dumps({"status": "complete", "output_dir": str(output_dir)}, indent=2))
            return 0
        if args.command == "extract":
            print(json.dumps(_run_extraction(args), indent=2))
            return 0
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted; completed responses remain safely cached", file=sys.stderr)
        return 130
    return 1


def main_calc_rfftrace(argv: Sequence[str] | None = None) -> int:
    """Flat CLI used by the dedicated ``calc-rfftrace`` console script."""

    parser = argparse.ArgumentParser(prog="calc-rfftrace")
    _add_extraction_arguments(parser)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(_run_extraction(args), indent=2))
        return 0
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
