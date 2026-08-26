from __future__ import annotations

import argparse
import importlib.resources
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import load_config, write_default_config
from .db import RadarDB
from .openrouter import OpenRouterClient
from .pipeline import discover, ingest_rankings, priority_queue, probe_model, scan_queue
from .reports import render_index, render_model_report, write_reports
from .server import serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-radar",
        description="Discover, fingerprint, and monitor emerging black-box LLMs.",
    )
    parser.add_argument("--config", help="YAML config path (default: model-radar.yaml)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create a starter config and initialize SQLite")
    init.add_argument("--path", default="model-radar.yaml", help="Config file to create")

    discovery = subparsers.add_parser("discover", help="Snapshot the OpenRouter model catalog")
    discovery.add_argument("--fixture", help="Use a local Models API JSON fixture")

    trends = subparsers.add_parser("trends", help="Ingest OpenRouter daily rankings")
    trends.add_argument("--fixture", help="Use a local rankings JSON fixture")
    trends.add_argument("--start-date")
    trends.add_argument("--end-date")
    trends.add_argument("--period", choices=["day", "week", "month"], default="day")
    trends.add_argument("--category", default="")
    trends.add_argument("--language-type", choices=["", "natural", "programming"], default="")
    trends.add_argument("--modality", default="")
    trends.add_argument("--context-bucket", default="")

    queue = subparsers.add_parser("queue", help="Show high-priority models")
    queue.add_argument("--min-score", type=float)
    queue.add_argument("--limit", type=int, default=20)

    probe = subparsers.add_parser("probe", help="Probe one model and compute a fingerprint")
    probe.add_argument("model_id")
    _probe_arguments(probe)

    scan = subparsers.add_parser("scan", help="Probe an explicit list or the priority queue")
    scan.add_argument("--models", help="Comma-separated model IDs")
    scan.add_argument("--min-score", type=float)
    scan.add_argument("--limit", type=int, default=3)
    scan.add_argument(
        "--yes-spend",
        action="store_true",
        help="Confirm API calls may consume OpenRouter credits (not needed for fixtures)",
    )
    _probe_arguments(scan)

    report = subparsers.add_parser("report", help="Render Markdown reports")
    report.add_argument("model_id", nargs="?")
    report.add_argument("--stdout", action="store_true")

    dashboard = subparsers.add_parser("serve", help="Serve the local read-only dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)

    return parser


def _probe_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fixture-responses", help="Use local response fixtures instead of API calls")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--transport", choices=["sync", "batch"])
    parser.add_argument("--backend", choices=["hash", "reptrace", "llm-dna"])
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--provider-only", help="Comma-separated OpenRouter provider slugs")
    parser.add_argument(
        "--allow-fallbacks",
        action=argparse.BooleanOptionalAction,
        default=None,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            target = write_default_config(args.path)
            target_probes = target.parent / "probes" / "default.json"
            if not target_probes.exists():
                target_probes.parent.mkdir(parents=True, exist_ok=True)
                resource = importlib.resources.files("model_radar.resources").joinpath(
                    "default_probes.json"
                )
                target_probes.write_text(resource.read_text(encoding="utf-8"), encoding="utf-8")
            config = load_config(target)
            db = RadarDB(config["database"])
            db.initialize()
            _print({"config": str(target), "database": str(db.path), "status": "initialized"})
            return

        config = load_config(args.config)
        if hasattr(args, "backend") and args.backend:
            config["fingerprint"]["backend"] = args.backend
        if hasattr(args, "dimension") and args.dimension:
            config["fingerprint"]["dimension"] = args.dimension
        db = RadarDB(config["database"])
        db.initialize()
        client = _client(config)

        if args.command == "discover":
            _print(discover(db, client, config, fixture=args.fixture))
        elif args.command == "trends":
            _print(
                ingest_rankings(
                    db,
                    client,
                    config,
                    fixture=args.fixture,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    period=args.period,
                    category=args.category,
                    language_type=args.language_type,
                    modality=args.modality,
                    context_bucket=args.context_bucket,
                )
            )
        elif args.command == "queue":
            minimum = (
                args.min_score
                if args.min_score is not None
                else float(config["priority"].get("minimum_score", 45))
            )
            rows = priority_queue(db, minimum_score=minimum, limit=args.limit)
            _print([_queue_row(row) for row in rows])
        elif args.command == "probe":
            _print(
                probe_model(
                    db,
                    client,
                    config,
                    args.model_id,
                    fixture_responses=args.fixture_responses,
                    overrides=_probe_overrides(args),
                )
            )
        elif args.command == "scan":
            if not args.fixture_responses and not args.yes_spend:
                raise ValueError("scan may spend API credits; pass --yes-spend or use fixtures")
            model_ids = (
                [item.strip() for item in args.models.split(",") if item.strip()]
                if args.models
                else None
            )
            _print(
                scan_queue(
                    db,
                    client,
                    config,
                    model_ids=model_ids,
                    minimum_score=args.min_score,
                    limit=args.limit,
                    fixture_responses=args.fixture_responses,
                    overrides=_probe_overrides(args),
                )
            )
        elif args.command == "report":
            if args.stdout:
                print(render_model_report(db, args.model_id) if args.model_id else render_index(db))
            else:
                _print(
                    {
                        "written": [
                            str(path)
                            for path in write_reports(db, config["reports_dir"], args.model_id)
                        ]
                    }
                )
        elif args.command == "serve":
            serve(db, args.host, args.port)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _client(config: dict[str, Any]) -> OpenRouterClient:
    settings = config["openrouter"]
    return OpenRouterClient(
        settings.get("api_key", ""),
        timeout_seconds=settings.get("timeout_seconds", 90),
        retries=settings.get("retries", 3),
        http_referer=settings.get("http_referer", ""),
        app_title=settings.get("app_title", "Model Radar"),
    )


def _probe_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for argument, key in (
        ("repetitions", "repetitions"),
        ("transport", "transport"),
        ("max_tokens", "max_tokens"),
        ("allow_fallbacks", "allow_fallbacks"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            overrides[key] = value
    if getattr(args, "provider_only", None):
        overrides["provider_only"] = [
            item.strip() for item in args.provider_only.split(",") if item.strip()
        ]
    return overrides


def _queue_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": row["model_id"],
        "score": row["priority_score"],
        "stealth": bool(row["is_stealth"]),
        "preview": bool(row["is_preview"]),
        "free": bool(row["is_free"]),
        "first_seen_at": row["first_seen_at"],
    }


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
