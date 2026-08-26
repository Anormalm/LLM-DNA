from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from .db import RadarDB
from .drift import evaluate_latest_drift, nearest_neighbors
from .fingerprint import create_backend, fingerprint_repetitions, probe_set_identity
from .notify import drift_message, send_webhook
from .openrouter import OpenRouterClient
from .scoring import rescore_all
from .utils import load_json, utc_now


def discover(
    db: RadarDB,
    client: OpenRouterClient,
    config: dict[str, Any],
    *,
    fixture: str | Path | None = None,
) -> dict[str, Any]:
    payload = load_json(fixture) if fixture else client.list_models()
    if isinstance(payload, dict):
        models = payload.get("data", [])
    else:
        models = payload
    if not isinstance(models, list):
        raise ValueError("Models payload must contain a list")
    observed_at = utc_now()
    stats = db.upsert_models([model for model in models if isinstance(model, dict)], observed_at)
    rescored = rescore_all(db, config)
    return {**stats, "seen": len(models), "rescored": rescored, "observed_at": observed_at}


def ingest_rankings(
    db: RadarDB,
    client: OpenRouterClient,
    config: dict[str, Any],
    *,
    fixture: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    period: str = "day",
    category: str = "",
    language_type: str = "",
    modality: str = "",
    context_bucket: str = "",
) -> dict[str, Any]:
    filters = {
        "period": period,
        "category": category,
        "language_type": language_type,
        "modality": modality,
        "context_bucket": context_bucket,
    }
    if fixture:
        payload = load_json(fixture)
    else:
        payload = client.rankings_daily(
            start_date=start_date,
            end_date=end_date,
            period=period,
            category=category,
            language_type=language_type,
            modality=modality,
            context_bucket=context_bucket,
        )
    rows = payload.get("data", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Rankings payload must contain a list")
    written = db.insert_rankings(rows, filters=filters)
    rescored = rescore_all(db, config)
    return {
        "written": written,
        "rescored": rescored,
        "estimated": bool(category or language_type),
        "meta": payload.get("meta", {}) if isinstance(payload, dict) else {},
    }


def priority_queue(
    db: RadarDB,
    *,
    minimum_score: float,
    limit: int,
) -> list[dict[str, Any]]:
    return [
        model
        for model in db.list_models(active_only=True)
        if float(model.get("priority_score") or 0.0) >= minimum_score
    ][:limit]


def probe_model(
    db: RadarDB,
    client: OpenRouterClient,
    config: dict[str, Any],
    model_id: str,
    *,
    fixture_responses: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    db.ensure_model(model_id)
    probe_payload = load_json(config["probes_file"])
    probe_set_id, probe_set_hash, probes = probe_set_identity(probe_payload)
    options = dict(config["probe"])
    options.update(overrides or {})
    repetitions = max(1, int(options.get("repetitions", 3)))
    transport = str(options.get("transport", "sync"))
    run_id = _run_id(model_id)
    db.start_run(
        {
            "run_id": run_id,
            "model_id": model_id,
            "transport": "fixture" if fixture_responses else transport,
            "repetitions": repetitions,
            "probe_set_id": probe_set_id,
            "probe_set_hash": probe_set_hash,
            "config": {
                **options,
                "probe_count": len(probes),
                "fixture": bool(fixture_responses),
            },
        }
    )

    try:
        if fixture_responses:
            items = _fixture_items(fixture_responses, model_id, probes, repetitions)
        elif transport == "batch":
            items = _batch_items(client, model_id, probes, repetitions, options)
        elif transport == "sync":
            items = _sync_items(client, model_id, probes, repetitions, options)
        else:
            raise ValueError(f"Unsupported transport: {transport}")

        for item in items:
            db.add_response(run_id, item)

        non_empty = sum(1 for item in items if str(item.get("response", "")).strip())
        expected = len(probes) * repetitions
        if non_empty < max(1, expected // 2):
            raise RuntimeError(f"Only {non_empty}/{expected} probe responses were non-empty")

        backend = create_backend(config["fingerprint"])
        vector, within, _ = fingerprint_repetitions(
            items,
            backend,
            prompt_count=len(probes),
            repetitions=repetitions,
        )
        fingerprint_key = backend.key(probe_set_hash)
        db.finish_run(
            run_id,
            status="completed",
            backend=backend.name,
            fingerprint_key=fingerprint_key,
            vector=vector.tolist(),
            within_run_distance=within,
        )
        drift = evaluate_latest_drift(db, model_id, fingerprint_key, config)
        neighbors = nearest_neighbors(
            db,
            model_id,
            fingerprint_key,
            limit=int(config["drift"].get("nearest_neighbors", 5)),
        )
        if drift and drift["alerted"] and config["alerts"].get("webhook_url"):
            send_webhook(
                config["alerts"]["webhook_url"],
                drift_message(drift),
                format_name=str(config["alerts"].get("format", "discord")),
            )
        return {
            "run_id": run_id,
            "model_id": model_id,
            "status": "completed",
            "responses": non_empty,
            "expected_responses": expected,
            "fingerprint_backend": backend.name,
            "fingerprint_key": fingerprint_key,
            "within_run_distance": within,
            "drift": drift,
            "neighbors": neighbors,
        }
    except Exception as exc:
        db.finish_run(run_id, status="failed", error=str(exc))
        raise


def scan_queue(
    db: RadarDB,
    client: OpenRouterClient,
    config: dict[str, Any],
    *,
    model_ids: Sequence[str] | None = None,
    minimum_score: float | None = None,
    limit: int = 3,
    fixture_responses: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if model_ids:
        selected = list(model_ids)[:limit]
    else:
        threshold = (
            float(minimum_score)
            if minimum_score is not None
            else float(config["priority"].get("minimum_score", 45))
        )
        selected = [
            row["model_id"] for row in priority_queue(db, minimum_score=threshold, limit=limit)
        ]
    return [
        probe_model(
            db,
            client,
            config,
            model_id,
            fixture_responses=fixture_responses,
            overrides=overrides,
        )
        for model_id in selected
    ]


def _sync_items(
    client: OpenRouterClient,
    model_id: str,
    probes: Sequence[str],
    repetitions: int,
    options: dict[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        for prompt_index, prompt in enumerate(probes):
            try:
                parsed = client.chat_completion(model_id, prompt, options)
            except Exception as exc:
                parsed = {
                    "response": "",
                    "error": str(exc),
                    "raw": None,
                    "created_at": utc_now(),
                }
            parsed.update(
                {
                    "prompt_index": prompt_index,
                    "repetition": repetition,
                    "prompt": prompt,
                }
            )
            items.append(parsed)
    return items


def _batch_items(
    client: OpenRouterClient,
    model_id: str,
    probes: Sequence[str],
    repetitions: int,
    options: dict[str, Any],
) -> list[dict[str, Any]]:
    submitted: list[tuple[str, str]] = []
    lookup: dict[str, tuple[int, int, str]] = {}
    for repetition in range(repetitions):
        for prompt_index, prompt in enumerate(probes):
            custom_id = f"r{repetition:03d}-p{prompt_index:05d}"
            submitted.append((custom_id, prompt))
            lookup[custom_id] = (repetition, prompt_index, prompt)
    results = client.run_batch(model_id, submitted, options)
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        custom_id = str(result.get("custom_id", ""))
        if custom_id not in lookup:
            continue
        seen.add(custom_id)
        repetition, prompt_index, prompt = lookup[custom_id]
        result.update(
            {"repetition": repetition, "prompt_index": prompt_index, "prompt": prompt}
        )
        items.append(result)
    for custom_id, (repetition, prompt_index, prompt) in lookup.items():
        if custom_id not in seen:
            items.append(
                {
                    "repetition": repetition,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "response": "",
                    "error": "Missing batch result",
                    "raw": None,
                    "created_at": utc_now(),
                }
            )
    items.sort(key=lambda item: (item["repetition"], item["prompt_index"]))
    return items


def _fixture_items(
    path: str | Path,
    model_id: str,
    probes: Sequence[str],
    repetitions: int,
) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("models"), dict):
        model_payload = payload["models"].get(model_id)
    else:
        model_payload = payload
    if isinstance(model_payload, dict):
        model_payload = model_payload.get("repetitions") or model_payload.get("responses")
    if not isinstance(model_payload, list):
        raise ValueError(f"Fixture has no responses for model {model_id}")
    if not model_payload:
        raise ValueError(f"Fixture response list is empty for model {model_id}")
    if model_payload and all(isinstance(item, str) for item in model_payload):
        model_payload = [model_payload]
    if not all(isinstance(rep, list) for rep in model_payload):
        raise ValueError("Fixture responses must be a list of repetitions")

    items: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        source_rep = model_payload[min(repetition, len(model_payload) - 1)]
        for prompt_index, prompt in enumerate(probes):
            response = str(source_rep[prompt_index]) if prompt_index < len(source_rep) else ""
            items.append(
                {
                    "prompt_index": prompt_index,
                    "repetition": repetition,
                    "prompt": prompt,
                    "response": response,
                    "generation_id": f"fixture-{repetition}-{prompt_index}",
                    "returned_model": model_id,
                    "provider": "fixture",
                    "finish_reason": "stop",
                    "usage": None,
                    "raw": {"fixture": True},
                    "error": None,
                    "created_at": utc_now(),
                }
            )
    return items


def _run_id(model_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(character if character.isalnum() else "-" for character in model_id)[:48]
    return f"{timestamp}-{safe}-{uuid.uuid4().hex[:8]}"
