from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .db import RadarDB
from .drift import nearest_neighbors
from .scoring import trend_features


def render_model_report(db: RadarDB, model_id: str, *, neighbor_limit: int = 5) -> str:
    model = db.get_model(model_id)
    if not model:
        raise KeyError(f"Unknown model: {model_id}")
    metadata = _decode(model.get("metadata_json"))
    components = _decode(model.get("score_components_json"))
    trends = trend_features(db.latest_rankings(model_id, limit=30))
    runs = db.latest_runs(model_id, limit=5)
    latest_run = runs[0] if runs else None
    drift = db.latest_drift(model_id)
    neighbors = (
        nearest_neighbors(db, model_id, latest_run["fingerprint_key"], limit=neighbor_limit)
        if latest_run and latest_run.get("fingerprint_key")
        else []
    )

    tags = []
    if model.get("is_stealth"):
        tags.append("stealth")
    if model.get("is_preview"):
        tags.append("preview")
    if model.get("is_free"):
        tags.append("free")

    lines = [
        f"# {model.get('name') or model_id}",
        "",
        f"- Model ID: `{model_id}`",
        f"- Status: {'active' if model.get('active') else 'inactive'}",
        f"- First seen: {model.get('first_seen_at')}",
        f"- Last seen: {model.get('last_seen_at')}",
        f"- Tags: {', '.join(tags) if tags else 'none'}",
        f"- Radar score: {float(model.get('priority_score') or 0):.2f}/100",
        "",
        str(metadata.get("description") or "No description supplied."),
        "",
        "## Priority signals",
        "",
        "| Signal | Value |",
        "| --- | ---: |",
    ]
    for key in ("novelty", "trend", "stealth", "free_preview", "capability_interest"):
        lines.append(f"| {key} | {float(components.get(key, 0.0)):.3f} |")

    lines.extend(["", "## Usage trend", ""])
    if trends["censored"]:
        lines.append("No exact top-50 daily observation is available; absence is treated as censored, not zero usage.")
    else:
        lines.extend(
            [
                f"- Latest top-50 rank: {trends.get('latest_rank') or 'unknown'}",
                f"- Latest tokens: {trends.get('latest_tokens'):,}",
                f"- 1-day growth: {_percent(trends.get('growth_1d'))}",
                f"- 7-day growth: {_percent(trends.get('growth_7d'))}",
                f"- Rank jump: {trends.get('rank_jump') if trends.get('rank_jump') is not None else 'n/a'}",
            ]
        )

    lines.extend(["", "## Behavioral fingerprint", ""])
    if not latest_run:
        lines.append("No completed fingerprint run yet.")
    else:
        lines.extend(
            [
                f"- Last run: `{latest_run['run_id']}`",
                f"- Backend: `{latest_run.get('fingerprint_backend')}`",
                f"- Protocol key: `{latest_run.get('fingerprint_key')}`",
                f"- Within-run distance: {float(latest_run.get('within_run_distance') or 0):.4f}",
            ]
        )
        if drift:
            state = "possible change" if drift.get("alerted") else "stable"
            lines.extend(
                [
                    f"- Drift status: **{state}**",
                    f"- Latest drift distance: {float(drift['distance']):.4f}",
                    f"- Alert threshold: {float(drift['threshold_used']):.4f}",
                ]
            )

    lines.extend(["", "## Behavioral neighbors", ""])
    if not neighbors:
        lines.append("No compatible reference fingerprints are available.")
    else:
        lines.extend(["| Model | Similarity | Distance |", "| --- | ---: | ---: |"])
        for item in neighbors:
            lines.append(
                f"| `{item['model_id']}` | {item['similarity']:.4f} | {item['distance']:.4f} |"
            )

    lines.extend(
        [
            "",
            "> Behavioral proximity is not proof of shared weights, provenance, or model identity.",
            "",
        ]
    )
    return "\n".join(lines)


def render_index(db: RadarDB) -> str:
    models = db.list_models(active_only=False)
    lines = [
        "# Model Radar",
        "",
        "| Model | Status | Tags | Score | First seen | Last seen |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for model in models:
        tags = ", ".join(
            label
            for label, flag in (
                ("stealth", model.get("is_stealth")),
                ("preview", model.get("is_preview")),
                ("free", model.get("is_free")),
            )
            if flag
        )
        lines.append(
            f"| `{model['model_id']}` | {'active' if model.get('active') else 'inactive'} | "
            f"{tags or '—'} | {float(model.get('priority_score') or 0):.2f} | "
            f"{model.get('first_seen_at')} | {model.get('last_seen_at')} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(db: RadarDB, reports_dir: str | Path, model_id: str | None = None) -> list[Path]:
    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if model_id:
        target = root / f"{_safe_name(model_id)}.md"
        target.write_text(render_model_report(db, model_id), encoding="utf-8")
        return [target]
    index = root / "index.md"
    index.write_text(render_index(db), encoding="utf-8")
    written.append(index)
    for model in db.list_models(active_only=False):
        target = root / f"{_safe_name(model['model_id'])}.md"
        target.write_text(render_model_report(db, model["model_id"]), encoding="utf-8")
        written.append(target)
    return written


def _decode(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):+.1%}"


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)

