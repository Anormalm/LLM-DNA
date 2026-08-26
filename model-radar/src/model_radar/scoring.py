from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any, Sequence

from .db import RadarDB
from .utils import parse_iso


def trend_features(rows: Sequence[dict[str, Any]]) -> dict[str, float | int | None | bool]:
    """Compute within-model trends without converting censored top-50 absences to zeros."""
    real = [row for row in rows if row.get("total_tokens") is not None]
    real.sort(key=lambda row: str(row.get("date")))
    if not real:
        return {
            "growth_1d": None,
            "growth_7d": None,
            "rank_jump": None,
            "latest_tokens": None,
            "latest_rank": None,
            "censored": True,
        }

    latest = real[-1]
    growth_1d = None
    if len(real) >= 2 and int(real[-2]["total_tokens"]) > 0:
        growth_1d = int(latest["total_tokens"]) / int(real[-2]["total_tokens"]) - 1.0

    growth_7d = None
    if len(real) >= 14:
        current = sum(int(row["total_tokens"]) for row in real[-7:])
        previous = sum(int(row["total_tokens"]) for row in real[-14:-7])
        if previous > 0:
            growth_7d = current / previous - 1.0

    rank_jump = None
    ranked = [row for row in real if row.get("rank") is not None]
    if len(ranked) >= 2:
        rank_jump = int(ranked[-2]["rank"]) - int(ranked[-1]["rank"])

    return {
        "growth_1d": growth_1d,
        "growth_7d": growth_7d,
        "rank_jump": rank_jump,
        "latest_tokens": int(latest["total_tokens"]),
        "latest_rank": latest.get("rank"),
        "censored": False,
    }


def score_model(
    model: dict[str, Any],
    trend: dict[str, Any],
    config: dict[str, Any],
    *,
    now: datetime | None = None,
) -> tuple[float, dict[str, float]]:
    now = now or datetime.now(UTC)
    weights = config["priority"]["weights"]
    metadata = _json_object(model.get("metadata_json"))

    first_seen = parse_iso(model.get("first_seen_at"))
    days_seen = (now - first_seen).total_seconds() / 86400 if first_seen else 9999
    if days_seen <= 1:
        novelty = 1.0
    elif days_seen <= 7:
        novelty = 0.8
    elif days_seen <= 30:
        novelty = 0.4
    else:
        novelty = 0.0

    growth = trend.get("growth_7d")
    if growth is None:
        growth = trend.get("growth_1d")
    if growth is None:
        trend_score = 0.0
    else:
        # Smoothly map non-negative growth to [0, 1); negative growth contributes no priority.
        trend_score = 1.0 - math.exp(-max(0.0, float(growth)))
    rank_jump = trend.get("rank_jump")
    if rank_jump is not None and rank_jump > 0:
        trend_score = max(trend_score, min(1.0, float(rank_jump) / 20.0))

    stealth = float(bool(model.get("is_stealth")))
    free_preview = float(bool(model.get("is_free") and model.get("is_preview")))
    searchable = " ".join(
        str(metadata.get(field, "")) for field in ("id", "name", "description")
    ).lower()
    capability_keywords = [str(word).lower() for word in config["priority"]["capability_keywords"]]
    matches = sum(1 for word in capability_keywords if word in searchable)
    capability_interest = min(1.0, matches / 3.0)

    components = {
        "novelty": novelty,
        "trend": trend_score,
        "stealth": stealth,
        "free_preview": free_preview,
        "capability_interest": capability_interest,
    }
    weighted = sum(float(weights.get(name, 0.0)) * value for name, value in components.items())
    total_weight = sum(max(0.0, float(value)) for value in weights.values()) or 1.0
    score = 100.0 * weighted / total_weight
    return round(score, 2), {name: round(value, 4) for name, value in components.items()}


def rescore_all(db: RadarDB, config: dict[str, Any]) -> int:
    count = 0
    for model in db.list_models(active_only=False):
        features = trend_features(db.latest_rankings(model["model_id"], limit=30))
        score, components = score_model(model, features, config)
        db.update_score(model["model_id"], score, components)
        count += 1
    return count


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}

