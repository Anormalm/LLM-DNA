from __future__ import annotations

from typing import Any

from .db import RadarDB
from .utils import cosine_similarity, utc_now


def evaluate_latest_drift(
    db: RadarDB,
    model_id: str,
    fingerprint_key: str,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    runs = db.latest_runs(model_id, fingerprint_key=fingerprint_key, limit=2)
    if len(runs) < 2:
        return None
    current, previous = runs[0], runs[1]
    if not current.get("fingerprint") or not previous.get("fingerprint"):
        return None
    similarity = cosine_similarity(previous["fingerprint"], current["fingerprint"])
    distance = max(0.0, 1.0 - similarity)
    static_threshold = float(config["drift"].get("distance_threshold", 0.25))
    noise_multiplier = float(config["drift"].get("noise_multiplier", 3.0))
    noise_floor = noise_multiplier * max(
        float(previous.get("within_run_distance") or 0.0),
        float(current.get("within_run_distance") or 0.0),
    )
    threshold = max(static_threshold, noise_floor)
    event = {
        "model_id": model_id,
        "previous_run_id": previous["run_id"],
        "current_run_id": current["run_id"],
        "cosine_similarity": similarity,
        "distance": distance,
        "threshold_used": threshold,
        "alerted": distance >= threshold,
        "created_at": utc_now(),
    }
    db.record_drift(event)
    return event


def nearest_neighbors(
    db: RadarDB,
    model_id: str,
    fingerprint_key: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    runs = db.latest_fingerprints(fingerprint_key)
    target = next((run for run in runs if run["model_id"] == model_id), None)
    if not target or not target.get("fingerprint"):
        return []
    matches: list[dict[str, Any]] = []
    for run in runs:
        if run["model_id"] == model_id or not run.get("fingerprint"):
            continue
        similarity = cosine_similarity(target["fingerprint"], run["fingerprint"])
        matches.append(
            {
                "model_id": run["model_id"],
                "run_id": run["run_id"],
                "similarity": similarity,
                "distance": max(0.0, 1.0 - similarity),
            }
        )
    matches.sort(key=lambda item: item["distance"])
    return matches[:limit]

