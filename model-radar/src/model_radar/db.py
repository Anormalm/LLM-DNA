from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .utils import canonical_json, json_hash, utc_now


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    canonical_slug TEXT,
    name TEXT,
    description TEXT,
    created_unix INTEGER,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    is_stealth INTEGER NOT NULL DEFAULT 0,
    is_preview INTEGER NOT NULL DEFAULT 0,
    is_free INTEGER NOT NULL DEFAULT 0,
    priority_score REAL NOT NULL DEFAULT 0,
    score_components_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at TEXT NOT NULL,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    metadata_hash TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    UNIQUE(model_id, metadata_hash)
);

CREATE TABLE IF NOT EXISTS ranking_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    model_permaslug TEXT NOT NULL,
    total_tokens INTEGER,
    rank INTEGER,
    period TEXT NOT NULL DEFAULT 'day',
    category TEXT NOT NULL DEFAULT '',
    language_type TEXT NOT NULL DEFAULT '',
    modality TEXT NOT NULL DEFAULT '',
    context_bucket TEXT NOT NULL DEFAULT '',
    is_estimated INTEGER NOT NULL DEFAULT 0,
    observed_at TEXT NOT NULL,
    UNIQUE(date, model_permaslug, period, category, language_type, modality, context_bucket)
);

CREATE TABLE IF NOT EXISTS probe_runs (
    run_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    transport TEXT NOT NULL,
    repetitions INTEGER NOT NULL,
    probe_set_id TEXT NOT NULL,
    probe_set_hash TEXT NOT NULL,
    config_json TEXT NOT NULL,
    fingerprint_backend TEXT,
    fingerprint_key TEXT,
    fingerprint_dimension INTEGER,
    fingerprint_json TEXT,
    within_run_distance REAL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS responses (
    response_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES probe_runs(run_id) ON DELETE CASCADE,
    prompt_index INTEGER NOT NULL,
    repetition INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL DEFAULT '',
    generation_id TEXT,
    returned_model TEXT,
    provider TEXT,
    finish_reason TEXT,
    usage_json TEXT,
    raw_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, prompt_index, repetition)
);

CREATE TABLE IF NOT EXISTS drift_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    previous_run_id TEXT NOT NULL REFERENCES probe_runs(run_id),
    current_run_id TEXT NOT NULL REFERENCES probe_runs(run_id),
    cosine_similarity REAL NOT NULL,
    distance REAL NOT NULL,
    threshold_used REAL NOT NULL,
    alerted INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(previous_run_id, current_run_id)
);

CREATE INDEX IF NOT EXISTS idx_rankings_model_date
ON ranking_observations(model_permaslug, date);
CREATE INDEX IF NOT EXISTS idx_runs_model_time
ON probe_runs(model_id, completed_at);
CREATE INDEX IF NOT EXISTS idx_runs_fingerprint_key
ON probe_runs(fingerprint_key, completed_at);
"""


class RadarDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def upsert_models(self, models: Sequence[dict[str, Any]], observed_at: str | None = None) -> dict[str, int]:
        observed_at = observed_at or utc_now()
        inserted = updated = unchanged = 0
        seen_ids: set[str] = set()
        with self.connect() as connection:
            for model in models:
                model_id = str(model.get("id") or model.get("canonical_slug") or "").strip()
                if not model_id:
                    continue
                seen_ids.add(model_id)
                raw = canonical_json(model)
                raw_hash = json_hash(model)
                existing = connection.execute(
                    "SELECT metadata_json FROM models WHERE model_id = ?", (model_id,)
                ).fetchone()
                flags = _model_flags(model)
                if existing is None:
                    inserted += 1
                    connection.execute(
                        """
                        INSERT INTO models (
                            model_id, canonical_slug, name, description, created_unix,
                            first_seen_at, last_seen_at, active, is_stealth,
                            is_preview, is_free, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                        """,
                        (
                            model_id,
                            model.get("canonical_slug"),
                            model.get("name"),
                            model.get("description"),
                            _safe_int(model.get("created")),
                            observed_at,
                            observed_at,
                            flags["is_stealth"],
                            flags["is_preview"],
                            flags["is_free"],
                            raw,
                        ),
                    )
                else:
                    if existing["metadata_json"] == raw:
                        unchanged += 1
                    else:
                        updated += 1
                    connection.execute(
                        """
                        UPDATE models SET canonical_slug=?, name=?, description=?, created_unix=?,
                            last_seen_at=?, active=1, is_stealth=?, is_preview=?, is_free=?, metadata_json=?
                        WHERE model_id=?
                        """,
                        (
                            model.get("canonical_slug"),
                            model.get("name"),
                            model.get("description"),
                            _safe_int(model.get("created")),
                            observed_at,
                            flags["is_stealth"],
                            flags["is_preview"],
                            flags["is_free"],
                            raw,
                            model_id,
                        ),
                    )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO model_snapshots
                    (observed_at, model_id, metadata_hash, metadata_json) VALUES (?, ?, ?, ?)
                    """,
                    (observed_at, model_id, raw_hash, raw),
                )
            if seen_ids:
                placeholders = ",".join("?" for _ in seen_ids)
                connection.execute(
                    f"UPDATE models SET active=0 WHERE model_id NOT IN ({placeholders})",
                    tuple(sorted(seen_ids)),
                )
        return {"inserted": inserted, "updated": updated, "unchanged": unchanged}

    def insert_rankings(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        filters: dict[str, str] | None = None,
        observed_at: str | None = None,
    ) -> int:
        observed_at = observed_at or utc_now()
        filters = filters or {}
        period = filters.get("period", "day")
        category = filters.get("category", "")
        language_type = filters.get("language_type", "")
        modality = filters.get("modality", "")
        context_bucket = filters.get("context_bucket", "")
        is_estimated = int(bool(category or language_type))
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("date", "")), []).append(row)

        written = 0
        with self.connect() as connection:
            for date, date_rows in grouped.items():
                real_rows = [r for r in date_rows if r.get("model_permaslug") != "other"]
                sorted_rows = sorted(real_rows, key=lambda item: _safe_int(item.get("total_tokens")) or 0, reverse=True)
                rank_by_slug = {str(row.get("model_permaslug")): rank for rank, row in enumerate(sorted_rows, 1)}
                for row in date_rows:
                    slug = str(row.get("model_permaslug", ""))
                    if not date or not slug:
                        continue
                    connection.execute(
                        """
                        INSERT INTO ranking_observations
                        (date, model_permaslug, total_tokens, rank, period, category,
                         language_type, modality, context_bucket, is_estimated, observed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(date, model_permaslug, period, category, language_type,
                                    modality, context_bucket)
                        DO UPDATE SET total_tokens=excluded.total_tokens, rank=excluded.rank,
                                      is_estimated=excluded.is_estimated,
                                      observed_at=excluded.observed_at
                        """,
                        (
                            date,
                            slug,
                            _safe_int(row.get("total_tokens")),
                            rank_by_slug.get(slug),
                            period,
                            category,
                            language_type,
                            modality,
                            context_bucket,
                            is_estimated,
                            observed_at,
                        ),
                    )
                    written += 1
        return written

    def update_score(self, model_id: str, score: float, components: dict[str, float]) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE models SET priority_score=?, score_components_json=? WHERE model_id=?",
                (float(score), canonical_json(components), model_id),
            )

    def list_models(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM models"
        params: tuple[Any, ...] = ()
        if active_only:
            query += " WHERE active=1"
        query += " ORDER BY priority_score DESC, first_seen_at DESC"
        with self.connect() as connection:
            return [_row_dict(row) for row in connection.execute(query, params).fetchall()]

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM models WHERE model_id=?", (model_id,)).fetchone()
            return _row_dict(row) if row else None

    def ensure_model(self, model_id: str, observed_at: str | None = None) -> None:
        if self.get_model(model_id) is not None:
            return
        observed_at = observed_at or utc_now()
        metadata = {"id": model_id, "canonical_slug": model_id, "name": model_id}
        flags = _model_flags(metadata)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO models
                (model_id, canonical_slug, name, description, created_unix,
                 first_seen_at, last_seen_at, active, is_stealth, is_preview,
                 is_free, metadata_json)
                VALUES (?, ?, ?, '', NULL, ?, ?, 1, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    model_id,
                    model_id,
                    observed_at,
                    observed_at,
                    flags["is_stealth"],
                    flags["is_preview"],
                    flags["is_free"],
                    canonical_json(metadata),
                ),
            )

    def latest_rankings(self, model_id: str, limit: int = 30) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM ranking_observations
                WHERE model_permaslug=? AND category='' AND language_type=''
                      AND modality='' AND context_bucket=''
                ORDER BY date DESC LIMIT ?
                """,
                (model_id, limit),
            ).fetchall()
            return [_row_dict(row) for row in rows]

    def start_run(self, run: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO probe_runs
                (run_id, model_id, started_at, status, transport, repetitions,
                 probe_set_id, probe_set_hash, config_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run["run_id"],
                    run["model_id"],
                    run.get("started_at", utc_now()),
                    run.get("status", "running"),
                    run["transport"],
                    int(run["repetitions"]),
                    run["probe_set_id"],
                    run["probe_set_hash"],
                    canonical_json(run.get("config", {})),
                ),
            )

    def add_response(self, run_id: str, item: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO responses
                (run_id, prompt_index, repetition, prompt, response, generation_id,
                 returned_model, provider, finish_reason, usage_json, raw_json, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    int(item["prompt_index"]),
                    int(item["repetition"]),
                    item["prompt"],
                    item.get("response", ""),
                    item.get("generation_id"),
                    item.get("returned_model"),
                    item.get("provider"),
                    item.get("finish_reason"),
                    canonical_json(item.get("usage")) if item.get("usage") is not None else None,
                    canonical_json(item.get("raw")) if item.get("raw") is not None else None,
                    item.get("error"),
                    item.get("created_at", utc_now()),
                ),
            )

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        backend: str | None = None,
        fingerprint_key: str | None = None,
        vector: Sequence[float] | None = None,
        within_run_distance: float | None = None,
        error: str | None = None,
    ) -> None:
        payload = json.dumps([float(value) for value in vector]) if vector is not None else None
        dimension = len(vector) if vector is not None else None
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE probe_runs SET completed_at=?, status=?, fingerprint_backend=?,
                    fingerprint_key=?, fingerprint_dimension=?, fingerprint_json=?,
                    within_run_distance=?, error=? WHERE run_id=?
                """,
                (
                    utc_now(),
                    status,
                    backend,
                    fingerprint_key,
                    dimension,
                    payload,
                    within_run_distance,
                    error,
                    run_id,
                ),
            )

    def get_responses(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM responses WHERE run_id=? ORDER BY repetition, prompt_index",
                (run_id,),
            ).fetchall()
            return [_row_dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM probe_runs WHERE run_id=?", (run_id,)).fetchone()
            return _decode_run(row) if row else None

    def latest_runs(
        self,
        model_id: str,
        *,
        fingerprint_key: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM probe_runs WHERE model_id=? AND status='completed'"
        params: list[Any] = [model_id]
        if fingerprint_key:
            query += " AND fingerprint_key=?"
            params.append(fingerprint_key)
        query += " ORDER BY completed_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as connection:
            return [_decode_run(row) for row in connection.execute(query, params).fetchall()]

    def latest_fingerprints(self, fingerprint_key: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT r.* FROM probe_runs r
                JOIN (
                    SELECT model_id, MAX(completed_at) AS completed_at
                    FROM probe_runs
                    WHERE status='completed' AND fingerprint_key=?
                    GROUP BY model_id
                ) latest
                ON r.model_id=latest.model_id AND r.completed_at=latest.completed_at
                WHERE r.fingerprint_key=?
                """,
                (fingerprint_key, fingerprint_key),
            ).fetchall()
            return [_decode_run(row) for row in rows]

    def record_drift(self, event: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO drift_events
                (model_id, previous_run_id, current_run_id, cosine_similarity,
                 distance, threshold_used, alerted, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["model_id"],
                    event["previous_run_id"],
                    event["current_run_id"],
                    float(event["cosine_similarity"]),
                    float(event["distance"]),
                    float(event["threshold_used"]),
                    int(bool(event["alerted"])),
                    event.get("created_at", utc_now()),
                ),
            )

    def latest_drift(self, model_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM drift_events WHERE model_id=? ORDER BY created_at DESC LIMIT 1",
                (model_id,),
            ).fetchone()
            return _row_dict(row) if row else None


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _zero_price(value: Any) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def _model_flags(model: dict[str, Any]) -> dict[str, int]:
    text = " ".join(
        str(model.get(field, "")) for field in ("id", "canonical_slug", "name", "description")
    ).lower()
    pricing = model.get("pricing") if isinstance(model.get("pricing"), dict) else {}
    is_free = bool(pricing) and _zero_price(pricing.get("prompt")) and _zero_price(
        pricing.get("completion")
    )
    return {
        "is_stealth": int("stealth" in text),
        "is_preview": int(any(word in text for word in ("alpha", "preview", "experimental"))),
        "is_free": int(is_free),
    }


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def _decode_run(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    if result.get("fingerprint_json"):
        result["fingerprint"] = json.loads(result["fingerprint_json"])
    else:
        result["fingerprint"] = None
    return result
