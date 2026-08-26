from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "database": "data/radar.db",
    "reports_dir": "reports",
    "probes_file": "probes/default.json",
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "timeout_seconds": 90,
        "retries": 3,
        "http_referer": "",
        "app_title": "Model Radar",
    },
    "discovery": {"keywords": ["stealth", "alpha", "preview", "experimental"]},
    "priority": {
        "minimum_score": 45,
        "weights": {
            "novelty": 0.30,
            "trend": 0.25,
            "stealth": 0.20,
            "free_preview": 0.15,
            "capability_interest": 0.10,
        },
        "capability_keywords": [
            "coding",
            "reasoning",
            "agentic",
            "tool",
            "multimodal",
            "long context",
        ],
    },
    "probe": {
        "repetitions": 3,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
        "system_prompt": "",
        "transport": "sync",
        "provider_only": [],
        "allow_fallbacks": False,
        "poll_interval_seconds": 30,
        "batch_timeout_seconds": 86400,
    },
    "fingerprint": {
        "backend": "hash",
        "dimension": 256,
        "seed": 42,
        "sentence_encoder": "sentence-transformers/all-mpnet-base-v2",
    },
    "drift": {
        "distance_threshold": 0.25,
        "noise_multiplier": 3.0,
        "nearest_neighbors": 5,
    },
    "alerts": {"webhook_env": "RADAR_WEBHOOK_URL", "format": "discord"},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration and resolve filesystem paths relative to the config file."""
    config_path: Path | None = None
    if path:
        config_path = Path(path).expanduser().resolve()
    else:
        for candidate in (Path("model-radar.yaml"), Path("model-radar.yml")):
            if candidate.exists():
                config_path = candidate.resolve()
                break

    override: dict[str, Any] = {}
    base_dir = Path.cwd()
    if config_path:
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Configuration root must be a mapping")
        override = loaded
        base_dir = config_path.parent

    config = _deep_merge(DEFAULT_CONFIG, override)
    for key in ("database", "reports_dir", "probes_file"):
        config[key] = _resolve_path(str(config[key]), base_dir)

    config["openrouter"]["api_key"] = os.getenv(
        config["openrouter"].get("api_key_env", "OPENROUTER_API_KEY"), ""
    ).strip()
    config["alerts"]["webhook_url"] = os.getenv(
        config["alerts"].get("webhook_env", "RADAR_WEBHOOK_URL"), ""
    ).strip()
    return config


def write_default_config(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing config: {target}")
    serializable = copy.deepcopy(DEFAULT_CONFIG)
    target.write_text(yaml.safe_dump(serializable, sort_keys=False), encoding="utf-8")
    return target

