"""Prompt, model, and decoding manifests for stochastic response collection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    text: str
    split: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.prompt_id, str)
            or not isinstance(self.text, str)
            or not self.prompt_id
            or not self.text
        ):
            raise ValueError("prompt_id and text must be non-empty")
        if not isinstance(self.split, str) or self.split not in {
            "calibration",
            "evaluation",
        }:
            raise ValueError("prompt split must be 'calibration' or 'evaluation'")


@dataclass(frozen=True)
class DecodingSetting:
    setting_id: str
    temperature: float
    top_p: float
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        if not isinstance(self.setting_id, str) or not self.setting_id:
            raise ValueError("setting_id must be non-empty")
        if (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValueError("temperature must be finite and non-negative")
        if (
            not isinstance(self.top_p, (int, float))
            or isinstance(self.top_p, bool)
            or not math.isfinite(self.top_p)
            or not 0 < self.top_p <= 1
        ):
            raise ValueError("top_p must be in (0, 1]")
        if (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or self.max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be positive")
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "top_p", float(self.top_p))


@dataclass(frozen=True)
class CollectionManifest:
    dataset_id: str
    model_ids: Tuple[str, ...]
    settings: Tuple[DecodingSetting, ...]
    prompts: Tuple[Prompt, ...]
    generations: int
    random_seed: int = 2027
    metadata: Dict[str, Any] = field(default_factory=dict)
    format_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")
        if not self.model_ids or any(
            not isinstance(value, str) or not value for value in self.model_ids
        ):
            raise ValueError("model_ids must contain non-empty identifiers")
        if any(not isinstance(value, DecodingSetting) for value in self.settings):
            raise ValueError("settings must contain DecodingSetting objects")
        if any(not isinstance(value, Prompt) for value in self.prompts):
            raise ValueError("prompts must contain Prompt objects")
        if len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("model_ids must be unique")
        setting_ids = self.setting_ids
        if not setting_ids or len(set(setting_ids)) != len(setting_ids):
            raise ValueError("decoding setting IDs must be non-empty and unique")
        prompt_ids = tuple(prompt.prompt_id for prompt in self.prompts)
        if not prompt_ids or len(set(prompt_ids)) != len(prompt_ids):
            raise ValueError("prompt IDs must be non-empty and unique")
        splits = {prompt.split for prompt in self.prompts}
        if splits != {"calibration", "evaluation"}:
            raise ValueError("manifest must contain calibration and evaluation prompts")
        if (
            not isinstance(self.generations, int)
            or isinstance(self.generations, bool)
            or self.generations <= 0
        ):
            raise ValueError("generations must be positive")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise ValueError("random_seed must be an integer")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a JSON object")
        try:
            json.dumps(self.metadata, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON serializable") from exc
        if (
            not isinstance(self.format_version, int)
            or isinstance(self.format_version, bool)
            or self.format_version != 1
        ):
            raise ValueError(f"unsupported manifest format_version: {self.format_version}")
        object.__setattr__(self, "model_ids", tuple(self.model_ids))
        object.__setattr__(self, "settings", tuple(self.settings))
        object.__setattr__(self, "prompts", tuple(self.prompts))

    @property
    def setting_ids(self) -> Tuple[str, ...]:
        return tuple(setting.setting_id for setting in self.settings)

    def prompts_for_split(self, split: str) -> Tuple[Prompt, ...]:
        if split not in {"calibration", "evaluation"}:
            raise ValueError(f"unknown prompt split: {split}")
        return tuple(prompt for prompt in self.prompts if prompt.split == split)

    def setting(self, setting_id: str) -> DecodingSetting:
        matches = [setting for setting in self.settings if setting.setting_id == setting_id]
        if len(matches) != 1:
            raise KeyError(f"unknown decoding setting: {setting_id}")
        return matches[0]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "dataset_id": self.dataset_id,
            "model_ids": list(self.model_ids),
            "settings": [asdict(setting) for setting in self.settings],
            "prompts": [asdict(prompt) for prompt in self.prompts],
            "generations": self.generations,
            "random_seed": self.random_seed,
            "metadata": self.metadata,
        }

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}-", dir=target.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    self.as_dict(), stream, indent=2, sort_keys=True, ensure_ascii=False
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, target)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return target

    @classmethod
    def load(cls, path: str | Path) -> "CollectionManifest":
        source = Path(path)
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("collection manifest root must be a JSON object")
        allowed = {
            "format_version",
            "dataset_id",
            "model_ids",
            "settings",
            "prompts",
            "generations",
            "random_seed",
            "metadata",
        }
        unknown = sorted(set(payload).difference(allowed))
        if unknown:
            raise ValueError(f"unknown collection manifest keys: {unknown}")
        required = {"dataset_id", "model_ids", "settings", "prompts", "generations"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"collection manifest is missing keys: {missing}")
        if not isinstance(payload["model_ids"], list):
            raise ValueError("model_ids must be a JSON array")
        if not isinstance(payload["settings"], list) or any(
            not isinstance(item, dict) for item in payload["settings"]
        ):
            raise ValueError("settings must be an array of JSON objects")
        if not isinstance(payload["prompts"], list) or any(
            not isinstance(item, dict) for item in payload["prompts"]
        ):
            raise ValueError("prompts must be an array of JSON objects")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        return cls(
            format_version=payload.get("format_version", 1),
            dataset_id=payload["dataset_id"],
            model_ids=tuple(payload["model_ids"]),
            settings=tuple(DecodingSetting(**item) for item in payload["settings"]),
            prompts=tuple(Prompt(**item) for item in payload["prompts"]),
            generations=payload["generations"],
            random_seed=payload.get("random_seed", 2027),
            metadata=metadata,
        )


def set_uniform_token_limit(
    manifest: CollectionManifest, *, max_new_tokens: int, dataset_id: str
) -> CollectionManifest:
    """Clone a manifest with one token ceiling and explicit parent provenance."""

    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string")
    previous_limits = sorted({item.max_new_tokens for item in manifest.settings})
    if previous_limits == [max_new_tokens]:
        raise ValueError("new token limit must differ from the source manifest")
    metadata = dict(manifest.metadata)
    metadata["parent_manifest_fingerprint"] = manifest.fingerprint
    metadata["parent_dataset_id"] = manifest.dataset_id
    metadata["parent_max_new_tokens"] = previous_limits
    return replace(
        manifest,
        dataset_id=dataset_id.strip(),
        settings=tuple(
            replace(setting, max_new_tokens=max_new_tokens)
            for setting in manifest.settings
        ),
        metadata=metadata,
    )
