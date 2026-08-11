"""Strict audit and import support for legacy LLM-DNA response folders."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

from .manifest import CollectionManifest, DecodingSetting
from .responses import ResponseCache, ResponseDataset, ResponseRecord


_RUN_NAME = re.compile(
    r"^(?P<model>.+)_t(?P<temperature>\d+)_p(?P<top_p>\d+)_r(?P<repeat>\d+)$"
)
_SHA1 = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class LegacyAuditIssue:
    severity: str
    code: str
    path: str
    detail: str

    def __post_init__(self) -> None:
        if self.severity not in {"error", "warning"}:
            raise ValueError("audit issue severity must be error or warning")


@dataclass(frozen=True)
class LegacyAuditReport:
    source_dir: str
    manifest_fingerprint: str
    scanned_runs: int
    parsed_runs: int
    expected_records: int
    candidate_records: int
    missing_records: int
    models: Tuple[str, ...]
    settings: Tuple[str, ...]
    issues: Tuple[LegacyAuditIssue, ...]
    records: Tuple[ResponseRecord, ...] = field(repr=False)

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    @property
    def complete(self) -> bool:
        return self.missing_records == 0 and self.candidate_records == self.expected_records

    @property
    def ready_for_import(self) -> bool:
        return self.error_count == 0 and self.complete

    def as_dict(self) -> Dict[str, Any]:
        return {
            "format_version": 1,
            "source_format": "llm-dna-responses-v1",
            "source_dir": self.source_dir,
            "manifest_fingerprint": self.manifest_fingerprint,
            "scanned_runs": self.scanned_runs,
            "parsed_runs": self.parsed_runs,
            "expected_records": self.expected_records,
            "candidate_records": self.candidate_records,
            "missing_records": self.missing_records,
            "models": list(self.models),
            "settings": list(self.settings),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "complete": self.complete,
            "ready_for_import": self.ready_for_import,
            "issues": [asdict(issue) for issue in self.issues],
        }

    def write(self, path: str | Path) -> Path:
        return _write_json_atomic(path, self.as_dict())


def load_model_aliases(path: str | Path | None) -> Dict[str, str]:
    if path is None:
        return {}
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value
        for key, value in payload.items()
    ):
        raise ValueError("model aliases must be a JSON object of non-empty strings")
    return dict(payload)


def audit_llm_dna_responses(
    source_dir: str | Path,
    manifest: CollectionManifest,
    model_aliases: Mapping[str, str] | None = None,
    repeat_base: int = 1,
) -> LegacyAuditReport:
    """Audit legacy ``*_tXX_pXX_rN/responses.json`` directories.

    Import eligibility is intentionally strict: every manifest cell must be present, every
    response must expose its actual generation seed, and every run must identify the exact model
    revision pinned by the destination manifest.
    """

    root = Path(source_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"legacy response directory not found: {root}")
    if repeat_base not in {0, 1}:
        raise ValueError("repeat_base must be 0 or 1")
    aliases = dict(model_aliases or {})
    issues: list[LegacyAuditIssue] = []
    records: list[ResponseRecord] = []
    scanned_runs = 0
    parsed_runs = 0
    observed_models: set[str] = set()
    observed_settings: set[str] = set()
    seen_keys: set[Tuple[str, str, str, int]] = set()

    revisions = manifest.metadata.get("model_revisions")
    if manifest.metadata.get("allow_external_seed_records") is not True:
        issues.append(
            LegacyAuditIssue(
                "error",
                "external_records_not_allowed",
                "manifest.json",
                "manifest metadata must set allow_external_seed_records to true",
            )
        )
    if not isinstance(revisions, dict):
        revisions = {}
        issues.append(
            LegacyAuditIssue(
                "error",
                "model_revisions_missing",
                "manifest.json",
                "manifest metadata must pin a model_revisions object",
            )
        )
    for model_id in manifest.model_ids:
        revision = revisions.get(model_id)
        if not isinstance(revision, str) or _SHA1.fullmatch(revision) is None:
            issues.append(
                LegacyAuditIssue(
                    "error",
                    "model_revision_unpinned",
                    "manifest.json",
                    f"model {model_id!r} must be pinned to a lowercase 40-character SHA",
                )
            )

    prompt_text_to_id: Dict[str, str] = {}
    ambiguous_prompt_texts: set[str] = set()
    for prompt in manifest.prompts:
        if prompt.text in prompt_text_to_id:
            ambiguous_prompt_texts.add(prompt.text)
        prompt_text_to_id[prompt.text] = prompt.prompt_id
    for prompt_text in sorted(ambiguous_prompt_texts):
        issues.append(
            LegacyAuditIssue(
                "error",
                "ambiguous_manifest_prompt",
                "manifest.json",
                f"multiple prompt IDs share text with sha256={_text_sha256(prompt_text)}",
            )
        )

    model_lookup = _model_lookup(manifest.model_ids, aliases, issues)
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        scanned_runs += 1
        match = _RUN_NAME.fullmatch(run_dir.name)
        relative = run_dir.relative_to(root).as_posix()
        if match is None:
            issues.append(
                LegacyAuditIssue(
                    "warning",
                    "ignored_directory",
                    relative,
                    "directory name does not match *_tXX_pXX_rN",
                )
            )
            continue
        parsed_runs += 1
        model_id = model_lookup.get(match.group("model"))
        if model_id is None:
            issues.append(
                LegacyAuditIssue(
                    "error",
                    "unknown_model",
                    relative,
                    f"legacy model token {match.group('model')!r} is not mapped by the manifest",
                )
            )
            continue
        setting = _match_setting(
            manifest.settings,
            int(match.group("temperature")) / 10.0,
            int(match.group("top_p")) / 10.0,
        )
        if setting is None:
            issues.append(
                LegacyAuditIssue(
                    "error",
                    "unknown_setting",
                    relative,
                    "directory temperature/top-p does not uniquely match a manifest setting",
                )
            )
            continue
        generation_index = int(match.group("repeat")) - repeat_base
        if generation_index < 0 or generation_index >= manifest.generations:
            issues.append(
                LegacyAuditIssue(
                    "error",
                    "repeat_out_of_range",
                    relative,
                    f"repeat maps to generation_index={generation_index}, outside the manifest",
                )
            )
            continue
        response_path = run_dir / "responses.json"
        if not response_path.is_file():
            issues.append(
                LegacyAuditIssue(
                    "error", "responses_missing", relative, "responses.json is missing"
                )
            )
            continue
        try:
            payload = _load_payload(response_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(
                LegacyAuditIssue(
                    "error", "responses_invalid", relative, str(exc)
                )
            )
            continue
        source_sha256 = _file_sha256(response_path)
        top_metadata = payload.get("metadata")
        if not isinstance(top_metadata, dict):
            top_metadata = {}
        top_seed = _optional_integer(
            payload.get("generation_seed", payload.get("seed", top_metadata.get("generation_seed", top_metadata.get("seed"))))
        )
        top_revision = _optional_revision(
            payload.get("model_revision", payload.get("revision", top_metadata.get("model_revision", top_metadata.get("revision"))))
        )
        pinned_revision = revisions.get(model_id)
        items = payload["items"]
        seen_prompts_in_run: set[str] = set()
        for item_index, item in enumerate(items):
            item_path = f"{relative}/responses.json#items[{item_index}]"
            if not isinstance(item, dict):
                issues.append(
                    LegacyAuditIssue(
                        "error", "item_invalid", item_path, "item must be a JSON object"
                    )
                )
                continue
            prompt_text = item.get("prompt")
            response_text = item.get("response")
            if not isinstance(prompt_text, str) or not prompt_text:
                issues.append(
                    LegacyAuditIssue(
                        "error", "prompt_missing", item_path, "prompt must be non-empty"
                    )
                )
                continue
            prompt_id = prompt_text_to_id.get(prompt_text)
            if prompt_id is None or prompt_text in ambiguous_prompt_texts:
                issues.append(
                    LegacyAuditIssue(
                        "error",
                        "unknown_prompt",
                        item_path,
                        f"prompt text sha256={_text_sha256(prompt_text)} is not uniquely mapped",
                    )
                )
                continue
            if prompt_id in seen_prompts_in_run:
                issues.append(
                    LegacyAuditIssue(
                        "error", "duplicate_prompt", item_path, f"duplicate prompt_id={prompt_id}"
                    )
                )
                continue
            seen_prompts_in_run.add(prompt_id)
            if not isinstance(response_text, str) or not response_text.strip():
                issues.append(
                    LegacyAuditIssue(
                        "error", "response_empty", item_path, "response must be non-empty"
                    )
                )
                continue
            item_metadata = item.get("metadata")
            if not isinstance(item_metadata, dict):
                item_metadata = {}
            seed = _optional_integer(
                item.get("generation_seed", item.get("seed", item_metadata.get("generation_seed", item_metadata.get("seed", top_seed))))
            )
            provenance_valid = True
            if seed is None or seed < 0:
                issues.append(
                    LegacyAuditIssue(
                        "error",
                        "generation_seed_missing",
                        item_path,
                        "actual source generation_seed is required",
                    )
                )
                provenance_valid = False
            revision = _optional_revision(
                item.get("model_revision", item.get("revision", item_metadata.get("model_revision", item_metadata.get("revision", top_revision))))
            )
            if revision is None:
                issues.append(
                    LegacyAuditIssue(
                        "error",
                        "model_revision_missing",
                        item_path,
                        "exact source model_revision is required",
                    )
                )
                provenance_valid = False
            elif revision != pinned_revision:
                issues.append(
                    LegacyAuditIssue(
                        "error",
                        "model_revision_mismatch",
                        item_path,
                        "source revision does not match the pinned manifest revision",
                    )
                )
                provenance_valid = False
            if not provenance_valid:
                continue
            record = ResponseRecord(
                model_id=model_id,
                setting_id=setting.setting_id,
                prompt_id=prompt_id,
                generation_index=generation_index,
                generation_seed=seed,
                response=response_text,
                metadata={
                    "source_format": "llm-dna-responses-v1",
                    "source_path": response_path.relative_to(root).as_posix(),
                    "source_sha256": source_sha256,
                    "source_repeat": int(match.group("repeat")),
                    "model_revision": revision,
                },
                seed_scheme="external",
            )
            if record.key in seen_keys:
                issues.append(
                    LegacyAuditIssue(
                        "error", "duplicate_generation_key", item_path, repr(record.key)
                    )
                )
                continue
            seen_keys.add(record.key)
            records.append(record)
            observed_models.add(model_id)
            observed_settings.add(setting.setting_id)

    expected_keys = {
        (model_id, setting.setting_id, prompt.prompt_id, generation_index)
        for model_id in manifest.model_ids
        for setting in manifest.settings
        for prompt in manifest.prompts
        for generation_index in range(manifest.generations)
    }
    missing_keys = sorted(expected_keys.difference(seen_keys))
    if missing_keys:
        preview = ", ".join(repr(key) for key in missing_keys[:5])
        suffix = "" if len(missing_keys) <= 5 else f"; plus {len(missing_keys) - 5} more"
        issues.append(
            LegacyAuditIssue(
                "error",
                "missing_generation_keys",
                root.name,
                f"missing {len(missing_keys)} expected keys: {preview}{suffix}",
            )
        )
    return LegacyAuditReport(
        source_dir=str(root.resolve()),
        manifest_fingerprint=manifest.fingerprint,
        scanned_runs=scanned_runs,
        parsed_runs=parsed_runs,
        expected_records=len(expected_keys),
        candidate_records=len(records),
        missing_records=len(missing_keys),
        models=tuple(sorted(observed_models)),
        settings=tuple(sorted(observed_settings)),
        issues=tuple(issues),
        records=tuple(sorted(records, key=lambda record: record.key)),
    )


def import_llm_dna_responses(
    source_dir: str | Path,
    manifest: CollectionManifest,
    cache_dir: str | Path,
    model_aliases: Mapping[str, str] | None = None,
    repeat_base: int = 1,
) -> tuple[ResponseDataset, LegacyAuditReport]:
    """Import a complete, provenance-safe legacy collection into a new cache atomically."""

    report = audit_llm_dna_responses(
        source_dir, manifest, model_aliases=model_aliases, repeat_base=repeat_base
    )
    if not report.ready_for_import:
        raise ValueError(
            "legacy collection is not importable: "
            f"{report.error_count} errors, {report.missing_records} missing records"
        )
    target = Path(cache_dir)
    if target.exists():
        raise FileExistsError(f"legacy import cache already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent)
    )
    temporary_cache = temporary_root / "cache"
    try:
        cache = ResponseCache(temporary_cache, manifest)
        for record in report.records:
            cache.append(record)
        dataset = cache.dataset
        dataset.require_complete()
        os.replace(temporary_cache, target)
        os.rmdir(temporary_root)
        return ResponseCache(target, manifest).dataset, report
    except Exception:
        _remove_import_temporary(temporary_root)
        raise


def _model_lookup(
    model_ids: Iterable[str],
    aliases: Mapping[str, str],
    issues: list[LegacyAuditIssue],
) -> Dict[str, str]:
    model_ids = tuple(model_ids)
    known = set(model_ids)
    lookup: Dict[str, str] = {}
    for model_id in model_ids:
        for token in {model_id, re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.strip("/"))}:
            if token in lookup and lookup[token] != model_id:
                issues.append(
                    LegacyAuditIssue(
                        "error",
                        "model_token_collision",
                        "manifest.json",
                        f"legacy token {token!r} maps to multiple models",
                    )
                )
            else:
                lookup[token] = model_id
    for token, model_id in aliases.items():
        if model_id not in known:
            issues.append(
                LegacyAuditIssue(
                    "error",
                    "alias_target_unknown",
                    "model-aliases.json",
                    f"alias {token!r} targets unknown model {model_id!r}",
                )
            )
            continue
        lookup[token] = model_id
    return lookup


def _match_setting(
    settings: Iterable[DecodingSetting], temperature: float, top_p: float
) -> DecodingSetting | None:
    matches = [
        setting
        for setting in settings
        if abs(setting.temperature - temperature) <= 1e-12
        and abs(setting.top_p - top_p) <= 1e-12
    ]
    return matches[0] if len(matches) == 1 else None


def _load_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("responses.json root must be an object")
    if not isinstance(payload.get("items"), list):
        raise ValueError("responses.json must contain an items array")
    return payload


def _optional_integer(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _optional_revision(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.lower()
    return normalized if _SHA1.fullmatch(normalized) is not None else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}-", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
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


def _remove_import_temporary(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()
