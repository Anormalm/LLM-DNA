"""Visible, resumable stochastic response caches."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Protocol, Tuple, Union

from .manifest import CollectionManifest, DecodingSetting, Prompt


def generation_seed(
    base_seed: int,
    model_id: str,
    setting_id: str,
    prompt_id: str,
    generation_index: int,
) -> int:
    value = f"{base_seed}\0{model_id}\0{setting_id}\0{prompt_id}\0{generation_index}"
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


@dataclass(frozen=True)
class GeneratedResponse:
    """Provider output plus JSON-serializable generation provenance."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("generated response text must be non-empty")
        if not isinstance(self.metadata, dict):
            raise ValueError("generated response metadata must be a JSON object")
        try:
            json.dumps(self.metadata, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("generated response metadata must be JSON serializable") from exc


@dataclass(frozen=True)
class ResponseRecord:
    model_id: str
    setting_id: str
    prompt_id: str
    generation_index: int
    generation_seed: int
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    seed_scheme: str = "distdna-v1"

    def __post_init__(self) -> None:
        if not self.model_id or not self.setting_id or not self.prompt_id:
            raise ValueError("response record identifiers must be non-empty")
        if self.generation_index < 0:
            raise ValueError("generation_index must be non-negative")
        if (
            not isinstance(self.generation_seed, int)
            or isinstance(self.generation_seed, bool)
            or self.generation_seed < 0
        ):
            raise ValueError("generation_seed must be a non-negative integer")
        if not isinstance(self.response, str) or not self.response.strip():
            raise ValueError("response must be a non-empty string")
        if not isinstance(self.metadata, dict):
            raise ValueError("response metadata must be a JSON object")
        try:
            json.dumps(self.metadata, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("response metadata must be JSON serializable") from exc
        if self.seed_scheme not in {"distdna-v1", "external"}:
            raise ValueError("seed_scheme must be 'distdna-v1' or 'external'")
        if self.seed_scheme == "external":
            required = {
                "source_format",
                "source_path",
                "source_sha256",
                "model_revision",
            }
            missing = sorted(required.difference(self.metadata))
            if missing:
                raise ValueError(
                    f"external response metadata is missing provenance keys: {missing}"
                )
            if self.metadata["source_format"] != "llm-dna-responses-v1":
                raise ValueError("unsupported external response source_format")
            source_path = self.metadata["source_path"]
            if not isinstance(source_path, str) or not source_path:
                raise ValueError("external response source_path must be non-empty")
            source_sha256 = self.metadata["source_sha256"]
            if (
                not isinstance(source_sha256, str)
                or len(source_sha256) != 64
                or any(character not in "0123456789abcdef" for character in source_sha256)
            ):
                raise ValueError("external response source_sha256 must be lowercase hex")
            model_revision = self.metadata["model_revision"]
            if (
                not isinstance(model_revision, str)
                or len(model_revision) != 40
                or any(character not in "0123456789abcdef" for character in model_revision)
            ):
                raise ValueError("external response model_revision must be a 40-character SHA")

    @property
    def key(self) -> Tuple[str, str, str, int]:
        return (self.model_id, self.setting_id, self.prompt_id, self.generation_index)


class ResponseGenerator(Protocol):
    def generate(
        self,
        model_id: str,
        prompt: Prompt,
        setting: DecodingSetting,
        seed: int,
    ) -> Union[str, GeneratedResponse]:
        ...


class CallableResponseGenerator:
    def __init__(
        self,
        function: Callable[[str, Prompt, DecodingSetting, int], str],
    ) -> None:
        self.function = function

    def generate(
        self,
        model_id: str,
        prompt: Prompt,
        setting: DecodingSetting,
        seed: int,
    ) -> Union[str, GeneratedResponse]:
        return self.function(model_id, prompt, setting, seed)


@dataclass(frozen=True)
class ResponseDataset:
    manifest: CollectionManifest
    records: Tuple[ResponseRecord, ...]

    def __post_init__(self) -> None:
        keys = [record.key for record in self.records]
        if len(set(keys)) != len(keys):
            raise ValueError("response cache contains duplicate generation keys")
        model_ids = set(self.manifest.model_ids)
        setting_ids = set(self.manifest.setting_ids)
        prompt_ids = {prompt.prompt_id for prompt in self.manifest.prompts}
        for record in self.records:
            if record.model_id not in model_ids:
                raise ValueError(f"response uses unknown model_id: {record.model_id}")
            if record.setting_id not in setting_ids:
                raise ValueError(f"response uses unknown setting_id: {record.setting_id}")
            if record.prompt_id not in prompt_ids:
                raise ValueError(f"response uses unknown prompt_id: {record.prompt_id}")
            if record.generation_index >= self.manifest.generations:
                raise ValueError("response generation_index exceeds manifest generations")
            expected_seed = generation_seed(
                self.manifest.random_seed,
                record.model_id,
                record.setting_id,
                record.prompt_id,
                record.generation_index,
            )
            if (
                record.seed_scheme == "distdna-v1"
                and record.generation_seed != expected_seed
            ):
                raise ValueError(f"response has unexpected seed for key: {record.key}")
            if record.seed_scheme == "external":
                if self.manifest.metadata.get("allow_external_seed_records") is not True:
                    raise ValueError(
                        "manifest must explicitly allow external seed records"
                    )
                revisions = self.manifest.metadata.get("model_revisions")
                if not isinstance(revisions, dict):
                    raise ValueError(
                        "external response manifest must pin model_revisions"
                    )
                if revisions.get(record.model_id) != record.metadata["model_revision"]:
                    raise ValueError(
                        f"external response revision does not match manifest: {record.model_id}"
                    )

    @property
    def by_key(self) -> Dict[Tuple[str, str, str, int], ResponseRecord]:
        return {record.key: record for record in self.records}

    @property
    def expected_count(self) -> int:
        return (
            len(self.manifest.model_ids)
            * len(self.manifest.settings)
            * len(self.manifest.prompts)
            * self.manifest.generations
        )

    @property
    def complete(self) -> bool:
        return len(self.records) == self.expected_count

    def require_complete(self) -> None:
        if not self.complete:
            raise ValueError(
                "response cache is incomplete: "
                f"found {len(self.records)} of {self.expected_count} generations"
            )


class ResponseCache:
    """Append-only JSONL cache with a pinned collection manifest."""

    def __init__(self, directory: str | Path, manifest: CollectionManifest) -> None:
        self.directory = Path(directory)
        self.manifest_path = self.directory / "manifest.json"
        self.records_path = self.directory / "responses.jsonl"
        if self.directory.exists():
            if not self.manifest_path.is_file():
                raise ValueError(f"response cache has no manifest: {self.directory}")
            existing = CollectionManifest.load(self.manifest_path)
            if existing.fingerprint != manifest.fingerprint:
                raise ValueError("response cache manifest does not match the requested collection")
        else:
            self.directory.mkdir(parents=True)
            manifest.save(self.manifest_path)
        self.manifest = manifest
        loaded = self.load()
        self._records = list(loaded.records)
        self._keys = set(loaded.by_key)

    def load(self) -> ResponseDataset:
        records = []
        if self.records_path.is_file():
            with self.records_path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                        records.append(ResponseRecord(**payload))
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            f"invalid response cache record at line {line_number}: {exc}"
                        ) from exc
        return ResponseDataset(self.manifest, tuple(records))

    def append(self, record: ResponseRecord) -> None:
        if record.key in self._keys:
            raise ValueError(f"response key already exists: {record.key}")
        # Validate membership and deterministic seed before mutating the cache.
        ResponseDataset(self.manifest, (record,))
        with self.records_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), sort_keys=True, ensure_ascii=False))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._records.append(record)
        self._keys.add(record.key)

    @property
    def dataset(self) -> ResponseDataset:
        """Return the in-memory cache view without re-reading the JSONL file."""

        return ResponseDataset(self.manifest, tuple(self._records))


def expected_response_keys(
    manifest: CollectionManifest,
) -> Iterator[Tuple[str, str, str, int]]:
    for model_id in manifest.model_ids:
        for setting in manifest.settings:
            for prompt in manifest.prompts:
                for generation_index in range(manifest.generations):
                    yield (
                        model_id,
                        setting.setting_id,
                        prompt.prompt_id,
                        generation_index,
                    )


def collect_responses(
    manifest: CollectionManifest,
    generator: ResponseGenerator,
    cache_dir: str | Path,
    progress: Callable[[int, int, ResponseRecord], None] | None = None,
) -> ResponseDataset:
    """Collect every missing response and safely resume an existing cache."""

    cache = ResponseCache(cache_dir, manifest)
    existing = cache.dataset.by_key
    prompts = {prompt.prompt_id: prompt for prompt in manifest.prompts}
    settings = {setting.setting_id: setting for setting in manifest.settings}
    total = cache.dataset.expected_count
    completed = len(existing)
    for model_id, setting_id, prompt_id, generation_index in expected_response_keys(manifest):
        key = (model_id, setting_id, prompt_id, generation_index)
        if key in existing:
            continue
        seed = generation_seed(
            manifest.random_seed,
            model_id,
            setting_id,
            prompt_id,
            generation_index,
        )
        generated = generator.generate(
            model_id=model_id,
            prompt=prompts[prompt_id],
            setting=settings[setting_id],
            seed=seed,
        )
        if isinstance(generated, GeneratedResponse):
            response = generated.text
            metadata = generated.metadata
        elif isinstance(generated, str):
            response = generated
            metadata = {}
        else:
            raise TypeError(
                "response generator must return str or GeneratedResponse, received "
                f"{type(generated).__name__}"
            )
        record = ResponseRecord(
            model_id=model_id,
            setting_id=setting_id,
            prompt_id=prompt_id,
            generation_index=generation_index,
            generation_seed=seed,
            response=response,
            metadata=metadata,
        )
        cache.append(record)
        existing[key] = record
        completed += 1
        if progress is not None:
            progress(completed, total, record)
    result = cache.dataset
    result.require_complete()
    return result
