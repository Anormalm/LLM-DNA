"""Validated, provenance-linked prompt revisions for collection manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping

from .manifest import CollectionManifest, Prompt


@dataclass(frozen=True)
class PromptRevisionSet:
    """A versioned mapping from stable prompt IDs to revised prompt text."""

    revision_id: str
    reason: str
    revisions: Mapping[str, str]
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.format_version != 1:
            raise ValueError(
                f"unsupported prompt revision format_version: {self.format_version}"
            )
        if not isinstance(self.revision_id, str) or not self.revision_id.strip():
            raise ValueError("revision_id must be a non-empty string")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        if not isinstance(self.revisions, Mapping) or not self.revisions:
            raise ValueError("revisions must be a non-empty JSON object")
        normalized: Dict[str, str] = {}
        for prompt_id, text in self.revisions.items():
            if not isinstance(prompt_id, str) or not prompt_id.strip():
                raise ValueError("revision prompt IDs must be non-empty strings")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"revised prompt text must be non-empty for {prompt_id!r}"
                )
            normalized[prompt_id] = text.strip()
        object.__setattr__(self, "revision_id", self.revision_id.strip())
        object.__setattr__(self, "reason", self.reason.strip())
        object.__setattr__(self, "revisions", normalized)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "revision_id": self.revision_id,
            "reason": self.reason,
            "revisions": dict(self.revisions),
        }

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "PromptRevisionSet":
        source = Path(path)
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("prompt revision root must be a JSON object")
        allowed = {"format_version", "revision_id", "reason", "revisions"}
        unknown = sorted(set(payload).difference(allowed))
        if unknown:
            raise ValueError(f"unknown prompt revision keys: {unknown}")
        required = {"revision_id", "reason", "revisions"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"prompt revision file is missing keys: {missing}")
        revisions = payload["revisions"]
        if not isinstance(revisions, dict):
            raise ValueError("revisions must be a JSON object")
        return cls(
            format_version=payload.get("format_version", 1),
            revision_id=payload["revision_id"],
            reason=payload["reason"],
            revisions=revisions,
        )


def apply_prompt_revisions(
    manifest: CollectionManifest,
    revision_set: PromptRevisionSet,
    *,
    dataset_id: str,
    require_all_prompts: bool = False,
) -> CollectionManifest:
    """Apply text-only revisions while preserving prompt IDs, splits, and protocol."""

    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string")
    source_by_id = {prompt.prompt_id: prompt for prompt in manifest.prompts}
    revised_ids = set(revision_set.revisions)
    source_ids = set(source_by_id)
    unknown = sorted(revised_ids.difference(source_ids))
    if unknown:
        raise ValueError(f"prompt revisions contain unknown prompt IDs: {unknown}")
    if require_all_prompts:
        missing = sorted(source_ids.difference(revised_ids))
        if missing:
            raise ValueError(f"prompt revisions are missing prompt IDs: {missing}")
    unchanged = sorted(
        prompt_id
        for prompt_id, text in revision_set.revisions.items()
        if source_by_id[prompt_id].text == text
    )
    if unchanged:
        raise ValueError(f"prompt revisions do not change prompt text: {unchanged}")

    prompts = tuple(
        Prompt(
            prompt_id=prompt.prompt_id,
            text=revision_set.revisions.get(prompt.prompt_id, prompt.text),
            split=prompt.split,
        )
        for prompt in manifest.prompts
    )
    texts = [prompt.text for prompt in prompts]
    if len(set(texts)) != len(texts):
        raise ValueError("revised prompt texts must remain unique")

    metadata = dict(manifest.metadata)
    metadata["parent_manifest_fingerprint"] = manifest.fingerprint
    metadata["parent_dataset_id"] = manifest.dataset_id
    metadata["prompt_revision"] = {
        "revision_id": revision_set.revision_id,
        "reason": revision_set.reason,
        "fingerprint": revision_set.fingerprint,
        "changed_prompt_ids": sorted(revised_ids),
    }
    return replace(
        manifest,
        dataset_id=dataset_id.strip(),
        prompts=prompts,
        metadata=metadata,
    )
