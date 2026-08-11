"""Validated storage contract for response embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


def _string_tuple(values: NDArray[np.generic], name: str) -> Tuple[str, ...]:
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if values.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must use a non-object string dtype")
    decoded = tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values.tolist()
    )
    if any(not value for value in decoded):
        raise ValueError(f"{name} cannot contain empty identifiers")
    if len(set(decoded)) != len(decoded):
        raise ValueError(f"{name} must contain unique identifiers")
    return decoded


@dataclass(frozen=True)
class EmbeddingDataset:
    """A dense response-embedding tensor.

    The canonical shape is ``[models, settings, prompts, generations, features]``.
    Dense, finite inputs are intentional: missing generations must be resolved before
    evaluation so an absent value can never silently become a reported NA cell.
    """

    embeddings: NDArray[np.floating]
    model_ids: Tuple[str, ...]
    setting_ids: Tuple[str, ...]
    prompt_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.embeddings)
        if values.ndim != 5:
            raise ValueError(
                "embeddings must have shape "
                "[models, settings, prompts, generations, features]"
            )
        if values.dtype.kind != "f":
            raise ValueError("embeddings must use a floating-point dtype")
        if any(size <= 0 for size in values.shape):
            raise ValueError("every embeddings dimension must be non-zero")
        if not np.isfinite(values).all():
            raise ValueError("embeddings must contain only finite values")

        expected = values.shape[:3]
        actual = (len(self.model_ids), len(self.setting_ids), len(self.prompt_ids))
        if actual != expected:
            raise ValueError(
                "identifier lengths do not match embeddings: "
                f"expected {expected}, received {actual}"
            )
        for name, identifiers in (
            ("model_ids", self.model_ids),
            ("setting_ids", self.setting_ids),
            ("prompt_ids", self.prompt_ids),
        ):
            if any(not isinstance(value, str) or not value for value in identifiers):
                raise ValueError(f"{name} must contain non-empty strings")
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"{name} must contain unique identifiers")

        object.__setattr__(self, "embeddings", values)
        object.__setattr__(self, "model_ids", tuple(self.model_ids))
        object.__setattr__(self, "setting_ids", tuple(self.setting_ids))
        object.__setattr__(self, "prompt_ids", tuple(self.prompt_ids))

    @property
    def shape(self) -> Tuple[int, int, int, int, int]:
        return self.embeddings.shape  # type: ignore[return-value]

    @property
    def feature_dim(self) -> int:
        return int(self.embeddings.shape[-1])

    @property
    def generation_count(self) -> int:
        return int(self.embeddings.shape[-2])

    def setting(
        self, setting_id: str, generations: int | None = None
    ) -> NDArray[np.floating]:
        """Return ``[models, prompts, generations, features]`` for one setting."""

        try:
            index = self.setting_ids.index(setting_id)
        except ValueError as exc:
            raise KeyError(f"unknown setting_id: {setting_id}") from exc
        count = self.generation_count if generations is None else generations
        if count <= 0 or count > self.generation_count:
            raise ValueError(
                f"generations must be in [1, {self.generation_count}], received {count}"
            )
        return self.embeddings[:, index, :, :count, :]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            embeddings=self.embeddings,
            model_ids=np.asarray(self.model_ids, dtype=np.str_),
            setting_ids=np.asarray(self.setting_ids, dtype=np.str_),
            prompt_ids=np.asarray(self.prompt_ids, dtype=np.str_),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingDataset":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"embedding dataset not found: {source}")
        required = {"embeddings", "model_ids", "setting_ids", "prompt_ids"}
        with np.load(source, allow_pickle=False) as payload:
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(f"dataset is missing arrays: {sorted(missing)}")
            embeddings = np.asarray(payload["embeddings"])
            model_ids = _string_tuple(payload["model_ids"], "model_ids")
            setting_ids = _string_tuple(payload["setting_ids"], "setting_ids")
            prompt_ids = _string_tuple(payload["prompt_ids"], "prompt_ids")
        return cls(embeddings, model_ids, setting_ids, prompt_ids)


def normalize_embeddings(
    embeddings: NDArray[np.floating], mode: str
) -> NDArray[np.floating]:
    """Normalize individual response embeddings without mutating the input."""

    values = np.asarray(embeddings)
    if mode == "none":
        return values
    if mode != "l2":
        raise ValueError(f"unsupported normalization mode: {mode}")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot L2-normalize a zero response embedding")
    return values / norms


def require_disjoint_prompts(
    calibration_prompt_ids: Sequence[str], evaluation_prompt_ids: Sequence[str]
) -> None:
    overlap = sorted(set(calibration_prompt_ids).intersection(evaluation_prompt_ids))
    if overlap:
        preview = ", ".join(overlap[:5])
        suffix = "..." if len(overlap) > 5 else ""
        raise ValueError(
            "calibration and evaluation prompt IDs must be disjoint; "
            f"overlap: {preview}{suffix}"
        )

