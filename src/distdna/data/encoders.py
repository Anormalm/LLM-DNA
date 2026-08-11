"""Response encoder adapters."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray


class ResponseEncoder(Protocol):
    @property
    def encoder_id(self) -> str:
        ...

    def encode(self, responses: Sequence[str]) -> NDArray[np.floating]:
        ...


class HashingResponseEncoder:
    """Dependency-free deterministic encoder for tests and pipeline diagnostics only."""

    def __init__(self, dimension: int = 128) -> None:
        if dimension <= 0:
            raise ValueError("hashing encoder dimension must be positive")
        self.dimension = dimension

    @property
    def encoder_id(self) -> str:
        return f"diagnostic-hashing-{self.dimension}"

    def encode(self, responses: Sequence[str]) -> NDArray[np.float32]:
        output = np.zeros((len(responses), self.dimension), dtype=np.float32)
        for row, response in enumerate(responses):
            if not isinstance(response, str) or not response.strip():
                raise ValueError("encoder inputs must be non-empty response strings")
            tokens = re.findall(r"\w+|[^\w\s]", response.lower(), flags=re.UNICODE)
            features = tokens + [f"{left}::{right}" for left, right in zip(tokens, tokens[1:])]
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
                index = int.from_bytes(digest[:8], "big") % self.dimension
                sign = 1.0 if digest[8] & 1 else -1.0
                output[row, index] += sign
            norm = float(np.linalg.norm(output[row]))
            if norm == 0.0:
                raise ValueError("hashing encoder produced a zero response embedding")
            output[row] /= norm
        return output


class SentenceTransformerResponseEncoder:
    """Lazy sentence-transformers adapter for real response embeddings."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        if not model_name:
            raise ValueError("sentence encoder model_name must be non-empty")
        if batch_size <= 0:
            raise ValueError("sentence encoder batch_size must be positive")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    @property
    def encoder_id(self) -> str:
        return self.model_name

    def encode(self, responses: Sequence[str]) -> NDArray[np.float32]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required; install distdna[embedding]"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        values = self._model.encode(
            list(responses),
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
            normalize_embeddings=False,
        )
        output = np.asarray(values, dtype=np.float32)
        if output.ndim != 2 or output.shape[0] != len(responses):
            raise ValueError("sentence encoder returned an invalid tensor shape")
        if not np.isfinite(output).all():
            raise ValueError("sentence encoder returned non-finite values")
        return output

