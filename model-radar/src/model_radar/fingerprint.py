from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from .utils import cosine_similarity, json_hash, normalize_vector


TOKEN_RE = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)


class FingerprintBackend(Protocol):
    name: str
    dimension: int

    def fingerprint(self, responses: Sequence[str]) -> np.ndarray: ...

    def key(self, probe_set_hash: str) -> str: ...


@dataclass(slots=True)
class HashFingerprint:
    """Dependency-free feature-hashing backend for fast monitoring and CI demos."""

    dimension: int = 256
    seed: int = 42
    name: str = "hash-v1"

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("Fingerprint dimension must be positive")

    def key(self, probe_set_hash: str) -> str:
        return f"{self.name}:d{self.dimension}:s{self.seed}:p{probe_set_hash[:16]}"

    def fingerprint(self, responses: Sequence[str]) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float64)
        for prompt_index, response in enumerate(responses):
            normalized = " ".join(str(response).lower().split())
            tokens = TOKEN_RE.findall(normalized)
            features: list[tuple[str, float]] = []
            features.extend((f"w:{token}", 1.0) for token in tokens)
            features.extend(
                (f"b:{left}\u241f{right}", 1.25) for left, right in zip(tokens, tokens[1:])
            )
            compact = normalized[:2000]
            features.extend((f"c3:{compact[i:i+3]}", 0.2) for i in range(max(0, len(compact) - 2)))
            if not features:
                features.append(("<empty>", 1.0))
            for feature, weight in features:
                key = f"{self.seed}|{prompt_index}|{feature}".encode("utf-8")
                digest = hashlib.blake2b(key, digest_size=16).digest()
                bucket = int.from_bytes(digest[:8], "little") % self.dimension
                sign = 1.0 if digest[8] & 1 else -1.0
                vector[bucket] += sign * weight
        return normalize_vector(vector)


@dataclass(slots=True)
class RepTraceFingerprint:
    """RepTrace-compatible shared random projection over ordered response embeddings."""

    dimension: int = 256
    seed: int = 42
    sentence_encoder: str = "sentence-transformers/all-mpnet-base-v2"
    name: str = "reptrace-rp-v1"
    _encoder: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("Fingerprint dimension must be positive")

    def key(self, probe_set_hash: str) -> str:
        encoder_hash = hashlib.sha256(self.sentence_encoder.encode("utf-8")).hexdigest()[:12]
        return (
            f"{self.name}:d{self.dimension}:s{self.seed}:e{encoder_hash}:"
            f"p{probe_set_hash[:16]}"
        )

    def fingerprint(self, responses: Sequence[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "RepTrace backend requires: pip install 'model-radar[reptrace]'"
            ) from exc
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.sentence_encoder)
        embeddings = self._encoder.encode(
            list(responses),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        flat = np.asarray(embeddings, dtype=np.float32).reshape(-1)
        if not np.any(flat):
            return np.zeros(self.dimension, dtype=np.float64)

        # A single deterministic Gaussian map is regenerated from the same seed for every model.
        # Chunking avoids materializing the full (probe_count * embed_dim) x DNA matrix.
        rng = np.random.default_rng(self.seed)
        projected = np.zeros(self.dimension, dtype=np.float64)
        chunk_size = 8192
        scale = 1.0 / math.sqrt(self.dimension)
        for start in range(0, len(flat), chunk_size):
            chunk = flat[start : start + chunk_size]
            matrix = rng.standard_normal((len(chunk), self.dimension), dtype=np.float32)
            projected += np.asarray(chunk @ matrix, dtype=np.float64) * scale
        return normalize_vector(projected)


def create_backend(config: dict[str, Any]) -> FingerprintBackend:
    backend = str(config.get("backend", "hash")).lower()
    common = {
        "dimension": int(config.get("dimension", 256)),
        "seed": int(config.get("seed", 42)),
    }
    if backend in {"hash", "hash-v1"}:
        return HashFingerprint(**common)
    if backend in {"reptrace", "llm-dna", "reptrace-rp"}:
        return RepTraceFingerprint(
            **common,
            sentence_encoder=str(
                config.get("sentence_encoder", "sentence-transformers/all-mpnet-base-v2")
            ),
        )
    raise ValueError(f"Unknown fingerprint backend: {backend}")


def fingerprint_repetitions(
    response_items: Sequence[dict[str, Any]],
    backend: FingerprintBackend,
    *,
    prompt_count: int,
    repetitions: int,
) -> tuple[np.ndarray, float, list[np.ndarray]]:
    vectors: list[np.ndarray] = []
    by_key = {
        (int(item["repetition"]), int(item["prompt_index"])): str(item.get("response", ""))
        for item in response_items
    }
    for repetition in range(repetitions):
        ordered = [by_key.get((repetition, index), "") for index in range(prompt_count)]
        vectors.append(backend.fingerprint(ordered))

    aggregate = normalize_vector(np.mean(np.stack(vectors), axis=0))
    distances: list[float] = []
    for left_index in range(len(vectors)):
        for right_index in range(left_index + 1, len(vectors)):
            distances.append(1.0 - cosine_similarity(vectors[left_index], vectors[right_index]))
    within = float(np.mean(distances)) if distances else 0.0
    return aggregate, within, vectors


def probe_set_identity(probe_payload: Any) -> tuple[str, str, list[str]]:
    if isinstance(probe_payload, list):
        probes = [str(item) for item in probe_payload]
        probe_set_id = "custom"
    elif isinstance(probe_payload, dict) and isinstance(probe_payload.get("probes"), list):
        probes = [str(item) for item in probe_payload["probes"]]
        probe_set_id = str(probe_payload.get("id") or "custom")
    else:
        raise ValueError("Probe file must be a JSON list or an object containing a probes list")
    if not probes or any(not probe.strip() for probe in probes):
        raise ValueError("Probe set must contain non-empty strings")
    return probe_set_id, json_hash(probes), probes
