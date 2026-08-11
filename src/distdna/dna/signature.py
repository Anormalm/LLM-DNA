"""DNA vector and metadata containers inspired by LLM-DNA's public objects."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DNAMetadata:
    model_name: str
    setting_id: str
    extraction_method: str
    probe_count: int
    generation_count: int
    dna_dimension: int
    embedding_dimension: int
    rff_dimension: int
    rbf_sigma: float
    normalization: str
    random_seed: int
    projection_dimension: int | None
    extraction_time: str
    extractor_config: Dict[str, Any]


class DNASignature:
    """One model-setting RFFTrace vector with auditable extraction metadata."""

    def __init__(
        self, signature: NDArray[np.floating] | Sequence[float], metadata: DNAMetadata
    ) -> None:
        vector = np.asarray(signature, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("DNA signature must be a non-empty one-dimensional vector")
        if not np.isfinite(vector).all():
            raise ValueError("DNA signature must contain only finite values")
        if vector.size != metadata.dna_dimension:
            raise ValueError(
                "signature dimension does not match metadata: "
                f"{vector.size} != {metadata.dna_dimension}"
            )
        self.signature = vector
        self.metadata = metadata

    @property
    def vector(self) -> NDArray[np.float32]:
        return self.signature

    @property
    def model_name(self) -> str:
        return self.metadata.model_name

    @property
    def setting_id(self) -> str:
        return self.metadata.setting_id

    @property
    def dimension(self) -> int:
        return int(self.signature.size)

    def __len__(self) -> int:
        return self.dimension

    def distance_to(self, other: "DNASignature", metric: str = "euclidean") -> float:
        if self.signature.shape != other.signature.shape:
            raise ValueError("DNA signatures must have the same dimension")
        delta = self.signature.astype(np.float64) - other.signature.astype(np.float64)
        if metric == "squared_euclidean":
            return float(np.sum(delta * delta))
        if metric == "euclidean":
            return math.sqrt(float(np.sum(delta * delta)))
        if metric == "cosine":
            left_norm = float(np.linalg.norm(self.signature))
            right_norm = float(np.linalg.norm(other.signature))
            if left_norm == 0.0 or right_norm == 0.0:
                raise ValueError("cosine distance is undefined for zero DNA vectors")
            dot = float(
                np.sum(
                    self.signature.astype(np.float64)
                    * other.signature.astype(np.float64)
                )
            )
            return 1.0 - max(-1.0, min(1.0, dot / (left_norm * right_norm)))
        raise ValueError(f"unsupported DNA distance metric: {metric}")

    def save(self, path: str | Path, format: str | None = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        selected = (format or target.suffix.lstrip(".") or "npz").lower()
        metadata_json = json.dumps(asdict(self.metadata), sort_keys=True)
        if selected == "npz":
            np.savez_compressed(
                target,
                signature=self.signature,
                metadata_json=np.asarray(metadata_json, dtype=np.str_),
            )
        elif selected == "json":
            with target.open("w", encoding="utf-8") as stream:
                json.dump(
                    {"signature": self.signature.tolist(), "metadata": asdict(self.metadata)},
                    stream,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
        elif selected == "csv":
            with target.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["dna_value"])
                writer.writerows((float(value),) for value in self.signature)
        else:
            raise ValueError("DNA signature format must be npz, json, or csv")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "DNASignature":
        source = Path(path)
        if source.suffix.lower() == ".npz":
            with np.load(source, allow_pickle=False) as payload:
                vector = np.asarray(payload["signature"], dtype=np.float32)
                metadata = json.loads(str(payload["metadata_json"]))
        elif source.suffix.lower() == ".json":
            with source.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            vector = np.asarray(payload["signature"], dtype=np.float32)
            metadata = payload["metadata"]
        else:
            raise ValueError("loading DNA signatures supports npz and json")
        return cls(vector, DNAMetadata(**metadata))


class DNACollection:
    def __init__(self, signatures: Iterable[DNASignature] = ()) -> None:
        self.signatures = tuple(signatures)
        if self.signatures:
            dimensions = {signature.dimension for signature in self.signatures}
            if len(dimensions) != 1:
                raise ValueError("all DNA signatures in a collection must share a dimension")
            keys = [
                (signature.model_name, signature.setting_id)
                for signature in self.signatures
            ]
            if len(set(keys)) != len(keys):
                raise ValueError("DNA collection model-setting keys must be unique")

    def __len__(self) -> int:
        return len(self.signatures)

    def __iter__(self) -> Iterator[DNASignature]:
        return iter(self.signatures)

    @property
    def vectors(self) -> NDArray[np.float32]:
        if not self.signatures:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([signature.vector for signature in self.signatures])

    @property
    def model_names(self) -> Tuple[str, ...]:
        return tuple(signature.model_name for signature in self.signatures)

    @property
    def setting_ids(self) -> Tuple[str, ...]:
        return tuple(signature.setting_id for signature in self.signatures)

    def save(self, path: str | Path) -> Path:
        if not self.signatures:
            raise ValueError("cannot save an empty DNA collection")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = np.asarray(
            [json.dumps(asdict(signature.metadata), sort_keys=True) for signature in self.signatures],
            dtype=np.str_,
        )
        np.savez_compressed(
            target,
            signatures=self.vectors,
            model_names=np.asarray(self.model_names, dtype=np.str_),
            setting_ids=np.asarray(self.setting_ids, dtype=np.str_),
            metadata_json=metadata,
        )
        return target
