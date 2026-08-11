"""Dataset loading facade matching the LLM-DNA package boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dataset import EmbeddingDataset


@dataclass(frozen=True)
class DatasetConfig:
    path: Path


class DatasetLoader:
    """Load validated response-embedding datasets.

    Text generation and response encoding will plug in behind this boundary. The
    current implementation deliberately accepts the stable NPZ interchange format.
    """

    def load(self, config: DatasetConfig | str | Path) -> EmbeddingDataset:
        path = config.path if isinstance(config, DatasetConfig) else Path(config)
        return EmbeddingDataset.load(path)

