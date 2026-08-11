"""RFFTrace extractor with an LLM-DNA-like object boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..data import normalize_embeddings
from ..features import RFFTrace
from .signature import DNACollection, DNAMetadata, DNASignature


class RFFTraceExtractor:
    """Extract model DNA vectors using one fixed shared RFFTrace map."""

    def __init__(
        self,
        embedding_dim: int,
        prompt_count: int,
        rff_dimension: int,
        sigma: float,
        random_seed: int = 42,
        dna_dimension: int | None = None,
        normalization: str = "l2",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.prompt_count = prompt_count
        self.rff_dimension = rff_dimension
        self.sigma = sigma
        self.random_seed = random_seed
        self.dna_dimension = dna_dimension
        self.normalization = normalization
        self.feature_map = RFFTrace(
            input_dim=embedding_dim,
            n_features=rff_dimension,
            sigma=sigma,
            n_prompts=prompt_count,
            seed=random_seed,
            projection_dim=dna_dimension,
        )

    @property
    def output_dimension(self) -> int:
        return self.dna_dimension or self.prompt_count * self.rff_dimension

    def extract_dna(
        self,
        embeddings: NDArray[np.floating],
        model_names: Sequence[str],
        setting_id: str,
    ) -> DNACollection:
        values = normalize_embeddings(np.asarray(embeddings), self.normalization)
        if values.ndim != 4:
            raise ValueError(
                "embeddings must have shape [models, prompts, generations, features]"
            )
        if values.shape[0] != len(model_names):
            raise ValueError("model_names length does not match the embedding tensor")
        vectors = self.feature_map.transform(values)
        extraction_time = datetime.now(timezone.utc).isoformat()
        config = {
            "shared_feature_map": True,
            "rff_dimension": self.rff_dimension,
            "rbf_sigma": self.sigma,
            "normalization": self.normalization,
            "projection_dimension": self.dna_dimension,
        }
        return DNACollection(
            DNASignature(
                vector,
                DNAMetadata(
                    model_name=model_name,
                    setting_id=setting_id,
                    extraction_method="rfftrace",
                    probe_count=values.shape[1],
                    generation_count=values.shape[2],
                    dna_dimension=self.output_dimension,
                    embedding_dimension=values.shape[3],
                    rff_dimension=self.rff_dimension,
                    rbf_sigma=self.sigma,
                    normalization=self.normalization,
                    random_seed=self.random_seed,
                    projection_dimension=self.dna_dimension,
                    extraction_time=extraction_time,
                    extractor_config=config,
                ),
            )
            for model_name, vector in zip(model_names, vectors)
        )

    def save_parameters(self, path: str | Path) -> None:
        self.feature_map.save(path)

